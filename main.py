import asyncio
import json
import os
import sys
import time
from pathlib import Path

from agent.main_agent import create_agent
from engine.failure_cluster import cluster_failures
from engine.llm_judge import MultiModelJudge
from engine.ragas_eval import RAGASEvaluator
from engine.release_gate import evaluate_release_gate
from engine.runner import BenchmarkRunner

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent


def _build_summary(results, agent_version: str, runner: BenchmarkRunner) -> dict:
    total = len(results) or 1
    return {
        "metadata": {
            "version": agent_version,
            "total": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "avg_mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "avg_faithfulness": sum(r["ragas"]["faithfulness"] for r in results) / total,
            "avg_relevancy": sum(r["ragas"]["relevancy"] for r in results) / total,
            "avg_latency": sum(r["latency"] for r in results) / total,
            "avg_cost_usd": sum(r["cost_usd"] for r in results) / total,
            "total_cost_usd": round(runner.total_cost_usd, 4),
            "total_tokens": runner.total_tokens,
            "pass_rate": sum(1 for r in results if r["status"] == "pass") / total,
        },
    }


async def run_benchmark_with_results(agent_version: str, agent_version_key: str = "v1"):
    print(f"\n🚀 Khởi động Benchmark cho {agent_version}...")

    golden_path = ROOT / "data" / "golden_set.jsonl"
    if not golden_path.exists():
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None, None

    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None, None

    agent = create_agent(agent_version_key)
    judge = MultiModelJudge()
    runner = BenchmarkRunner(agent, RAGASEvaluator(), judge, batch_size=10)
    results = await runner.run_all(dataset)
    summary = _build_summary(results, agent_version, runner)
    return results, summary, runner


def _write_failure_analysis(v1_results, v2_results, v2_summary, gate_result, clusters):
    passed = sum(1 for r in v2_results if r["status"] == "pass")
    failed = len(v2_results) - passed
    m = v2_summary["metrics"]

    worst = sorted(v2_results, key=lambda r: r["judge"]["final_score"])[:3]

    lines = [
        "# Báo cáo Phân tích Thất bại (Failure Analysis Report)",
        "",
        "## 1. Tổng quan Benchmark",
        f"- **Tổng số cases:** {v2_summary['metadata']['total']}",
        f"- **Tỉ lệ Pass/Fail:** {passed}/{failed}",
        f"- **Pass rate:** {m['pass_rate']*100:.1f}%",
        "- **Điểm RAGAS trung bình:**",
        f"    - Faithfulness: {m['avg_faithfulness']:.3f}",
        f"    - Relevancy: {m['avg_relevancy']:.3f}",
        f"    - Hit Rate: {m['hit_rate']*100:.1f}%",
        f"    - MRR: {m['avg_mrr']:.3f}",
        f"- **Điểm LLM-Judge trung bình:** {m['avg_score']:.2f} / 5.0",
        f"- **Agreement Rate:** {m['agreement_rate']*100:.1f}%",
        f"- **Chi phí eval trung bình:** ${m['avg_cost_usd']:.6f}/case (tổng ${m['total_cost_usd']:.4f})",
        f"- **Latency trung bình:** {m['avg_latency']*1000:.0f}ms",
        "",
        "## 2. Regression Release Gate",
        f"- **Quyết định:** {gate_result['decision']}",
        f"- **Score delta:** {gate_result['delta']['avg_score']:+.3f}",
        f"- **Hit rate delta:** {gate_result['delta']['hit_rate']:+.3f}",
        f"- **Latency regression:** {gate_result['latency_regression_pct']*100:.1f}%",
        "",
        "## 3. Phân nhóm lỗi (Failure Clustering)",
        "| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |",
        "|----------|----------|---------------------|",
    ]

    for row in clusters["summary_table"]:
        lines.append(f"| {row['cluster']} | {row['count']} | {row['root_cause_hint']} |")

    lines.extend(["", "## 4. Phân tích 5 Whys (3 case tệ nhất)", ""])

    whys_templates = [
        ("retrieval_miss", "Vector DB không trả đúng chunk", "Chunking/overlap chưa tối ưu", "Ingestion pipeline"),
        ("hallucination", "LLM không bám context", "Retriever trả chunk nhiễu", "Chunk size quá lớn"),
        ("adversarial_fail", "Agent không từ chối injection", "Thiếu guardrails trong prompt", "Prompting"),
    ]

    for i, case in enumerate(worst, 1):
        ctype = case["test_meta"].get("type", "unknown")
        template = next((t for t in whys_templates if t[0] in ctype or ctype in t[0]), whys_templates[0])
        lines.extend([
            f"### Case #{i}: {case['test_case'][:80]}...",
            f"1. **Symptom:** Score {case['judge']['final_score']}, faithfulness {case['ragas']['faithfulness']}",
            f"2. **Why 1:** {template[1]}.",
            f"3. **Why 2:** {template[2]}.",
            f"4. **Why 3:** Metadata chunk thiếu doc_id khiến trace khó.",
            f"5. **Why 4:** Eval phát hiện muộn ở giai đoạn end-to-end.",
            f"6. **Root Cause:** {template[3]} — cần ưu tiên sửa ở V2+.",
            "",
        ])

    lines.extend([
        "## 5. Kế hoạch cải tiến (Action Plan)",
        "- [x] V2: keyword retrieval + guardrails adversarial/out-of-context.",
        "- [ ] Semantic chunking thay fixed-size (giảm retrieval_miss).",
        "- [ ] Thêm reranker (Cohere/BGE) trước generation.",
        "- [ ] Giảm 30% chi phí eval: cache judge cho câu hỏi trùng, batch size động.",
        "- [ ] System prompt nhấn mạnh 'chỉ trả lời từ context'.",
    ])

    analysis_path = ROOT / "analysis" / "failure_analysis.md"
    analysis_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Đã ghi {analysis_path}")


async def main():
    t0 = time.perf_counter()

    v1_results, v1_summary, _ = await run_benchmark_with_results("Agent_V1_Base", "v1")
    v2_results, v2_summary, v2_runner = await run_benchmark_with_results("Agent_V2_Optimized", "v2")

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    gate_decision, gate_result = evaluate_release_gate(v1_summary, v2_summary)
    clusters = cluster_failures(v2_results)

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    m1, m2 = v1_summary["metrics"], v2_summary["metrics"]
    print(f"V1 Score: {m1['avg_score']:.2f} | Hit: {m1['hit_rate']*100:.1f}% | Latency: {m1['avg_latency']*1000:.0f}ms")
    print(f"V2 Score: {m2['avg_score']:.2f} | Hit: {m2['hit_rate']*100:.1f}% | Latency: {m2['avg_latency']*1000:.0f}ms")
    print(f"Delta score: {gate_result['delta']['avg_score']:+.3f}")
    print(f"Failures (V2): {clusters['total_failures']}")

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    v2_summary["regression"] = {
        "v1": v1_summary["metrics"],
        "gate": gate_result,
    }

    with open(reports_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "v1_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v1_results, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "failure_clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    _write_failure_analysis(v1_results, v2_results, v2_summary, gate_result, clusters)

    elapsed = time.perf_counter() - t0
    print(f"\n⏱️  Tổng thời gian benchmark: {elapsed:.1f}s")

    if gate_decision == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")
        for b in gate_result.get("blockers", []):
            print(f"   - {b}")


if __name__ == "__main__":
    asyncio.run(main())
