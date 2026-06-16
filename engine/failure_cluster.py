from collections import Counter, defaultdict
from typing import Dict, List


FAILURE_RULES = {
    "hallucination": lambda r: (
        r["status"] == "fail"
        and r["ragas"]["faithfulness"] < 0.4
        and r["test_meta"].get("type") != "out-of-context"
    ),
    "retrieval_miss": lambda r: (
        r["ragas"]["retrieval"]["hit_rate"] == 0
        and len(r.get("expected_retrieval_ids", [])) > 0
    ),
    "adversarial_fail": lambda r: (
        r["status"] == "fail" and r["test_meta"].get("type") == "adversarial"
    ),
    "incomplete": lambda r: (
        r["status"] == "fail" and r["ragas"]["relevancy"] < 0.35
    ),
    "out_of_context_fail": lambda r: (
        r["status"] == "fail" and r["test_meta"].get("type") == "out-of-context"
    ),
    "tone_mismatch": lambda r: (
        r["status"] == "fail"
        and r["judge"]["final_score"] < 3
        and r["ragas"]["faithfulness"] >= 0.4
    ),
}


def cluster_failures(results: List[Dict]) -> Dict:
    clusters = defaultdict(list)
    primary_labels = []

    for i, r in enumerate(results):
        if r["status"] == "pass":
            continue

        matched = [name for name, rule in FAILURE_RULES.items() if rule(r)]
        label = matched[0] if matched else "other"
        primary_labels.append(label)
        clusters[label].append(
            {
                "index": i,
                "question": r["test_case"][:120],
                "score": r["judge"]["final_score"],
                "faithfulness": r["ragas"]["faithfulness"],
                "hit_rate": r["ragas"]["retrieval"]["hit_rate"],
                "type": r["test_meta"].get("type", "unknown"),
            }
        )

    counts = Counter(primary_labels)
    return {
        "total_failures": len(primary_labels),
        "clusters": {k: {"count": len(v), "cases": v} for k, v in clusters.items()},
        "summary_table": [
            {
                "cluster": k,
                "count": counts[k],
                "root_cause_hint": _root_cause_hint(k),
            }
            for k in sorted(counts, key=counts.get, reverse=True)
        ],
    }


def _root_cause_hint(cluster: str) -> str:
    hints = {
        "hallucination": "Retriever lấy sai context hoặc LLM bịa thông tin",
        "retrieval_miss": "Embedding/chunking không khớp query",
        "adversarial_fail": "Thiếu guardrails chống prompt injection",
        "incomplete": "Prompt không yêu cầu trả lời đủ chi tiết",
        "out_of_context_fail": "Agent không từ chối trả lời khi thiếu tài liệu",
        "tone_mismatch": "Ngôn ngữ không phù hợp ngữ cảnh hỗ trợ",
        "other": "Cần review thủ công",
    }
    return hints.get(cluster, "Chưa phân loại")
