# Reflection — Trường

**Vai trò:** Tech Lead — Integration, Regression & Agent  
**Lab:** Day 14 — AI Evaluation Factory  
**Ngày:** 16/06/2026

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

### Module đã triển khai

| File | Nội dung đóng góp |
|------|-------------------|
| `main.py` | Orchestration toàn pipeline: chạy benchmark V1/V2, tổng hợp `summary.json`, gọi Release Gate, sinh tự động `failure_analysis.md` |
| `engine/release_gate.py` | Logic Auto-Gate đa tiêu chí (score, hit_rate, latency, cost) — quyết định `APPROVE` / `BLOCK` |
| `agent/main_agent.py` | RAG Agent 2 phiên bản: V1 baseline (retrieval yếu) và V2 optimized (keyword retrieval + guardrails) |
| `report.md` | Báo cáo nhóm tổng hợp kết quả benchmark và phân công |

### Chi tiết kỹ thuật

**`main.py` — Integration layer**

- Hàm `_build_summary()` gom metrics từ 62 cases: `avg_score`, `hit_rate`, `avg_mrr`, `agreement_rate`, `pass_rate`, `total_cost_usd`, `total_tokens`
- Chạy tuần tự V1 rồi V2 qua `create_agent("v1")` / `create_agent("v2")`, ghi `regression` block vào `summary.json`
- Hàm `_write_failure_analysis()` tự động điền số liệu thật + 5 Whys cho 3 case score thấp nhất, tránh lỗi format khi nộp bài

**`engine/release_gate.py` — Regression Release Gate**

- So sánh delta 6 metric giữa V1 và V2
- Block nếu: score delta < +0.05, hit_rate giảm, latency tăng > 25%, cost tăng > 30%, hoặc V2 avg_score < 2.5
- Override APPROVE khi cải thiện chất lượng đáng kể (delta score ≥ +0.3 và hit_rate không âm)

**`agent/main_agent.py` — Agent V1 vs V2**

- **V1:** `_retrieve_v1()` lấy cố định `corpus[:top_k]` → Hit Rate 14.5%, câu trả lời generic
- **V2:** `_retrieve_v2()` keyword overlap scoring + guardrails adversarial/out-of-context → Hit Rate 87.1%, pass rate 83.9%
- Thiết kế cố ý tạo delta regression có ý nghĩa để Release Gate hoạt động đúng

### Commit / PR liên quan

- Repository: https://github.com/tuk3kCS/Lab14-AI-Evaluation-Benchmarking
- Commit nhóm: `Do Lab Exercise` — tích hợp toàn bộ module eval factory

---

## 2. Hiểu biết kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank)

MRR = 1/vị_trí (1-indexed) của tài liệu ground-truth **đầu tiên** xuất hiện trong danh sách retrieved. Khác Hit Rate (chỉ hỏi có/không có trong top-k), MRR phạt nặng khi document đúng nằm ở vị trí thấp.

Trong benchmark nhóm: V1 MRR = **0.091** (chunk đúng hiếm khi lọt top-3), V2 MRR = **0.804** (keyword scoring đưa chunk liên quan lên đầu). Điều này giải thích vì sao V1 có faithfulness 0.20 dù vẫn retrieve được *một số* chunk — chunk sai vị trí vẫn làm LLM trả lời lệch.

### Cohen's Kappa

Cohen's Kappa đo mức đồng thuận **vượt ngẫu nhiên** giữa 2 annotator (judge), công thức κ = (P_o − P_e) / (1 − P_e). Khác Agreement Rate đơn giản (% cases lệch ≤ 1 điểm), Kappa điều chỉnh cho trường hợp 2 judge cùng bias về một phía.

Nhóm dùng Agreement Rate = `max(0, 1 − |score_a − score_b| / 4)` trong `llm_judge.py` — đủ cho lab nhưng trong production nên bổ sung Kappa khi rubric có nhiều mức điểm rời rạc. Case "Cohen's Kappa khác Agreement Rate" trong dataset là ví dụ agent cần nói "không có trong corpus" thay vì hallucinate.

### Position Bias

Position bias xảy ra khi LLM-judge cho điểm cao hơn cho câu trả lời xuất hiện trước trong prompt. `check_position_bias()` trong `llm_judge.py` chạy eval 2 lần với thứ tự đổi chỗ để phát hiện bias (flag khi lệch > 1.5 điểm).

Khi thiết kế Release Gate, tôi không chỉ nhìn `avg_score` mà còn `agreement_rate` (93.6%) để đảm bảo điểm V2 không đến từ một judge bị bias.

### Trade-off Chi phí / Chất lượng

| Chiến lược | Chi phí | Chất lượng |
|------------|---------|------------|
| Heuristic judge (fallback) | ~$0.00005/case | Đủ cho pattern rõ (adversarial, overlap) |
| API judge (GPT-4o-mini) | ~$0.00015/case | Chính xác hơn với câu mơ hồ |
| Batch async (size=10) | Giảm wall-clock 1.2s/124 runs | Không giảm token, chỉ giảm thời gian chờ |

Quyết định thiết kế: Release Gate cho phép cost tăng tối đa 30% nếu score và hit_rate cải thiện — thực tế V2 cost giảm 6.9% nhờ latency factor nhỏ hơn và câu trả lời ngắn hơn (từ context thay vì template dài).

---

## 3. Vấn đề đã giải quyết (Problem Solving)

### Vấn đề 1: Regression delta = 0 khi V1 và V2 giống nhau

**Triệu chứng:** Template gốc chỉ đổi tên version, metrics không đổi → Release Gate vô nghĩa.

**Giải pháp:** Thiết kế V1 và V2 khác biệt rõ ràng ở 3 lớp — retrieval (`corpus[:k]` vs keyword score), generation (template vs context), guardrails (không có vs adversarial/OOC). Kết quả: delta score **+1.98**, hit_rate **+72.6%**, Gate **APPROVE**.

### Vấn đề 2: `failure_analysis.md` dễ bị quên hoặc sai format

**Triệu chứng:** README yêu cầu file đã điền đầy đủ; nếu quên sẽ mất điểm Failure Analysis và có thể lỗi `check_lab.py`.

**Giải pháp:** Tích hợp `_write_failure_analysis()` vào `main.py` — mỗi lần chạy benchmark tự ghi số liệu thật, bảng clustering từ `failure_cluster.py`, và 5 Whys cho 3 case tệ nhất. Đảm bảo reproducibility: chạy lại `main.py` là có báo cáo mới nhất.

### Vấn đề 3: Release Gate quá đơn giản (`if delta > 0`)

**Triệu chứng:** Agent V2 có thể score cao hơn nhưng hit_rate giảm (hallucination nhiều hơn) — vẫn bị approve.

**Giải pháp:** `evaluate_release_gate()` kiểm tra đồng thời 5 điều kiện + override có điều kiện cho cải thiện lớn. Trong run thực tế: 0 blocker, APPROVE vì mọi metric đều cải thiện.

### Vấn đề 4: Phối hợp nhóm 5 người trong 4 giờ

**Giải pháp:** Chia module độc lập (SDG, retrieval, judge, runner) với interface rõ (`retrieved_ids`, `judge.final_score`, `ragas.retrieval`). Tech Lead chỉ wire ở `main.py` và định nghĩa Agent V1/V2 để unblock các nhánh song song. Schema dataset có `expected_retrieval_ids` ngay từ đầu để Duyên không bị block.

---

## 4. Bài học rút ra

1. **Đo retrieval trước generation** — V1 chứng minh rõ: generation tốt vô nghĩa nếu retrieval sai (faithfulness 0.20 vs 0.94).
2. **Regression Gate cần nhiều metric** — một con số `avg_score` không đủ; hit_rate và latency bắt buộc.
3. **Automation giảm lỗi nộp bài** — sinh `failure_analysis.md` từ pipeline giúp nhóm tập trung phân tích thay vì copy số thủ công.

---

*Trường — Tech Lead, Lab Day 14*
