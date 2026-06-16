# Reflection — Duyên

**Vai trò:** Retrieval Engineer  
**Lab:** Day 14 — AI Evaluation Factory  
**Ngày:** 16/06/2026

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

### Module đã triển khai

| File | Nội dung đóng góp |
|------|-------------------|
| `engine/retrieval_eval.py` | Hit Rate @K, MRR, Recall@K; `evaluate_single()` và `evaluate_batch()` với danh sách case miss |
| `engine/ragas_eval.py` | Faithfulness, Relevancy, Context Precision; tích hợp retrieval metrics và giải thích liên kết retrieval ↔ answer |
| `tests/test_retrieval_ragas.py` | Unit tests xác minh MRR, retrieval miss cap faithfulness, out-of-context edge cases |

### Chi tiết kỹ thuật

**`engine/retrieval_eval.py`**

- `calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)`: binary hit trong top-k
- `calculate_mrr()`: reciprocal rank của ground-truth document đầu tiên
- `calculate_recall_at_k()`: hỗ trợ multi-doc ground truth (case multi-hop)
- `evaluate_batch()`: trả về `retrieval_miss_count`, `missed_cases` (top 10) phục vụ failure clustering
- `compare_versions()`: delta hit_rate / MRR giữa V1 và V2 cho regression analysis

**`engine/ragas_eval.py`**

- `_faithfulness()`: overlap answer–context; **cap ≤ 0.35 khi hit_rate = 0** để phản ánh retrieval miss → hallucination
- `_relevancy()`: so sánh answer với `expected_answer` và `question`
- `_context_precision()`: đo context retrieved có liên quan câu hỏi không
- `retrieval_answer_link`: chuỗi giải thích tự động cho failure analysis (5 Whys)
- Interface output tương thích `BenchmarkRunner`: `faithfulness`, `relevancy`, `retrieval.hit_rate`, `retrieval.mrr`

### Kết quả benchmark (62 cases)

| Metric | Agent V1 | Agent V2 |
|--------|----------|----------|
| Hit Rate | 14.5% | **87.1%** |
| MRR | 0.091 | **0.804** |
| Faithfulness | 0.20 | **0.94** |

Nhận xét: V1 lấy cố định `corpus[:3]` → hit rate thấp → faithfulness bị cap thấp. V2 keyword retrieval cải thiện cả retrieval lẫn generation quality.

### Commit / PR liên quan

- Repository: https://github.com/tuk3kCS/Lab14-AI-Evaluation-Benchmarking
- Module: `engine/retrieval_eval.py`, `engine/ragas_eval.py`, `tests/test_retrieval_ragas.py`

---

## 2. Hiểu biết kỹ thuật (Technical Depth)

### Hit Rate vs MRR

**Hit Rate @K** trả lời câu hỏi nhị phân: *có ít nhất một tài liệu đúng trong top-k không?* Phù hợp gate tối thiểu (pass/fail retrieval).

**MRR (Mean Reciprocal Rank)** nhạy hơn với **thứ hạng**: document đúng ở vị trí 1 cho MRR=1.0, ở vị trí 3 cho MRR=0.33. Trong benchmark nhóm, V1 có Hit Rate 14.5% nhưng MRR chỉ 0.091 — nghĩa là ngay cả khi hit, chunk đúng thường nằm cuối danh sách, LLM vẫn dễ bám context sai.

### Mối liên hệ Retrieval Quality ↔ Answer Quality

Thiết kế có chủ đích trong `ragas_eval.py`:

```
hit_rate = 0  →  faithfulness cap ≤ 0.35
hit_rate = 1, context overlap cao  →  faithfulness có thể > 0.8
```

Dữ liệu thực tế khớp giả thuyết: V1 faithfulness 0.20 (retrieval yếu), V2 faithfulness 0.94 (retrieval tốt). Cluster `retrieval_miss` (4 failures V2) cho thấy paraphrase query vẫn miss chunk dù generation logic đúng — cần semantic embedding thay keyword overlap.

### Cohen's Kappa (liên quan cross-module)

Cohen's Kappa đo đồng thuận judge vượt ngẫu nhiên: κ = (P_o − P_e) / (1 − P_e). Phía retrieval, metric tương đương về mặt ý nghĩa là **Recall@K vs Hit Rate**: Hit Rate có thể = 1 với multi-doc ground truth nhưng Recall@K thấp nếu chỉ hit 1/3 doc cần thiết (case multi-hop). Module `calculate_recall_at_k()` bổ sung góc nhìn này.

### Trade-off Chi phí / Chất lượng

| Lựa chọn | Chi phí | Chất lượng eval |
|----------|---------|-----------------|
| Heuristic overlap (đã dùng) | ~$0, chạy local | Đủ cho 62 cases, tương quan tốt với retrieval thật |
| RAGAS library + LLM API | ~$0.01–0.05/case | Chính xác hơn với câu mơ hồ, paraphrase |
| Embedding retrieval eval | Thêm compute embedding | Giảm false negative paraphrase |

Quyết định: dùng heuristic cho pipeline async nhanh (< 2 phút); đề xuất bước 2 dùng embedding similarity cho case `retrieval_miss` trong production.

---

## 3. Vấn đề đã giải quyết (Problem Solving)

### Vấn đề 1: Agent không trả `retrieved_ids` — không đo được retrieval

**Triệu chứng:** Ban đầu agent chỉ trả `answer` + `contexts`, thiếu ID để đối chiếu ground truth.

**Giải pháp:** Thống nhất schema với Tech Lead: agent phải trả `retrieved_ids: List[str]` map với `documents.json`. Dataset SDG (Tùng) bổ sung `expected_retrieval_ids` song song.

### Vấn đề 2: Faithfulness cao giả khi retrieval sai

**Triệu chứng:** Overlap heuristic giữa answer generic và context random vẫn cho điểm > 0.4.

**Giải pháp:** Truyền `hit_rate` vào `_faithfulness()` và cap điểm khi miss. Kết quả V1 faithfulness 0.20 phản ánh đúng thực tế hơn.

### Vấn đề 3: Out-of-context cases làm méo Hit Rate trung bình

**Triệu chứng:** 3 case out-of-context cố ý để trống `expected_retrieval_ids`.

**Giải pháp:** Rule riêng — không có ground truth thì pass nếu agent không retrieve; fail nếu vẫn retrieve (agent không nên bịa context). `evaluate_batch()` tách `cases_with_ground_truth` và `retrieval_miss_rate` chỉ trên subset có GT.

### Vấn đề 4: Paraphrase gây retrieval_miss dù V2 tốt hơn nhiều

**Triệu chứng:** "đổi password" vs "thay đổi mật khẩu" — keyword overlap = 0.

**Giải pháp:** Ghi nhận trong failure analysis; đề xuất semantic chunking + embedding retrieval cho V3. Hiện tại `missed_cases` trong batch report giúp Hiểu cluster `retrieval_miss` tự động.

---

## 4. Bài học rút ra

1. **Luôn eval retrieval trước generation** — metric faithfulness không có ý nghĩa nếu không biết chunk nào được retrieve.
2. **MRR bổ sung Hit Rate** — biết hit chưa đủ, cần biết chunk đúng ở rank mấy.
3. **Schema dataset quan trọng bằng code** — `expected_retrieval_ids` phải có từ giai đoạn SDG.
4. **Heuristic eval rẻ và nhanh** — phù hợp lab; production cần hybrid (heuristic pre-screen + embedding cho edge cases).

---

*Duyên — Retrieval Engineer, Lab Day 14*
