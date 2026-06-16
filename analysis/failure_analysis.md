# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 62
- **Tỉ lệ Pass/Fail:** 52/10
- **Pass rate:** 83.9%
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.892
    - Relevancy: 0.478
    - Hit Rate: 87.1%
    - MRR: 0.804
- **Điểm LLM-Judge trung bình:** 4.03 / 5.0
- **Agreement Rate:** 93.6%
- **Chi phí eval trung bình:** $0.000081/case (tổng $0.0050)
- **Latency trung bình:** 46ms

## 2. Regression Release Gate
- **Quyết định:** APPROVE
- **Score delta:** +1.975
- **Hit rate delta:** +0.726
- **Latency regression:** -43.5%

## 3. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| hallucination | 5 | Retriever lấy sai context hoặc LLM bịa thông tin |
| incomplete | 3 | Prompt không yêu cầu trả lời đủ chi tiết |
| out_of_context_fail | 1 | Agent không từ chối trả lời khi thiếu tài liệu |
| adversarial_fail | 1 | Thiếu guardrails chống prompt injection |

## 4. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: Làm sao để reset?...
1. **Symptom:** Score 1.62, faithfulness 0.35
2. **Why 1:** Vector DB không trả đúng chunk.
3. **Why 2:** Chunking/overlap chưa tối ưu.
4. **Why 3:** Metadata chunk thiếu doc_id khiến trace khó.
5. **Why 4:** Eval phát hiện muộn ở giai đoạn end-to-end.
6. **Root Cause:** Ingestion pipeline — cần ưu tiên sửa ở V2+.

### Case #2: Bị lỗi 429 khi gọi API nghĩa là gì?...
1. **Symptom:** Score 1.94, faithfulness 0.35
2. **Why 1:** Vector DB không trả đúng chunk.
3. **Why 2:** Chunking/overlap chưa tối ưu.
4. **Why 3:** Metadata chunk thiếu doc_id khiến trace khó.
5. **Why 4:** Eval phát hiện muộn ở giai đoạn end-to-end.
6. **Root Cause:** Ingestion pipeline — cần ưu tiên sửa ở V2+.

### Case #3: Tóm tắt toàn bộ chính sách trong corpus: bảo mật, HR, API và đánh giá AI....
1. **Symptom:** Score 2.08, faithfulness 1.0
2. **Why 1:** Vector DB không trả đúng chunk.
3. **Why 2:** Chunking/overlap chưa tối ưu.
4. **Why 3:** Metadata chunk thiếu doc_id khiến trace khó.
5. **Why 4:** Eval phát hiện muộn ở giai đoạn end-to-end.
6. **Root Cause:** Ingestion pipeline — cần ưu tiên sửa ở V2+.

## 5. Kế hoạch cải tiến (Action Plan)
- [x] V2: keyword retrieval + guardrails adversarial/out-of-context.
- [ ] Semantic chunking thay fixed-size (giảm retrieval_miss).
- [ ] Thêm reranker (Cohere/BGE) trước generation.
- [ ] Giảm 30% chi phí eval: cache judge cho câu hỏi trùng, batch size động.
- [ ] System prompt nhấn mạnh 'chỉ trả lời từ context'.