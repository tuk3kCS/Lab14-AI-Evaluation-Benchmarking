# Reflection — Tùng

**Vai trò:** Data Lead — SDG & Golden Dataset  
**Lab:** Day 14 — AI Evaluation Factory  
**Ngày:** 16/06/2026

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

### Module đã triển khai

| File | Nội dung đóng góp |
|------|-------------------|
| `data/synthetic_gen.py` | Phát triển bộ sinh dữ liệu Golden Dataset với 8 nhóm test cases chuyên biệt từ corpus |
| `data/corpus/documents.json` | Chuẩn bị corpus gồm 20 tài liệu đa lĩnh vực (Bảo mật, HR, Finance, Dev Docs, RAG practices) |
| `data/golden_set.jsonl` | Tạo dataset hoàn chỉnh gồm 62 cases đa dạng độ khó (easy, medium, hard) và thể loại |

### Chi tiết kỹ thuật

**`data/corpus/documents.json`**
- Xây dựng kho tài liệu 20 văn bản làm nền tảng tri thức cho RAG Agent.
- Cấu trúc tài liệu chứa đầy đủ: `id`, `source`, `section`, và `content`.
- Thiết kế nội dung chứa các bẫy logic như:
  - Bản nháp v1 vs bản chính thức v2 của chính sách Data Retention để kiểm tra khả năng xử lý thông tin mâu thuẫn (conflicting-info).
  - Các khái niệm kỹ thuật phức tạp (RAG, SLA, API Rate Limit) để kiểm tra các câu hỏi dạng multi-hop và technical.

**`data/synthetic_gen.py` — Synthetic Data Generation (SDG)**
- Phân tách logic sinh dữ liệu thành 8 hàm sinh tương ứng với 8 dạng bài kiểm tra:
  - `generate_fact_check_cases`: 18 cases hỏi đáp thông tin trực tiếp, độ khó Easy/Medium.
  - `generate_paraphrase_cases`: 10 cases diễn đạt lại câu hỏi bằng từ đồng nghĩa để test khả năng Semantic Retrieval.
  - `generate_adversarial_cases`: 6 cases Red Teaming kiểm tra khả năng phòng chống prompt injection, jailbreak, DAN.
  - `generate_edge_cases`: 7 cases Out-of-Context (3 cases rỗng context để test từ chối), Ambiguous (cần làm rõ), và Conflicting-info (ưu tiên thông tin bản v2 có hiệu lực).
  - `generate_multi_hop_cases`: 5 cases yêu cầu kết hợp thông tin từ nhiều tài liệu khác nhau.
  - `generate_conversation_cases`: 4 cases hội thoại nhiều lượt (Multi-turn) mang theo ngữ cảnh.
  - `generate_technical_stress_cases`: 4 cases kiểm tra hiệu năng hệ thống với câu hỏi sâu và context cực dài.
  - `generate_extra_variants`: 8 cases bổ sung để tăng độ phong phú và đạt quy mô dataset mong muốn.
- Tích hợp hàm `generate_qa_from_text` gọi OpenAI API (`gpt-4o-mini`) bất đồng bộ (async) với cấu hình `response_format={"type": "json_object"}` và cơ chế fallback tự động nếu thiếu API Key hoặc lỗi kết nối.
- Mỗi test case được gán nhãn chi tiết metadata: `difficulty` (easy/medium/hard), `type` (fact-check, adversarial, ...), và danh sách tags giúp phân nhóm lỗi ở bước Failure Clustering.

### Kết quả thống kê Golden Dataset (62 cases)

- **Tổng số cases:** 62 cases (vượt xa yêu cầu tối thiểu 50 cases của Lab).
- **Có expected_retrieval_ids:** 59 cases (3 cases out-of-context để trống `[]` để kiểm thử khả năng chống bịa đặt).
- **Phân bổ độ khó:** 22 Easy, 21 Medium, 19 Hard.

### Commit / PR liên quan

- Repository: https://github.com/tuk3kCS/Lab14-AI-Evaluation-Benchmarking
- Module đóng góp: `data/synthetic_gen.py`, `data/corpus/documents.json`, `data/golden_set.jsonl`

---

## 2. Hiểu biết kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank) dưới góc nhìn Data Lead
- MRR đo lường thứ hạng của tài liệu đúng đầu tiên. Trong pha tạo dataset, tôi nhận thấy việc định nghĩa chính xác và đầy đủ trường `expected_retrieval_ids` cho từng test case là điều kiện tiên quyết để tính toán MRR một cách đúng đắn.
- Đặc biệt đối với các câu hỏi phức tạp như multi-hop (cần kết hợp nhiều tài liệu), việc xác định đúng các ID nguồn giúp nhóm đánh giá được chính xác liệu Agent có lấy đủ và lấy đúng thứ tự ưu tiên các tài liệu cần thiết hay không.

### Agreement Rate dưới góc nhìn Data Lead
- Sự đồng thuận giữa các Model Judge chịu ảnh hưởng rất lớn từ **chất lượng và độ rõ ràng của Golden Dataset**.
- Nếu câu hỏi sinh ra bị mơ hồ hoặc câu trả lời kỳ vọng (`expected_answer`) không chi tiết, hai Model Judge (GPT-4o-mini và Claude) sẽ dễ dàng đưa ra các cách đánh giá khác nhau, dẫn tới Agreement Rate giảm. Để duy trì Agreement Rate cao (thực tế nhóm đạt 93.6%), các câu hỏi và câu trả lời trong SDG đều được tôi tinh chỉnh thủ công và viết cực kỳ rõ ràng, chuẩn xác theo corpus.

### Trade-off chi phí / chất lượng trong SDG
- Thiết kế SDG của tôi áp dụng chiến lược kết hợp (Hybrid):
  - **Heuristic/Template-based:** Chi phí bằng 0, tốc độ sinh cực nhanh, dễ kiểm soát cấu trúc và logic bài test (đặc biệt là các case Red Teaming và Edge cases).
  - **LLM-based Generation (`generate_qa_from_text`):** Đa dạng hóa câu hỏi tự nhiên nhưng tốn chi phí API và dễ gặp lỗi định dạng nếu LLM trả về không đúng schema mong muốn.
- Quyết định: Sử dụng Heuristic Template làm bộ khung lõi ổn định để chạy offline, và tích hợp LLM-based làm cổng mở rộng linh hoạt cho phép scale tập dữ liệu khi cần thiết.

---

## 3. Vấn đề đã giải quyết (Problem Solving)

### Vấn đề 1: Retrieval Evaluation bị block do thiếu Ground Truth ID
- **Triệu chứng:** Ban đầu bộ sinh dữ liệu SDG chỉ tạo cặp Question - Answer thông thường, khiến Duyên (Retrieval Engineer) không thể tính toán Hit Rate và MRR tự động do không biết câu hỏi tương ứng với tài liệu nào.
- **Giải pháp:** Tôi bổ sung trường `expected_retrieval_ids: List[str]` khớp với ID tài liệu trong `documents.json`. Đối với các câu hỏi multi-hop, danh sách này chứa nhiều ID tương ứng, hỗ trợ đầy đủ cho logic đánh giá nâng cao.

### Vấn đề 2: Agent V2 không từ chối các câu hỏi nằm ngoài cơ sở dữ liệu
- **Triệu chứng:** Agent V1 và V2 ban đầu vẫn cố gắng bịa câu trả lời (hallucination) cho các câu hỏi không liên quan đến công ty (ví dụ: giá cổ phiếu Apple, thời tiết Hà Nội).
- **Giải pháp:** Tôi cố ý tạo ra 3 test case `out-of-context` và gán nhãn `expected_retrieval_ids = []`. Nhờ có các case benchmark này, nhóm đã phát hiện và thêm guardrails vào prompt của Agent V2 để từ chối trả lời an toàn khi không tìm thấy tài liệu phù hợp.

### Vấn đề 3: Lỗi format khi parse kết quả LLM SDG
- **Triệu chứng:** Khi chạy `generate_qa_from_text`, LLM đôi lúc trả về chuỗi JSON bọc trong markdown block (` ```json ... ``` `) khiến `json.loads` bị lỗi.
- **Giải pháp:** Sử dụng tham số `response_format={"type": "json_object"}` của OpenAI API để bắt buộc model trả về JSON thô, kết hợp khối `try-except` an toàn và fallback thông minh để không gây lỗi crash luồng chạy dữ liệu.

---

## 4. Bài học rút ra

1. **Dữ liệu tốt là gốc rễ của đánh giá** — Mọi metric như Hit Rate, MRR, Faithfulness hay điểm số của Judge đều trở nên vô nghĩa nếu tập Golden Dataset không phản ánh đúng các tình huống thực tế và các lỗi logic.
2. **Red Teaming và Edge cases phải được đưa vào sớm** — Việc kiểm tra bảo mật (Adversarial) và thông tin mâu thuẫn cần được thiết kế ngay từ bước sinh dữ liệu để đảm bảo Agent có đủ độ bền bỉ (robustness) trước khi đưa vào sản xuất.
3. **Thiết kế schema dữ liệu thống nhất** — Thống nhất schema (`expected_retrieval_ids`, `metadata`) giữa Data Lead, Retrieval Engineer và Tech Lead ngay từ ngày đầu giúp việc tích hợp pipeline diễn ra trơn tru, giảm thiểu xung đột code.

---

*Tùng — Data Lead, Lab Day 14*
