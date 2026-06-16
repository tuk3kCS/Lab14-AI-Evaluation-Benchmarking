# Reflection — Hiếu

**Vai trò:** Evaluation Engine & Backend Developer  
**Lab:** Day 14 — AI Evaluation Factory  
**Ngày:** 16/06/2026

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

### Module đã triển khai

| File | Nội dung đóng góp |
|------|-------------------|
| `engine/runner.py` | Phát triển `BenchmarkRunner` với cơ chế bất đồng bộ (async execution) để đánh giá hàng loạt test cases với tốc độ cao, đồng thời tracking số lượng tokens và chi phí của cả Agent và Judge. |
| `engine/failure_cluster.py` | Thiết kế module Failure Clustering bằng cách sử dụng các luật (heuristic rules) kết hợp với RAGAS score và metadata để tự động phân loại các test cases thất bại thành các nhóm nguyên nhân cụ thể. |

### Chi tiết kỹ thuật

**`engine/runner.py` — Benchmark Runner Engine**
- Khởi tạo class `BenchmarkRunner` liên kết với Agent, RAGAS Evaluator và MultiModelJudge.
- Sử dụng hàm `run_single_test` để đo độ trễ (latency), tiêu thụ tokens và tổng chi phí (`total_cost_usd`) bằng cách gom kết quả gọi API của Agent và Judge qua `asyncio.gather`.
- Triển khai logic xử lý batch trong hàm `run_all`: chia nhỏ tập dữ liệu (dataset) thành các batch (mặc định `batch_size = 10`) và chạy song song (concurrent) để tối ưu thời gian chờ I/O, giúp pipeline eval đạt tốc độ tối đa.
- Tính toán final status dựa trên ngưỡng Pass/Fail (`PASS_SCORE_THRESHOLD = 3.0`).

**`engine/failure_cluster.py` — Root Cause Analysis & Clustering**
- Xây dựng bộ `FAILURE_RULES` với 6 loại nguyên nhân lỗi thường gặp trong RAG pipeline:
  - `hallucination`: `status == fail` & `faithfulness < 0.4` & không phải loại `out-of-context`.
  - `retrieval_miss`: Hit rate bằng 0 trong khi câu hỏi có yêu cầu document (`expected_retrieval_ids`).
  - `adversarial_fail`: Các câu hỏi dạng red-teaming bị fail.
  - `incomplete`: Trả lời thiếu thông tin (`relevancy < 0.35`).
  - `out_of_context_fail`: Trả lời sai với câu hỏi không có trong corpus.
  - `tone_mismatch`: Điểm tổng của Judge thấp dù `faithfulness` cao.
- Triển khai hàm `cluster_failures` giúp ánh xạ (map) từng test case lỗi vào nhóm tương ứng và xuất ra báo cáo `summary_table` với các gợi ý "root cause hint" để định hướng việc sửa lỗi một cách trực quan.

### Commit / PR liên quan

- Repository: https://github.com/tuk3kCS/Lab14-AI-Evaluation-Benchmarking
- Module đóng góp: `engine/runner.py`, `engine/failure_cluster.py`

---

## 2. Hiểu biết kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank) dưới góc nhìn Engine Developer
- Không giống như Hit Rate (chỉ cần lấy trúng ít nhất một chunk đúng), MRR cực kỳ nhạy cảm với "vị trí" của chunk tài liệu đúng đầu tiên. 
- Trong hệ thống Evaluation Engine, khi thu thập dữ liệu trả về từ bộ Retriever, tôi nhận thấy việc tính đúng MRR giúp đội ngũ AI đo lường chính xác hiệu năng của hàm `search`. Nếu MRR thấp nhưng Hit Rate cao, điều đó có nghĩa là model có lấy được tài liệu nhưng ranker đang hoạt động kém, đẩy tài liệu quan trọng xuống dưới làm LLM có nguy cơ bị phân tâm.

### Agreement Rate dưới góc nhìn Engine Developer
- Agreement Rate là một chỉ số "sanity check" tuyệt vời. Khi tôi chạy batch eval, việc chỉ tin vào một model judge duy nhất tiềm ẩn rủi ro rất cao (bị thiên vị - bias của riêng model đó). 
- Sự đồng thuận giữa các Judge đo lường mức độ "khách quan" của thang điểm. Nếu Agreement Rate thấp, vấn đề thường không nằm ở Engine mà nằm ở Rubric chấm điểm chưa rõ ràng, hoặc prompt đánh giá chứa nhiều yếu tố định tính mơ hồ.

### Trade-off chi phí / chất lượng trong Evaluation Engine
- Chạy hệ thống đánh giá bằng API Judge xịn (ví dụ GPT-4o) cho mọi test cases đem lại kết quả đáng tin cậy cao nhưng chi phí (Cost) và giới hạn Rate Limit API lại trở thành nút thắt cổ chai.
- Trong quá trình phát triển `runner.py`, tôi đã giải quyết Trade-off này bằng việc áp dụng Batch async (tối ưu tốc độ) và tracking chặt chẽ luồng cost. Nhờ theo dõi được `judge_tokens` và `judge_cost_usd` trực tiếp qua Runner, nhóm có thể dễ dàng kiểm soát chi phí của API Judge và điều phối mức độ phụ thuộc vào Heuristic Judge để tối ưu ngân sách mà vẫn giữ được độ tin cậy.

---

## 3. Vấn đề đã giải quyết (Problem Solving)

### Vấn đề 1: Pipeline Evaluation bị nghẽn (Bottleneck) khi scale lên nhiều test case
- **Triệu chứng:** Chạy hàng chục test cases một cách tuần tự (sequential) tiêu tốn thời gian rất lớn do phần lớn thời gian Engine phải "ngồi chờ" phản hồi mạng từ LLM API cho cả Agent và Judge.
- **Giải pháp:** Tôi đã cấu trúc lại `runner.py` bằng cách sử dụng thư viện `asyncio`. Phương thức `run_single_test` được dùng để chạy lệnh query và score một cách bất đồng bộ. Sau đó ở `run_all`, tôi thiết lập `batch_size = 10` gọi qua `asyncio.gather` để submit 10 luồng request cùng lúc, giúp rút ngắn thời gian chạy tổng thể xuống nhiều lần.

### Vấn đề 2: Chết ngập trong log lỗi khi phân tích nguyên nhân thất bại (Failure Analysis)
- **Triệu chứng:** Sau mỗi lần benchmark, hệ thống trả về rất nhiều test cases bị đánh "fail". Ban đầu, nhóm phải mở report lên, đọc mắt từng câu trả lời và đối chiếu với dataset để phán đoán xem lỗi nằm ở khâu Retrieval (tìm kiếm) hay Generation (sinh text), vô cùng tốn thời gian.
- **Giải pháp:** Tôi phát triển module `failure_cluster.py` hoạt động giống như một bộ chẩn đoán. Dựa vào bộ luật `FAILURE_RULES` kết hợp điểm RAGAS và metadata (Ví dụ: `faithfulness < 0.4` -> rớt vào nhóm Hallucination; `Hit Rate = 0` -> rớt vào nhóm Retrieval Miss). Nhờ đó, output báo cáo tự động phân loại rành mạch kèm theo "Root cause hint" (gợi ý nguyên nhân gốc rễ), chỉ đích danh module nào cần sửa trong chu trình.

---

## 4. Bài học rút ra

1. **Hiệu năng hệ thống đánh giá (Eval System Performance) cũng quan trọng như hệ thống chính:** Một hệ thống Eval chạy quá chậm sẽ làm nhụt chí việc lặp lại (iterate) các thử nghiệm thường xuyên. Sử dụng Asynchronous I/O là tiêu chuẩn bắt buộc cho công cụ Benchmark.
2. **Chi phí Eval phải được coi là một metric tối quan trọng:** Việc đo lường trực tiếp lượng `tokens` và `cost_usd` trên mỗi case ngay lúc code `runner.py` giúp toàn đội có nhận thức tốt hơn về chi phí và chủ động tìm giải pháp cân bằng.
3. **Phân tích lỗi (Error Analysis) cần được hệ thống hóa:** Dữ liệu điểm số khô khan sẽ vô nghĩa nếu không thể phân cụm. Rule-based Clustering là một hướng tiếp cận "chi phí thấp - hiệu quả cao" để bắt bệnh cho hệ thống RAG trong thực tiễn.
