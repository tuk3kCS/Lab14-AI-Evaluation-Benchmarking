# Reflection — Trần Hoàng Hà (Multi-Judge Engineer)

## Đóng góp kỹ thuật

- **Module phụ trách**: Multi-Judge Consensus Engine.
- **File chính**: `engine/llm_judge.py`.
- **Các hạng mục đã triển khai**:
  - Xây dựng class `MultiModelJudge` chạy **2 judge song song**:
    - **Judge A**: GPT-4o-mini (API nếu có `OPENAI_API_KEY`, nếu không thì heuristic fallback)
    - **Judge B**: Claude-style judge (heuristic “style factor”, hoặc proxy qua API)
  - Thiết kế **consensus & calibration**:
    - **Conflict resolution**: lệch ≤ 1 điểm → lấy trung bình; lệch > 1 → `conservative_min`
    - **Agreement Rate**:  \max(0, 1 - |s_a - s_b|/4)  và giảm thêm 50% khi có conflict
  - Bổ sung `check_position_bias()` để kiểm tra **position bias** khi đổi thứ tự response.

## Hiểu biết kỹ thuật

- **Agreement Rate**: phản ánh mức đồng thuận giữa các judge; hữu ích để phát hiện rubric mơ hồ/bias nhưng không thay thế được phân tích định tính.
- **Conflict Resolution**: dùng chiến lược “conservative_min” giúp giảm rủi ro “thả” câu trả lời tệ khi 2 judge bất đồng mạnh.
- **Chi phí vs độ tin cậy**:
  - Heuristic judge giúp chạy nhanh/offline để sanity check.
  - Khi có API key, gọi judge thật sẽ tăng độ tin cậy nhưng cần theo dõi tokens/cost và tránh gọi không cần thiết.

## Vấn đề đã giải quyết

- **Offline/No-API**: thiết kế heuristic fallback để hệ thống vẫn chạy được khi không có API key, đảm bảo pipeline benchmark không bị block.
- **Bất đồng điểm số**: bổ sung ngưỡng conflict và logic xử lý, giúp kết quả ổn định và có “cờ” để điều tra khi agreement thấp.

