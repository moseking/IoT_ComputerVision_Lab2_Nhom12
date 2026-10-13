# Demo với baseline mới

Đã train/evaluate, backend V3 đang chạy, review26camera hoàn tất. **Bắt đầu ở [TEST_CAMERA_NOW](TEST_CAMERA_NOW.md)** để nạp firmware và test. 91tests pass, firmware compile lại pass; đã kiểm JPEG camera qua HTTP từ laptop. Chưa xác nhận capture/send ESP32 trực tiếp; đeo sai chỉ khớp6/14camera, xem [báo cáo](../reports/esp32_v1/FINAL_RESULT.md).

# Demo Lab 3 và bằng chứng

Train đầy đủ/evaluate trước demo; kiểm tra metrics lớp đeo sai.
Nạp firmware với WiFi2.4GHz, IP máy tính/cloud đúng và nguồn5V ổn định.
Không dùng model verification một epoch để tuyên bố chất lượng final.
Cloud yêu cầu server cloud thật; không gọi LAN là cloud.

## Trình tự live

1. Start FastAPI một worker, log model_ready; /health model_loaded=true.
2. Mở dashboard và /docs.
3. ESP32-CAM connect WiFi, Serial115200 in IP thật.
4. Camera capture JPEG và gửi multipart /predict.
5. Server log request_id, bytes, source=esp32, count, inference/processing.
6. Model detect từng mặt/box/class/confidence.
7. Camera nhận JSON, đọc hết body rồi chốt RTT/total.
8. Dashboard tự hiển thị annotation/detections, không reload trang.
9. Camera gửi /metrics/report; history/summary xuất client timings.
10. Test1 người: lần lượt đeo đúng/không đeo/đeo dưới mũi.
11. Test2 người: một đeo đúng, một không đeo; kiểm tra2 box riêng.
12. Test3 người: mỗi người một nhãn; cần thấy3 box/3nhãn cùng frame.
13. Test>=4 người, trộn các nhãn; ghi count thật và lỗi bỏ sót.
14. Mở /capture và /stream chứng minh camera IoT thật.
15. Đối chiếu Serial/JSON/history theo request_id; xuất CSV nếu cần.

## Bảng thực nghiệm thiết bị

Điền sau khi chạy thật; các ô chưa đo giữ trống:

| Case | Mặt nhìn rõ | Model count | Nhãn các box | Inference | RTT | Total | Screenshot/request_id |
|---|---:|---:|---|---|---|---|---|
| 1 người | 1 | | | | | | |
| 2 người | 2 | | | | | | |
| 3 người/3 nhãn | 3 | | | | | | |
| Nhiều người | >=4 | | | | | | |

Ghi đúng sai/bỏ sót, không chỉnh kết quả thủ công.
Confidence không phải accuracy. Người che mặt/quay lưng có thể không có box.
scripts/test_multi_person.py là kiểm thử phụ ảnh test thật, không thay live camera.

## Screenshot cần chụp

- Raw/audit lỗi gốc, split và phân bố lớp sau xử lý.
- Train command/device/log hoàn tất/curves/results.csv.
- Test precision/recall/mAP từng lớp, confusion matrix/PR.
- Board AI Thinker, compile/upload, Serial WiFi/IP/detection/latency.
- Camera /capture và /stream.
- Dashboard1/2/3/nhiều người; frame cả ba nhãn.
- Swagger JSON từng box/class/confidence/request_id.
- History/chart/summary cùng request Serial; null nếu chưa có số đo.
- Cloud URL và độ trễ cloud nếu đã deploy thật.

## Sự cố thường gặp

| Triệu chứng | Kiểm tra |
|---|---|
| 503 | Train/checkpoint/classes, restart API |
| WiFi lỗi | SSID/password/2.4GHz/nguồn |
| Timeout | LANIP, --host0.0.0.0, firewall/router isolation, CPU đang train |
| 401 | API key server/firmware/dashboard khớp |
| 400 image | JPEG/pixel limit/byte count |
| RTT trống | /metrics/report chưa đến hoặc thất bại |
| Stream không mở | IPcamera; HTTPS chặn HTTP, mở tab camera riêng |
| Bỏ sót mặt | Khoảng cách/ánh sáng/imgsz/conf/model |
| Brownout/reset | Nguồn5V/cáp ổn định; không tắt brownout để che lỗi |

## Các tình huống bổ sung bắt buộc

| Tình huống | Thao tác | Điều cần ghi nhận |
|---|---|---|
| Không có người | Camera hướng nền không mặt | Count thực tế, nếu false positive thì ghi lỗi; không sửa output |
| Đeo đúng | Một người, khẩu trang che mũi/miệng | Bbox/class/confidence thực tế |
| Không đeo | Một người nhìn camera | Bbox class1; cảnh báo Telegram khi violations |
| Đeo sai | Khẩu trang dưới mũi | Bbox class2, đối chiếu tiêu chí nhãn |
| Nhiều người | Trộn ba trạng thái cùng frame | Mỗi mặt có bbox và nhãn riêng; ghi bỏ sót |
| Backend tắt | Dừng terminal Uvicorn có kiểm soát | Dashboard disconnected, Serial HTTP lỗi; camera /capture vẫn thử được |
| Telegram | Token mới, bot đã /start, đủ cooldown | Ảnh annotate khớp request_id/caption, không gửi mọi frame |
| Telegram lỗi | Tắt TELEGRAM_ENABLED hoặc test mock timeout | Inference/history vẫn chạy; UI báo trạng thái thật |
| History | Sau response client gửi report | RTT/e2e có số đo, report mất giữ null, CSV khớp request_id |

Chụp thêm trang camera /flash ON/OFF, manifest thu ảnh thực tế, Telegram caption/box
(không để token trong screenshot), dashboard khi backend offline.
Các tình huống live này chưa được xác nhận bởi unit test hoặc compile;
điền kết quả sau khi có model đầy đủ và ESP32 thật.
