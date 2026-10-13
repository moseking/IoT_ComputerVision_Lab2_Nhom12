# Telegram cho demo camera

Theo yêu cầu hiện tại, `.env` đã dùng token và chat ID trong `D:\PTIT\IoT_UD\lab2\lab2.ino`. Không cần tạo bot mới hoặc nhập lại token. Bot là [@ESP32_lab2_ptit_bot](https://t.me/ESP32_lab2_ptit_bot). Token/chat ID chỉ giữ local; không ghi trong tài liệu, screenshot hoặc log.

Backend xử lý Telegram bằng [Bot API HTTPS chính thức](https://core.telegram.org/bots/api), ESP32 phục vụ camera/flash. Đây là bản chạy local; cloud thực hiện sau theo yêu cầu của bạn. Cấu hình đang bật: `TELEGRAM_ENABLED=true`, `TELEGRAM_ALERT_MODE=violations`, cooldown 30 giây, commands enabled. `ESP32_CAMERA_URL=http://192.168.137.28`.

## Test ngay

1. Giữ backend đang chạy tại `http://127.0.0.1:8000/`; nếu đã tắt, chạy `scripts/start_demo.ps1` trong thư mục project. Script đọc cấu hình Telegram local. Có thể dùng `scripts/start_demo.ps1 -NoTelegram` để tắt riêng phiên demo.
2. Camera phải bật, kết nối Wi-Fi và có `/capture` trên IP đã cấu hình. Dashboard: Ctrl+F5, Chụp + nhận diện, đưa mặt không đeo/đeo sai vào khung hình.
3. Mở bot cũ trong Telegram, nhắn `/start` nếu cần. Khi YOLO trả class 1 hoặc 2, backend gửi ảnh có box và caption. Nếu model bỏ sót hoặc trả class 0, chế độ violations không gửi cảnh báo. Các cảnh báo tự động cách nhau ít nhất 30 giây.
4. Nhắn `/result` để nhận ảnh YOLO gần nhất; `/photo` lấy ảnh camera gốc mới, chưa chạy YOLO. `/flash on` và `/flash off` đặt trạng thái đèn. `/flash` đảo đèn. Chỉ chat ID đã cấu hình được điều khiển; các lệnh trực tiếp cách nhau tối thiểu 2 giây.

Không chạy đồng thời hai backend polling cùng bot; nếu firmware Lab 2 còn polling Telegram bằng token đó, dừng polling cũ hoặc Upload firmware `esp32_cam` của project (firmware mới không chứa token). Đổi model/token backend không cần Upload Arduino. Để ngừng board tự gửi ảnh mỗi 2 giây và áp dụng URL mới, cần Upload firmware rồi reset.

## Nội dung và trạng thái

Ảnh cảnh báo và `/result` dùng đúng JPEG annotate của request được chọn, kèm request_id, thời gian UTC, số mặt theo lớp, confidence và latency thực đo. [sendPhoto](https://core.telegram.org/bots/api#sendphoto) hỗ trợ upload ảnh multipart; caption được giới hạn theo API. Confidence không phải accuracy.

Hàng đợi tối đa 2 ảnh; gửi nền, không chặn inference. Worker chờ báo cáo latency tối đa 3 giây; số đo chưa có ghi “chưa đo”. Thời gian gửi Telegram không nằm trong inference_ms/toàn trình camera. Cooldown tính từ lúc đặt lịch, kể cả gửi lỗi; không retry ảnh cũ.

`GET /telegram/status` và dashboard cho biết trạng thái thật. `ready/verified` nghĩa là bot đã qua getMe; `sent_count/last_sent` chỉ tăng sau sendPhoto thành công. Đã xác minh getMe/getChat, không có webhook, và backend đã ghi nhận gửi một cảnh báo thật lúc 22:13:12 ngày 11/10/2026. Xem `reports/camera_v8/runtime_verification.json`; đây là bằng chứng Bot API trả thành công, không phải xác nhận người nhận đã đọc.

URL camera chỉnh trên dashboard không tự sửa `.env` cho Telegram: khi IP đổi, sửa `ESP32_CAMERA_URL` rồi restart backend. Lỗi camera không làm YOLO/Telegram tắt; kiểm IP, Wi-Fi và nguồn camera. Lỗi Telegram được ghi tên loại lỗi để tránh lộ token. Unit test dùng MockTransport để kiểm đúng bytes annotate, cooldown, lọc chat và timeout; việc gửi thật được báo cáo riêng.

## Gửi khi bấm Chụp + nhận diện

Theo yêu cầu bổ sung, dashboard gửi `send_telegram=true` trên chính POST `/camera/predict` của nút chụp. Backend xếp hàng ảnh annotate của request đó đúng một lần, bỏ qua lọc class/cooldown tự động nhưng vẫn giới hạn hàng đợi 2 ảnh; không chụp lại để gửi. Live không đặt flag này. `last_sent_request_id` chỉ đổi sau sendPhoto thành công để trang xác nhận đúng ảnh đang giữ. Thử thật đã gửi frame 0 detection thành công lúc 22:30:59 ngày 11/10/2026; `manual_telegram_verification.json` giữ request_id và hash. Không thêm tự chụp/gửi ảnh định kỳ 30 giây.
