# Kiểm tra connection refused và ảnh camera

Cập nhật 11/10/2026: backend cổng 8000 đã chạy, camera V7 đã load và `/camera/predict` từ camera 192.168.137.28 trả HTTP200. IP hotspot laptop 192.168.137.1. `Response:-1 / connection refused` trong Serial là lỗi kết nối outbound của firmware; board chạy config cũ cho đến khi Upload. Firmware local đã compile với `INFERENCE_ENABLED=false`; Chụp + nhận diện/Live dùng server kéo JPEG, không cần board tự POST. Chưa Upload vào board trong phiên này. [Hướng dẫn hiện tại](TEST_CAMERA_NOW.md).


Audit ngày 2026-10-09: netstat chưa có listener cổng 8000/8001 tại thời điểm kiểm tra.
Đây là trạng thái laptop lúc kiểm tra, chưa xác định mọi nguyên nhân lỗi trên ESP32.
Sau khi khởi động backend, HTTP thật được kiểm tra riêng trong verification.

## Backend trên laptop

```powershell
cd D:\PTIT\IoT_UD\Lab2_FaceMask
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Giữ terminal chạy. Trên laptop mở http://127.0.0.1:8000/health.
HTTP 200 với model_unavailable nghĩa là server đã chạy nhưng checkpoint thiếu;
POST /predict sẽ trả 503 cho đến khi có model hợp lệ và restart backend.

```powershell
netstat -ano -p tcp | Select-String ':8000'
ipconfig
```

Lấy IPv4 WiFi cùng mạng ESP32. SERVER_URL và METRICS_URL trong config.h dùng IP laptop.
API_BASE_URL phục vụ browser; 127.0.0.1 chỉ dùng browser trên chính laptop.
ESP32 dùng localhost sẽ gọi chính ESP32, không gọi laptop.
Mở http://IP_LAPTOP:8000/health từ điện thoại cùng WiFi trước khi nạp firmware.

| Hiện tượng | Kiểm tra |
|---|---|
| Connection refused | Server chưa chạy, sai IP/port, bind chỉ 127.0.0.1, process đã thoát |
| Timeout | WiFi khác subnet, guest/AP isolation, firewall TCP8000 mạng Private, VPN |
| 503 | Checkpoint thiếu/sai classes; đọc /health và /model/info |
| 401 | X-API-Key trong firmware/browser chưa khớp .env |
| 400/415 | Body ảnh hỏng hoặc content-type không đúng |
| 413 | JPEG/request lớn hơn MAX_UPLOAD_BYTES |
| 408 | Upload body quá UPLOAD_TIMEOUT_SECONDS |
| HTTPS lỗi | Root CA, tên host/certificate, thời gian NTP trên ESP32 |
| Cổng đã dùng | Dừng đúng process của bạn hoặc chọn cổng khác và sửa cả hai URL firmware |

Không tự mở firewall hoặc tắt bảo vệ; chỉ kiểm tra cấu hình mạng phù hợp cho môi trường lab.

## Camera

Trang / mới tự tải /capture ngay khi mở, có trạng thái load/error.
Capture thêm ?t=timestamp và header no-store để không hiển thị ảnh cache.
MJPEG chạy riêng port81; /stream port80 redirect sang http://IP_CAMERA:81/stream.
Camera HTTP server chạy task riêng, độc lập với loop gọi backend bị timeout.

1. Serial 115200, camera init phải thành công và có WiFi/IP hiện tại.
2. Mở http://IP_CAMERA/capture trực tiếp. Nếu 503, xem camera busy/init/PSRAM/RAM.
3. Kiểm tra cáp ribbon cắm đúng, nguồn 5V ổn định, chung GND; tháo GPIO0-GND sau nạp rồi reset.
4. Nếu ảnh tĩnh đúng nhưng stream lỗi, kiểm tra port81 và đóng các tab stream cũ.
5. Dashboard HTTPS không nhúng được camera HTTP do mixed content; mở camera LAN ở tab riêng.
   Một số browser cũng chặn truy cập mạng riêng từ trang public.
6. Flash GPIO4 dùng chung SD trên board; firmware này không dùng SD.
7. Mất WiFi: firmware retry và in IP khi kết nối lại; DHCP có thể đổi IP.
   Cập nhật ESP32_CAMERA_URL rồi restart backend và cấu hình dashboard.

Chưa có thử nghiệm trên phần cứng trong phiên này; compile thành công không xác nhận nguồn/cáp/WiFi.

## CUDA/HEIC/review/versioned datasets

Xem [BASELINE_RUNBOOK.md](BASELINE_RUNBOOK.md): wheel CPU→CUDA, pillow-heif/EXIF, reviewer/bbox guard, output no-overwrite, completed checkpoint không resume và IP/firewall Private. Không ép verified trong manifest để vượt guard.
