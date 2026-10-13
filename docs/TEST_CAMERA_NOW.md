# Test camera và Telegram ngay

Project: `D:\PTIT\IoT_UD\Lab2_FaceMask`. Dashboard: http://127.0.0.1:8000/. Camera gần nhất: http://192.168.137.188/. Luôn kiểm tra IP trong Serial sau khi reset. Backend hiện dùng V7, V8 đã train và lưu riêng; xem [kết quả V8](../reports/camera_v8/FINAL_RESULT.md). Không cần train lại để test.

## Chụp một ảnh và gửi Telegram

1. Bật camera, giữ Wi-Fi cùng laptop. IP phải khớp Serial. Ctrl+F5 trên dashboard sau cập nhật.
2. Bấm **Chụp + nhận diện**. Server lấy một JPEG, chạy YOLO, giữ nguyên ảnh có box và gửi chính ảnh đó tới bot cũ [@ESP32_lab2_ptit_bot](https://t.me/ESP32_lab2_ptit_bot). Gửi cả khi 0 mặt hoặc tất cả đeo đúng; lượt chụp thủ công không chờ cooldown cảnh báo.
3. Dòng bên nút hiện “đang chờ gửi” khi xếp hàng; “Đã gửi ảnh này” chỉ sau Bot API xác nhận đúng request_id. Gửi nền có thể tới sau ảnh trên dashboard vài giây. Nếu cấu hình lỗi hoặc hàng đợi 2 ảnh đầy, trang báo chưa xếp hàng; vẫn giữ được kết quả YOLO.
4. Ảnh giữ nguyên để giảng viên kiểm tra. **Xem ảnh** dừng Live và giữ frame. **Lưu ảnh gốc** dừng Live, yêu cầu trình duyệt tải đúng JPEG/PNG chưa vẽ box của frame đang xem; kiểm mục tải xuống. Không lấy ảnh đã có box/chữ để train.
5. **Live nhận diện** liên tục lấy frame và hiển thị box thật. Cảnh báo tự động chỉ khi có class 1/2, tối đa một lần mỗi 30 giây. Không có lịch chụp/gửi định kỳ 30 giây. Chụp thủ công gửi một lần, không đồng thời gửi cảnh báo tự động trùng request.

Token/chat ID đã lấy từ `lab2/lab2.ino` theo yêu cầu của bạn và giữ trong `.env`, không cần tạo bot mới. Backend phải tiếp tục chạy và có Internet để gửi Telegram. `/photo` lấy ảnh camera gốc mới chưa YOLO; `/result` gửi kết quả YOLO gần nhất; `/flash on` / `/flash off` điều khiển đèn. [Hướng dẫn Telegram](TELEGRAM.md).

Nếu backend đã tắt:

```powershell
Set-Location 'D:\PTIT\IoT_UD\Lab2_FaceMask'
.\scripts\start_demo.ps1
```

Giữ terminal chạy. Script đọc cấu hình Telegram; `-NoTelegram` tắt riêng phiên này. Không khởi động hai backend cùng port hoặc hai tiến trình polling cùng bot.

## Arduino và đèn flash

Telegram và đổi model nằm trên laptop, không cần Upload Arduino cho hai thay đổi đó. Nếu Serial còn `Sending image / Response: -1 / connection refused`, board còn tự POST bằng chương trình/cấu hình cũ. Firmware project đã compile thành công, nhưng agent chưa Upload vào board. Mở `esp32_cam/esp32_cam.ino`, chọn AI Thinker ESP32-CAM/đúng COM rồi Upload để áp dụng:

```cpp
#define SERVER_URL "http://192.168.137.1:8000/predict"
#define METRICS_URL "http://192.168.137.1:8000/metrics/report"
#define INFERENCE_ENABLED false
```

`.1` là IP hotspot laptop quan sát, `.188` là IP camera người dùng cập nhật ngày 2026-10-12. Kiểm lại nếu đổi mạng. Sau Upload, tháo GPIO0–GND rồi reset, Serial 115200. Đường Chụp/Live dùng `/capture` nên không cần board tự POST. Nút flash đặt on/off rõ ràng, chỉ xác nhận sau camera trả JSON; timeout thì kiểm nguồn/IP/Wi-Fi. Chưa thấy flash bật tức thì không có nghĩa YOLO lỗi.

## Đối chiếu nhãn

0 che kín mũi và miệng; 1 không đeo; 2 hở mũi/miệng, kéo cằm hoặc treo một tai. Thử đủ sáng và cả hai tai, một/hai/ba người. Chỉ tính người thật, không poster. Confidence không phải accuracy; ghi lại ảnh bỏ sót/nhầm lớp, không sửa output để giả đúng. Model đọc cả frame; box vẽ là kết quả, không phải vùng crop đầu vào. Review nháp class 2 mặc định nới 35% hai bên, 10% trên, 35% dưới và clamp; bản train mới dùng 5 box context bao gồm mặt + khẩu trang thấy được, giữ nhãn review gốc.

26 camera cũ và 5 camera mới đã vào train/development. Dùng lượt chụp mới để đánh giá độc lập. Giữ raw JPEG không có chữ/box trong thư mục mới; queue 8768 đã đóng băng cho V8, không import tiếp vào snapshot đã train.

## Thử V8 hoặc quay lại V7/V3

V8 nhận đúng 5/5 box trên chính 5 ảnh train mới (V7 4/5), nhưng thêm 1 false positive và public validation tổng giảm 0.5315 → 0.5174. Vì thế giữ V7 mặc định; V8 là bản thử cho camera, chưa công nhận tốt hơn trên ảnh chưa thấy. V3/V7 nguyên vẹn.

Dừng terminal backend đang chạy trước khi đổi rồi chạy:

```powershell
Set-Location 'D:\PTIT\IoT_UD\Lab2_FaceMask'
.\.venv\Scripts\python.exe scripts/select_model.py camera-v8
.\scripts\start_demo.ps1
```

Đổi `camera-v8` thành `camera-v7` để quay lại hiện tại, hoặc `v3` để dùng baseline. Restart backend mới áp dụng, không cần Upload Arduino. Server warmup model trước khi báo ready; chờ Application startup complete. Live vẫn cần đợi tải JPEG thật từ camera. Thống kê history đang trộn các phiên model/kiểm thử, không dùng average lịch sử để so riêng V8.


## Live chậm — cập nhật 2026-10-12

Đã đổi IP camera mặc định sang `http://192.168.137.188`. Trình duyệt có thể đang giữ URL cũ trong localStorage: vào **Cấu hình kết nối → Camera URL → Áp dụng** theo IP Serial hiện tại. Mỗi lần Upload/reset camera, kiểm tra lại IP.

Mở đúng sketch `D:\PTIT\IoT_UD\Lab2_FaceMask\esp32_cam\esp32_cam.ino`. ESP32 chạy chương trình Upload thành công gần nhất; mở/Verify file khác không đổi chương trình trên board. Upload nhầm sketch khác có thể vẫn thấy camera nếu sketch đó có `/capture`, nhưng các chức năng flash/timing/auto-push có thể khác. Sau Upload, tháo GPIO0–GND nếu đang nối để nạp, rồi reset và xem Serial 115200.

Live đã giảm khoảng nghỉ sau frame từ 250 xuống 50 ms, không chờ POST ghi số đo, chỉ giữ một report đang gửi và report mới nhất đang chờ. Khi server/DB chậm, các report trung gian có thể không được gửi; số đo thiếu giữ null/“—”, không tự điền giá trị giả. Camera requests vẫn tuần tự, không xếp hàng nhiều frame. Khi tải camera lỗi, giữ ảnh/box cũ với nhãn **GIỮ ẢNH CŨ · ĐANG THỬ LẠI**, không coi đó là frame mới.

Khởi động lại server trong terminal VSCode để áp dụng warmup và IP mặc định mới:

```powershell
# Ctrl + C trong terminal server hiện tại, rồi:
cd 'D:\PTIT\IoT_UD\Lab2_FaceMask'
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Đợi server báo ready, Ctrl + F5 dashboard, kiểm tra Camera URL rồi bấm Live. Không cần train lại. Các thay đổi hôm nay nằm trên laptop; lần Upload này là để sửa việc đã nạp nhầm sketch.

Số đo trước sửa: một frame lấy ảnh 4249.419 ms, YOLO 79.019 ms, RTT 4377.9 ms. Probe trực tiếp `.188` ba lần đều ReadTimeout; ping ngắn trả 3/4 gói (20–382 ms). Đây là bằng chứng độ trễ phía camera/kết nối, không đủ kết luận riêng Wi-Fi, nguồn hay firmware. Nới box không xử lý lỗi đường truyền này.

Khi Live vẫn chậm: đặt ESP32-CAM gần hotspot, dùng nguồn ổn định, đóng các tab đang mở stream camera khác và thử lại sau khi đã nạp đúng sketch. Live hiện lấy JPEG rồi chạy YOLO từng frame, không phải video 30 FPS. Đọc dòng **Lấy ảnh … ms · YOLO … ms** phía trên để thấy phần nào chậm. Không hứa FPS cố định khi đường truyền chưa ổn định.
