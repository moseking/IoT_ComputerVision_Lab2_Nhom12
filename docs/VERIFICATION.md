# Báo cáo lịch sử kiểm thử ngày 2026-10-08

Trạng thái hiện tại ngày 2026-10-11 nằm ở [BASELINE_RESULT.md](BASELINE_RESULT.md). Nội dung bên dưới là snapshot cũ một epoch, không mô tả checkpoint đang publish hiện nay.

# Kiểm chứng cuối — 2026-10-09

Làm trực tiếp trong D:\PTIT\IoT_UD\Lab2_FaceMask, không tạo project Lab 3 khác.
Source Lab 2 tham chiếu: D:\PTIT\IoT_UD\lab2\lab2.ino (và lab2_1 để đối chiếu camera).
Giữ raw dataset, checkpoint smoke cũ và lịch sử SQLite; không chép credentials từ firmware cũ.

| Kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| Baseline source cũ | 12/12 pass | results/verification/baseline-2026-10-09.log |
| Toàn bộ test cuối | 30/30 pass, 0 skip khi đặt TEST_MODEL_PATH | results/verification/tests-2026-10-09.log |
| YOLO prepared audit mới | 843 ảnh / 3970 box, 0 lỗi / 0 cảnh báo | results/prepared_dataset_report.json |
| Classes | 0 with_mask, 1 without_mask, 2 mask_weared_incorrect | dataset/data.yaml và prepared report |
| Split | train591 / val126 / test126 | prepared report |
| JavaScript syntax | node --check pass | web/app.js |
| Dependency | pip check: No broken requirements found | lệnh kiểm tra thực tế |
| Firmware | Compile thành công esp32:esp32:esp32cam | firmware-compile-2026-10-09.log |
| Arduino-ESP32 / ArduinoJson | 3.3.12 / 7.4.3 | package đã cài, CLI compile |
| Bộ nhớ firmware | 1.178.201 bytes flash37%; 60.816 bytes RAM tĩnh18% | compile log |
| HTTP thật | health/model/Telegram/latest/JPEG/history pass; corrupt400 và null metrics | http-v2-report.json |
| Dashboard thật | File chooser/upload dataset thật, ảnh load, history/summary đo thật | dashboard-v2-verification.jpg |
| Backend tắt | UI API DISCONNECTED, model/Telegram không xác nhận, lịch sử cuối giữ lại | dashboard-v2-offline.jpg |
| Thu ảnh manifest | Mock HTTP trả bytes ảnh dataset thật; SHA/dimensions khớp; không label/train | CaptureTests |
| Telegram | Mock getMe/sendPhoto/commands/timeout/cooldown/allowlist, JPEG annotate khớp | TelegramTests |
| Atomic report | 2 report cùng request không ghi đè phép đo đầu | StoreTests |
| Checkpoint guard | Một epoch không publish; không ghi đè ngầm; archive khi thay rõ ràng | ModelContractTests, train-guard log |

Mock chỉ dùng trong unit test. Runtime không có prediction hoặc ảnh giả.
Test HTTP/browser chạy checkpoint thật, dùng source=test_client/dashboard, database riêng
results/verification/network_latency_v2.db; không giả làm nguồn esp32.

## Checkpoint thực tế

Chỉ dùng checkpoint kiểm thử:
results/training/verification_2026-10-08T164330.145012_0000/weights/best.pt,
YOLOv8n train từ đầu trên dataset project 1 epoch, imgsz320, CPU.
Last.pt và metadata cùng run vẫn giữ nguyên.
Lượt đánh giá trước ở results/verification/evaluation có precision/recall/mAP đều0.
HTTP ảnh test có ground truth1/2/3/51 mặt: model_detected đều0; report tại
results/verification/multi_person_v2/report.json. Không khẳng định nhận dạng nhiều người
đạt chất lượng từ các test này. Unit test nhiều box chỉ xác minh chuyển đổi/schema/annotate.

**Chưa có checkpoint demo đạt chất lượng trong models/.**
Cần train đầy đủ, đánh giá held-out test và camera thực tế trước khi chọn demo.
Metadata train mới không tự xác nhận demo_eligible; việc đánh giá/accept thuộc bước bàn giao thực nghiệm.

## Lệnh chạy bằng Python đã cài trong project

```powershell
cd D:\PTIT\IoT_UD\Lab2_FaceMask
.\.venv\Scripts\python.exe scripts/audit_prepared_dataset.py
.\.venv\Scripts\python.exe scripts/train.py --pretrained --epochs 100 --imgsz 640 --batch 8 --device auto
.\.venv\Scripts\python.exe scripts/evaluate.py --model models/face_mask_best.pt --imgsz 640 --device auto
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1
```

--pretrained dùng YOLOv8n COCO công khai để khởi tạo rồi fine-tune dataset project.
Không truyền --pretrained nếu muốn train từ đầu. Đây là lệnh cho lượt đầy đủ CHƯA chạy.
PyTorch hiện cài torch CPU; DEVICE auto sẽ chọn CPU trên môi trường hiện tại.
NVIDIA/CUDA cần cài đúng PyTorch GPU theo phần README.

Firmware compile đã chạy:
```powershell
& 'D:\Arduino\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe' compile --fqbn esp32:esp32:esp32cam --library 'D:\PTIT\IoT_UD\lab2\libraries\ArduinoJson' esp32_cam
```

## Cần bạn thực hiện để demo thật

1. Train đầy đủ và xem metrics/test từng lớp; không chọn ngưỡng bằng tập test.
2. Điền WiFi/IP laptop/API/TLS vào config.h, nạp qua CAM-MB và mở Serial115200.
3. Xác nhận /, /capture, stream port81 và /flash trên ESP32 thật, thử backend offline.
4. Điền token mới/chat ID vào .env, bật Telegram rồi restart server, kiểm tra sendPhoto thật.
5. Thu ảnh camera riêng và gán nhãn đúng nếu chất lượng camera chưa đạt.
6. Chạy các tình huống docs/DEMO.md, ghi detection/latency thực tế và lỗi bỏ sót.
7. Cloud chỉ triển khai sau khi local ổn và có thông tin nền tảng, theo docs/CLOUD.md.

Chưa kiểm thử: upload firmware/Serial/camera/WiFi/flash thật, Telegram thật, GPU,
Docker build/run hoặc Cloud. Không có token mới và IP/thiết bị camera để kiểm thử các phần đó.
Backend kiểm thử port8001 và tab browser đã được dừng/đóng sau xác minh.
