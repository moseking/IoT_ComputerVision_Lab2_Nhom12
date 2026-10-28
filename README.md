# IoT Face Mask Detection — Nhóm 12

Hệ thống nhận diện từng khuôn mặt và phân loại khẩu trang bằng YOLOv8n đã
fine-tune. Hỗ trợ webcam laptop, ESP32-CAM, ảnh JPEG/PNG, chụp một ảnh hoặc Live,
hiển thị box, gửi Telegram và lưu kết quả cùng độ trễ trong SQLite.

| ID | Class | Ý nghĩa |
|---:|---|---|
| 0 | `with_mask` | Che kín mũi và miệng |
| 1 | `without_mask` | Không đeo khẩu trang |
| 2 | `mask_weared_incorrect` | Hở mũi/miệng, kéo xuống cằm hoặc treo một bên tai |

## Thành phần và cấu trúc

```text
Lab2_FaceMask/
├── README.md, .env.example, .gitignore
├── requirements.txt              # Backend local và công cụ dữ liệu/train
├── server/                       # API local, inference và các module API cloud
├── web/                          # Dashboard và webcam trình duyệt
├── esp32_cam/                    # Firmware và config.h.example
├── cloud/
│   ├── cloudrun/                 # Dockerfile, ignore và dependencies GCP
│   └── render/                   # Gói Render tương thích
├── models/
│   ├── camera_v7/                # Face Mask V7 .pt và metadata
│   └── cloud_v7/                 # V7 ONNX và manifest/checksum
├── data/raw/, data/processed/    # Dataset, nhãn, review và manifest trên máy
├── dataset/                      # Bộ YOLO legacy, giữ cho pipeline/test hiện có
├── scripts/                      # Xử lý dữ liệu, train/evaluate và vận hành
├── tests/                        # Python unittest và kiểm tra JavaScript
├── results/training/             # args, metrics, checkpoint và metadata các run
├── results/evaluation/           # Kết quả đánh giá đã lưu
├── reports/                      # Provenance/ledger còn được dataset/model dùng
├── Dockerfile, render.yaml       # API cloud Render
└── Dockerfile.local              # Backend local trong container
```

Giữ các tên `server`, `web`, `data`, `dataset`, `scripts` để đường dẫn hiện có
tiếp tục hoạt động. Dataset, checkpoint cũ, `.venv`, cache và output lớn được giữ
trên máy, loại khỏi Git theo `.gitignore`; không nằm trong source chia sẻ mặc định.
Tài liệu hướng dẫn chính thức là README này. `cloud/render/README.md` là template
được `prepare_cloud.py` đọc; README trong dataset là output/provenance của pipeline.

## Local và cloud

```text
Webcam laptop / ESP32-CAM / ảnh kiểm thử
                 ↓
Dashboard local → Backend local
                 ├── Máy tính: model Face Mask V7 .pt
                 └── Google Cloud Run: API + V7 ONNX
                              ↑
                 Supabase Storage private: file model
```

Dashboard, SQLite và Telegram chạy trên laptop. Backend local giữ khóa cloud,
gửi ảnh tới GCP và nhận box/nhãn/ảnh đã vẽ. Trình duyệt không nhận khóa GCP.
Chọn Cloud mà gặp lỗi sẽ báo lỗi; không tự chuyển sang model local.
Render là lựa chọn tương thích riêng, chỉ dùng khi cấu hình URL/key Render.

## Cài đặt và chạy

Chạy tại thư mục dự án trong terminal PowerShell của VS Code. Dùng Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu chưa có `.env`, sao chép `.env.example` thành `.env`. Giữ nguyên `.env` khi
máy đã được cấu hình; không ghi đè bằng file mẫu. Các biến chính:

| Biến | Công dụng |
|---|---|
| `MODEL_PATH` | `models/camera_v7/face_mask_best.pt` cho inference local |
| `API_BASE_URL` | `http://127.0.0.1:8000` cho dashboard local |
| `DEFAULT_CAMERA_SOURCE` | `webcam` hoặc `esp32` |
| `DEFAULT_INFERENCE_TARGET` | `local` hoặc `cloud` |
| `CLOUD_API_URL` | URL API Cloud Run/Render, không có `/docs` ở cuối |
| `CLOUD_API_KEY` | Khóa của API cloud, chỉ đặt phía backend |
| `CLOUD_MODEL_SHA256` | SHA-256 của V7 ONNX ở phần model bên dưới |
| `CLOUD_TIMEOUT_SECONDS` | Timeout gọi cloud; bản demo dùng `120` |
| `LATENCY_DB_PATH` | SQLite local; mặc định `data/latency.db` |
| `ESP32_CAMERA_URL` | IP/URL từ Serial Monitor, chỉ cần khi dùng ESP32 |
| `TELEGRAM_ENABLED` | Bật/tắt gửi Telegram |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Thông tin bot và nơi nhận |

Chạy backend và dashboard:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Mở [dashboard local](http://127.0.0.1:8000/) bằng Chrome/Edge. Chọn
**Webcam laptop → Google Cloud Run**, bấm **Bật webcam** và cấp quyền camera,
rồi **Chụp + nhận diện** hoặc **Live nhận diện**. Webcam không cần IP.
Sau thay đổi `.env`, khởi động lại backend; Ctrl+F5 để cập nhật giao diện.

**Xem ảnh** dừng Live và giữ kết quả. Chụp một ảnh cũng giữ nguyên kết quả sau
khi nhận xong. Webcam được tắt khi dừng/đổi nguồn/chuyển tab nền. Flash chỉ dành
cho ESP32-CAM. **Lưu ảnh gốc** tải đúng JPEG của frame webcam/ESP32 đang xem.
Ảnh offline được gửi qua mục **Kiểm thử ảnh offline**.

Nếu dùng ESP32-CAM, tạo `esp32_cam/config.h` từ file mẫu, đặt Wi-Fi và URL backend,
nạp `esp32_cam.ino`, đọc IP trong Serial rồi cấu hình `ESP32_CAMERA_URL`.
Backend nhận kết nối từ board cần bind `--host 0.0.0.0` và cho phép cổng 8000
trên mạng nội bộ. Không đưa config Wi-Fi thật vào Git.

## Model và dữ liệu huấn luyện

| Artifact chính thức | SHA-256 |
|---|---|
| `models/camera_v7/face_mask_best.pt` | `722614b198153c75c2248d1e0e9f7af9a0420123d62d3438f0911d9aa95bbaef` |
| `models/cloud_v7/face_mask_v7.onnx` | `f1fc04ca98bace9f1a1a5537d83fc9c52574514c0641959084c94a9f0750cbf4` |

Input inference 512, confidence 0.35, IoU 0.45, NMS chung giữa class. ONNX là
bản xuất FP32/opset 17 của checkpoint Face Mask V7. `yolov8n.pt` tại root là
pretrained COCO dùng khởi tạo train, không phải model Face Mask phục vụ demo.

Dataset public gốc nằm ở `data/raw/facemask_detection`; bộ V7 đã chuẩn bị ở
`data/processed/facemask_camera_v7`. Giữ nguyên ảnh, TXT nhãn YOLO, `data.yaml`,
`split_manifest.json`, manifest review và ledger loại ảnh/box. Box đeo sai cần
bao quát mặt và phần khẩu trang liên quan. Ảnh đã dùng train/debug không phải
tập kiểm thử độc lập. Checkpoint chọn theo validation, không chọn theo test.

Dataset không được đóng gói trong Git. Thành viên cần bản dữ liệu riêng có quyền
sử dụng; chép vào đúng đường dẫn trước khi audit/train. Các `data.yaml` có thể chứa
đường dẫn tuyệt đối của máy tạo dữ liệu: điều chỉnh bản sao trên máy mới hoặc chuẩn
bị lại dữ liệu bằng script, không coi việc clone source là đã có dataset.

```powershell
.\.venv\Scripts\python.exe scripts/audit_prepared_dataset.py --data data/processed/facemask_camera_v7/data.yaml --output results/prepared_dataset_report.json
.\.venv\Scripts\python.exe scripts/train.py --data data/processed/facemask_camera_v7/data.yaml --init-checkpoint models/camera_v7/face_mask_best.pt --epochs 12 --imgsz 512 --batch 8 --workers 0 --seed 42 --optimizer AdamW --lr0 0.00008 --patience 5 --device auto --run-name facemask_continue --no-publish
```

Lệnh thứ hai tạo run fine-tune mới, không phải lệnh tái tạo byte-identical V7 và
không ghi đè model chính thức. Để kiểm tra cấu hình V7 đã dùng, xem
`models/camera_v7/model_metadata.json`, `results/training/<run>/args.yaml`,
`results.csv`, metadata và manifest dataset. Các script chuyển VOC, review,
build dataset, evaluate và xuất cloud được giữ trong `scripts/`; xem `--help`
trước khi dùng. Chạy demo không cần train lại.

## API cloud và độ trễ

Cloud Run: `https://facemask-v7-ppk526twwq-as.a.run.app`.
[Health](https://facemask-v7-ppk526twwq-as.a.run.app/health) và
[Swagger](https://facemask-v7-ppk526twwq-as.a.run.app/docs) dùng để kiểm tra API.
`GET /model/info` và `POST /predict` yêu cầu header `X-API-Key`; predict nhận
multipart field `file` là JPEG/PNG. API cloud không phục vụ dashboard hay SQLite.

Supabase private lưu object `facemask-models/v7/face_mask_v7.onnx`. Container
tải model khi startup, kiểm SHA-256, nạp ONNX Runtime rồi mới sẵn sàng. Dockerfile
Cloud Run không nhúng model/secret vào image. Signed URL là secret: gói này kiểm
thời hạn tối đa 7 ngày, còn tối thiểu 1 giờ khi startup; nên gia hạn trước 24–48 giờ.
URL hết hạn làm container mới không tải được model. Gia hạn và cập nhật revision
là thao tác cloud riêng, cần được duyệt; không tự đổi bucket thành public.

| Số đo | Phạm vi |
|---|---|
| Inference | Preprocess + model + NMS tại nơi suy luận |
| Round trip cloud | Backend laptop gửi ảnh → nhận hết response cloud |
| End-to-end | Bắt đầu chụp/gửi ảnh → nhận đủ kết quả trên client, không gồm vẽ giao diện |
| Capture webcam | Lấy frame và mã hóa JPEG trong trình duyệt |

Lịch sử SQLite lưu từng request và số mặt theo ba nhãn; không phải số người duy nhất.
Nút chụp yêu cầu gửi ảnh qua Telegram nếu đã cấu hình; Live theo chính sách
`TELEGRAM_ALERT_MODE` và cooldown. Confidence hiển thị trên box là độ tin cậy của
dự đoán, không phải độ chính xác đo trên toàn bộ dataset.

## Kiểm thử

Tests dùng database/thư mục tạm và tắt Telegram. Không kiểm thử bằng database demo:

```powershell
$env:TELEGRAM_ENABLED='false'
$env:TELEGRAM_COMMANDS_ENABLED='false'
$env:CLOUD_API_URL=''
$env:CLOUD_API_KEY=''
$env:TEST_MODEL_PATH=(Resolve-Path models/camera_v7/face_mask_best.pt).Path
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py"
node --test tests/test_live_ui.cjs tests/test_webcam.cjs
```

`TEST_MODEL_PATH` bật các ca inference thật; test dữ liệu cần dataset local.
Sau khi test, đóng terminal đó hoặc bỏ các biến tạm trước khi chạy demo cloud.

Kiểm tra build Linux cục bộ bằng Docker Desktop, không push/deploy:

```powershell
docker build --platform linux/amd64 -f cloud/cloudrun/Dockerfile -t facemask-cloudrun-local:v7 .
```

Giữ `.env`, `config.h`, signed URL, API key và credentials ngoài Git. Các file
ONNX/PT chính thức được `.gitignore` cho phép riêng; dataset và output lớn vẫn
được giữ local để chia sẻ qua kênh lưu trữ riêng.
