# Vận hành baseline và demo Lab 3

Tất cả lệnh PowerShell chạy từ `D:\PTIT\IoT_UD\Lab2_FaceMask`. Các dataset đã tồn tại không được ghi đè; dùng suffix phiên bản mới. Xem `BASELINE_RESULT.md` để biết bản hiện tại.

## Môi trường

`.venv` hiện có Python 3.12.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.174, OpenCV 4.14.0, Pillow 12.3.0, pillow-heif 1.8.0. Hardware report ở `reports/baseline_v1/hardware.json`. GPU test tensor thật và training đều dùng CUDA 0, RTX 3050 Laptop 6 GB; không phải RTX 3060. `pip check` kiểm tra môi trường sau cài CUDA.

Bản package pin để tái tạo stack đã test: requirements-cuda-tested.txt (pip install -r requirements-cuda-tested.txt); file này dùng index CUDA chính thức.

Nếu cài mới, Python 3.12 64-bit và driver NVIDIA tương thích CUDA 12.8:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# requirements không tự bảo đảm wheel GPU; thay torch CPU bằng wheel chính thức.
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA wheel lớn (~2.7 GB). Phiên này đường tải pip bị đứng nên tải ranged từ index PyTorch chính thức và kiểm SHA-256 khớp index trước cài; log ở reports/baseline_v1. Không cài lại nếu CUDA hiện đã hoạt động. CPU có thể chạy với `--device cpu --batch 4`, sẽ chậm hơn; không cam kết thời gian CPU khi chưa đo. Workers 0 an toàn trên Windows; batch 8 dùng khoảng 1.55 GiB CUDA khi train v1. RAM máy thấp trong phiên, tránh mở nhiều tiến trình train.

Nguồn: [PyTorch CUDA wheels](https://pytorch.org/get-started/locally/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [train](https://docs.ultralytics.com/modes/train/).

## Public và tọa độ

Public nguồn `data/raw/facemask_detection`; bộ strict `data/processed/facemask_public_yolo`. Audit thực tế 832 ảnh/3725 box, không clip lỗi. VOC trong bộ này được xử lý theo zero-based continuous: `xc=(xmin+xmax)/(2W)`, `yc=(ymin+ymax)/(2H)`, `w=(xmax-xmin)/W`, `h=(ymax-ymin)/H`. XML có thể khác convention ở dataset khác; không đổi convention chỉ để hợp thức hóa box vượt biên.

Trước train đã so sánh tọa độ XML/TXT và vẽ 24 mẫu GT, 8/class: `reports/baseline_v1/coordinate_verification.json`, `public_ground_truth_review.jpg`. Nhãn public theo nguồn, không được con người xác minh lại từng ảnh; có ảnh nhỏ/mơ hồ, đặc biệt class incorrect.

```powershell
.\.venv\Scripts\python.exe scripts/audit_prepared_dataset.py --data data/processed/facemask_public_yolo/data.yaml --output reports/manual/public_validation.json
```

## Phone/HEIC và review

Raw single_person có 91 with_mask, 59 without_mask, 130 wore_incorrect, tổng 280 (235 HEIC/45 JPEG). `custom_dataset.py` đọc HEIF thật qua pillow-heif, EXIF transpose, RGB, JPEG quality 93, max side 1600 và mở lại JPEG. Không đổi đuôi và không sửa raw. HEIF decoder có thể đã áp dụng orientation; metadata ghi kích thước gốc/oriented/processed thực tế. Test có HEIF codec thật và JPEG EXIF rotation fixture.

YuNet ONNX chính thức ở `.runtime/face_detector/face_detection_yunet_2023mar.onnx`; SHA ghi manifest. [Nguồn YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Nếu file mất, tải từ OpenCV Zoo (media GitHub raw), không thay bằng person detector.

```powershell
# Output v1 đã tồn tại; ví dụ dưới dùng bản MỚI, chưa có.
.\.venv\Scripts\python.exe scripts/custom_dataset.py --source data/raw/custom_capture/single_person --output data/processed/single_person_review_v2 --source-kind phone --reports reports/single_person_v2
# Tiếp tục review bản đang làm; không convert lại manifest cũ.
.\.venv\Scripts\python.exe scripts/review_annotations.py --folder data/processed/single_person_review_v1 --port 8765
```

Mở `http://127.0.0.1:8765`; nhập tên reviewer; xem full ảnh và từng mặt; giữ box đúng, xóa/vẽ lại box lệch; chọn một box rồi chọn class của riêng mặt đó. Kiểm tra mũi/miệng, người nền và mặt bị bỏ sót. Bấm **Xác minh + Sau** mới tạo provenance human_reviewed/reviewer/time. **Lưu nháp** không được train. **Loại + Sau** cho ảnh mờ/không chắc.

Tên single_person không bảo đảm chỉ một mặt: detector ban đầu 256 ảnh có 1 proposal, 17 có 2, 6 có 0, 1 có 3. Box nháp đôi khi quá nhỏ/không bao toàn mặt. App cho phép nhiều box khác class, chặn xác minh nếu bỏ proposal mà chưa đủ box tương ứng; ảnh ambiguous/false proposal nên loại cho đến khi sửa rõ. Những nhãn đã lưu từ trước khi bổ sung guard, nhưng thiếu box mặt nền, bị builder loại. Không dùng tên folder/confidence như bằng chứng class.

## Đóng dataset mới và leakage

```powershell
# v1–v5 đã tồn tại, giữ nguyên. Lệnh này tái xuất snapshot hiện đã audit thành version mới.
# Ledger pin hash manifest; nếu bạn review thêm, cần audit ledger mới trước khi xuất.
.\.venv\Scripts\python.exe scripts/build_baseline_dataset.py --review data/processed/single_person_review_v1 --review data/processed/multi_person_review_v1 --box-exclusions reports/baseline_v5/non_person_box_exclusions.json --image-exclusions reports/baseline_v6/image_qc_exclusions.json --output data/processed/facemask_baseline_v6 --reports reports/baseline_v6
.\.venv\Scripts\python.exe scripts/audit_prepared_dataset.py --data data/processed/facemask_baseline_v6/data.yaml --output reports/baseline_v6/validator.json
```

Builder pin mapping, kiểm hash raw/processed, class/bbox/provenance; chỉ human verified. Public split giữ nguyên 584/124/124; phone/multi cùng nhóm chụp chỉ vào train. Theo target chỉ người thật, v5 bỏ đúng83box poster đã kiểm trực quan ở76ảnh; giữ các box người thật phía sau. Ledger exact snapshot giữ nguyên nhãn review gốc, ghi rõ audit bởi assistant và provenance human của box còn lại; không biến draft thành verified. SHA/pixel duplicate loại (cả giữa các nguồn); same-image classes xung đột loại tất cả bản; dHash<=4 với public holdout loại. Near-duplicate phone giữ cùng train; 99 cặp gần trùng phát hiện ở 280 ảnh. Không có person ID cho public và phone có nhiều người nền: chưa chứng minh tách danh tính tuyệt đối. Không tạo nhóm giả để đạt tỷ lệ 70/15/15. Manifest.csv/split_manifest ghi source/hashes/reviewer/exclusions. Dataset/version frozen lúc build; review thêm không tự thay đổi dataset đã train.

## Train/resume

```powershell
# Reproduce v1 với official COCO initialization trong run mới.
.\.venv\Scripts\python.exe scripts/train.py --data data/processed/facemask_baseline_v1/data.yaml --pretrained --epochs 30 --imgsz 512 --batch 8 --workers 0 --device 0 --seed 42 --patience 8 --run-name baseline_v1_repeat --no-publish
# Bổ sung custom đã review: fine-tune checkpoint mặt nạ trong run mới, không resume run đã hoàn tất.
.\.venv\Scripts\python.exe scripts/train.py --data data/processed/facemask_baseline_v6/data.yaml --init-checkpoint results/training/baseline_v5_2026-10-11T110715.751030_0000/weights/best.pt --epochs 12 --imgsz 512 --batch 8 --workers 0 --device 0 --seed 42 --patience 5 --optimizer AdamW --lr0 0.00008 --lrf 0.2 --run-name baseline_v6_repeat --no-publish
```

Scratch: bỏ `--pretrained` và `--init-checkpoint`. Fine-tune ít dữ liệu mới có thể dùng AdamW/LR thấp như lệnh trên để giảm quên dữ liệu public; auto optimizer sẽ tự chọn LR, không dùng lr0 do bạn truyền. Script kiểm mọi state tensor của init checkpoint trước optimizer bước đầu, log initialization_verification.json.

Resume run bị ngắt: `scripts/train.py --data <YAML_CUA_RUN_CU> --resume <LAST_PT_CUA_RUN_BI_NGAT> --no-publish`; thay placeholders bằng file thực tế. Last.pt của run hoàn tất bị strip optimizer, guard từ chối và hướng dẫn `--init-checkpoint`. Run mới có timestamp; không ghi đè checkpoint. Chọn best theo validation fitness; test chỉ đánh giá sau khi khóa checkpoint và conf 0.35. `--replace-published` lưu archive trước thay model; không tự publish run 1 epoch. Backend cần restart để nhận model mới.

## Evaluate/inference

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --model models/face_mask_best.pt --data data/processed/facemask_baseline_v6/data.yaml --split val --imgsz 512 --batch 8 --device 0 --conf 0.35 --output results/evaluation/manual_v1/validation
.\.venv\Scripts\python.exe scripts/evaluate.py --model models/face_mask_best.pt --data data/processed/facemask_baseline_v6/data.yaml --split test --imgsz 512 --batch 8 --device 0 --conf 0.35 --output results/evaluation/manual_v1/test
.\.venv\Scripts\python.exe scripts/test_inference.py --image data/processed/facemask_baseline_v6/images/test/public_maksssksksss852.png --api-url http://127.0.0.1:8000 --output results/verification/private/manual_single
```

Output mới có JSON/CSV/Markdown, predictions.json, plots/confusion matrix và examples GT xanh/pred đỏ. Matching class-aware IoU>=0.5, một GT chỉ match một pred; nhầm class là cả FP lẫn FN. AP quét conf>=0.001; FP/FN riêng tại conf 0.35. Latency per-image loại warmup, CUDA synchronize, gồm preprocess/forward/NMS, không gồm decode. Framework speed batch có định nghĩa khác. Benchmark trong môi trường máy đang dùng, không cam kết FPS ESP32.

## Camera qualitative và multi-person

Khi khảo sát đầu chưa có ảnh ESP32. Hiện đã nhận26JPEG tại `data/raw/camera32_capture`, chuyển26/26; giữ nguyên raw. Review `data/processed/esp32_review_v1` đang chạy tại http://127.0.0.1:8767/. Camera đã xác minh26/26 và đánh giá; giữ ngoài train. Bộ26 đã dùng debug hậu xử lý nên kết quả sau sửa là phát triển/qualitative, cần capture lượt mới cho kiểm độc lập. Xem TEST_CAMERA_NOW.md và reports/esp32_v1/FINAL_RESULT.md. Folder `data/raw/esp32_eval_v1` vẫn là nơi cho lượt capture bằng script sau này.

```powershell
# Thay địa chỉ dưới bằng IP camera từ Serial thật.
.\.venv\Scripts\python.exe scripts/capture_dataset.py --camera http://192.168.1.50 --count 20 --interval 2 --output data/raw/esp32_eval_v1
.\.venv\Scripts\python.exe scripts/custom_dataset.py --source data/raw/esp32_eval_v1 --output data/processed/esp32_review_v2 --source-kind esp32 --reports reports/esp32_v2
.\.venv\Scripts\python.exe scripts/review_annotations.py --folder data/processed/esp32_review_v2 --port 8768
```

Capture lưu ảnh và manifest timestamp/hash/size/status; conversion tạo contact sheets, orientation/blur/decode/proposals và review. Review trên port riêng; builder chặn source_kind=esp32 để giữ camera độc lập. Ưu tiên smoke/qualitative domain evaluation; nếu dùng ảnh camera để điều chỉnh model/threshold thì không gọi bộ đó là test độc lập.

Multi-person snapshot `data/processed/multi_person_review_v1` có1006ảnh (1001JPEG/5HEIC), chuyển1006/1006. Bạn đã review846verified/158excluded/2draft. V6 xuất683ảnh sau dedup, class-conflict gate và giữ ngoài 3 ảnh chuyển trạng thái khó xác định (468–470), lọc83box poster theo scope chỉ người thật. Ảnh255 giữ nguyên nhãn đúng [1,0]. V5 dừng sau7epoch để audit, checkpoint best được dùng khởi tạo run v6; xem BASELINE_RESULT để biết trạng thái hoàn tất và model được chọn. Không đổi nhãn review gốc. Che kín cả mũi và miệng là đeo đúng; hở mũi/miệng hoặc kéo xuống cằm là đeo sai. Giữ bản review gốc; bạn vẫn review tiếp được nhưng không tự thay đổi dataset/model đã đóng băng.

```powershell
# Chỉ chạy nếu output v2 chưa có và đợt copy đã ổn định.
.\.venv\Scripts\python.exe scripts/custom_dataset.py --source data/raw/custom_capture/multi_person --output data/processed/multi_person_review_v2 --multi-person --source-kind multi --reports reports/multi_person_v2
.\.venv\Scripts\python.exe scripts/review_annotations.py --folder data/processed/multi_person_review_v2 --port 8769
```

Ảnh nằm ở folder không biết class: mỗi proposal class null, phải chọn riêng 0/1/2 từng mặt; mixed-state hợp lệ. Không train multi-person từ draft. `build_baseline_dataset.py` nhận nhiều `--review`; đối chiếu exact SHA/pixel giữa phone/multi, bảo toàn public holdout và provenance từng nguồn. Giữ ledger poster đúng snapshot nếu xuất theo target chỉ người thật. Không chứng minh tách danh tính hay session độc lập; toàn bộ custom bổ sung hiện chỉ train nên không leak giữa custom train/test. Không gộp nhiều manifests bằng tay rồi mất provenance. Khi cùng người/capture chưa đủ nhóm, tiếp tục giữ custom ở train, public val/test ổn định, bổ sung camera holdout độc lập.

## Backend/dashboard/latency

`.env`: MODEL_PATH=models/face_mask_best.pt, IMAGE_SIZE=512, conf 0.35, IoU 0.45. `python -m uvicorn server.main:app --host 0.0.0.0 --port 8000`, một worker. Model load một lần, mutex predictor, ảnh latest RAM; SQLite chỉ lịch sử số đo. `.env` không in secret; API key trong header nếu bật. Dashboard có input API key trong bộ nhớ; không lưu localStorage.

Missing model health=`model_unavailable`, predict503; corrupt400, unsupported415, oversized413, thiếu multipart422. HTTP200/zero detection là kết quả hợp lệ, không bịa mặt. Nút test ảnh offline để kiểm API, không nhận là ESP32. Camera chưa gửi frame hiển thị chưa thấy thiết bị. Backend URL sai hiển thị API DISCONNECTED; sửa URL rồi Áp dụng.

Inference=preprocess/model/NMS; server_processing thêm decode/vẽ nhưng chưa gồm upload/DB/serialization. Client roundtrip đo request đến đọc hết response. ESP32 total từ bắt đầu acquisition/copy JPEG đến đọc xong body. Capture null cho ảnh offline. Upload riêng không đo thì null; không gọi roundtrip-inference là upload một chiều. Client gửi `/metrics/report` request_id/token; retry idempotent; refresh dashboard không tạo row.

## ESP32, IP, Wi-Fi, firewall và demo lớp

Firmware AI Thinker tại esp32_cam/ giữ mapping chân, JPEG, timeout/retry giới hạn, interval; upload/metrics URL cùng backend LAN. config.h local bị ignore. Code Lab2 `D:\PTIT\IoT_UD\lab2\lab2.ino` đã được đối chiếu; không thay đổi file nguồn Lab2.

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -eq 'Wi-Fi'} | Select-Object IPAddress
# Laptop chạy server: tự kiểm health/model.
Invoke-RestMethod http://127.0.0.1:8000/health
# Compile không upload thiết bị:
& 'D:\Arduino\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe' compile --fqbn esp32:esp32:esp32cam --library 'D:\PTIT\IoT_UD\lab2\libraries\ArduinoJson' esp32_cam
```

Compile đã thành công: flash 1,178,201 byte (37%), static RAM 60,816 (18%). Upload qua Arduino IDE, chọn AI Thinker, đúng COM; GPIO0-GND lúc flash, tháo rồi reset; nguồn 5 V ổn định. Serial 115200 kiểm Wi-Fi/IP, HTTP status, timeout. Đây là checklist thủ công vì không có thử nghiệm capture/send trên thiết bị thật trong phiên.

Laptop/camera cùng mạng Wi-Fi 2.4 GHz có thể truy cập nhau; tránh AP client isolation. IP phiên này 192.168.1.5, có thể đổi DHCP. Truy cập `http://<IP_LAPTOP>:8000/health` từ thiết bị khác trước debug model. Nếu Windows Firewall chặn, cho phép python.exe của project/port8000 trên đúng profile Private trong Windows Security; không tắt toàn bộ firewall. Rule thủ công quản trị nếu cần: TCP inbound local port8000, profile Private, remote subnet LAN phù hợp. Không tự mở rule Public. Camera URL cấu hình ở dashboard và ESP32_CAMERA_URL là `http://<IP_ESP32>` (base port80). Firmware đã có `/capture` ở port80 và `/stream` ở port80 redirect sang port81; vì vậy cả hai nút dùng cùng base URL. Không cấu hình base port81 vì `/capture` không chạy ở port81; camera_endpoint chỉ thay path, không tự đổi port.

Trước lớp: load model/metadata, kiểm `/health`; mở dashboard; kiểm 3 trạng thái/2 mặt/ảnh không mặt, ghi nhận lỗi thực tế. ESP32 gửi source=esp32 mới có trạng thái recent online. Đo roundtrip/total/capture trên Wi-Fi thật; nêu riêng inference. Thử tắt backend rồi bật lại để xem timeout/retry; giữ logs, không công bố offline latencies là camera. Chỉ bật Telegram nếu bạn cấu hình và muốn gửi; không gửi ảnh cá nhân tự động trong kiểm thử của phiên này.

## Lỗi thường gặp và giới hạn

- CUDA false: kiểm đúng .venv và wheel +cu128; cài CUDA Toolkit riêng không biến torch+cpu thành GPU wheel.
- OOM: giảm batch 8→4 rồi 2, đóng tiến trình nặng; không thay dataset/test để chữa OOM.
- HEIC decode: cài pillow-heif, giữ log file thất bại; không đổi extension.
- Review accept bị chặn: kiểm tất cả mặt/proposals, mỗi box class ID số 0/1/2; ảnh không chắc loại, không sửa manifest để ép verified.
- Output exists: dùng phiên bản mới; không ghi đè để thêm nhãn vào YAML của run cũ.
- Backend connection refused/timeout: IP/port/process/firewall/AP trước model; thiếu model trả503 có detail.
- Incorrect ít mẫu, ảnh Internet nhỏ, phone/camera khác domain, identity/sessions thiếu. Cần người/phiên đa dạng và camera heldout; nhiều near duplicates không tương đương nhiều dữ liệu độc lập.

Run v2 đầu tiên đã dừng do pretrained=False bỏ checkpoint; log ở reports/baseline_v2/initialization_failure.json. Run đó không publish, không dùng test/demo. Run đã sửa có tensor-equality guard; không resume nhầm checkpoint của run dừng này.

## Camera đã hoàn tất review và evaluation

26/26ảnh human verified,33box người thật. Original human manifest và raw giữ nguyên; 8box ngữ cảnh mở rộng có ledger riêng, không đổi class, chưa human re-verified và không dùng train. Backend hiện dùng V3 + NMS chung giữa class, IoU0.45/conf0.35. NMS loại box trùng cùng mặt nhưng không bảo đảm class đúng/đủ mọi người. AP của baseline và runtime được báo riêng; xem reports/esp32_v1/FINAL_RESULT.md.

```powershell
# Output phải mới; không ghi đè báo cáo cũ.
.\.venv\Scripts\python.exe scripts/evaluate_camera.py --review data/processed/esp32_review_v1 --context-ledger reports/esp32_v1/camera_bbox_context.json --model models/face_mask_best.pt --output results/evaluation/esp32_manual_v2 --device 0 --agnostic-nms
.\.venv\Scripts\python.exe scripts/test_camera_review.py --output results/verification/private/esp32_manual_v2
```

JPEG gốc API:18/33TP,4FP,15FN, đeo sai6/14. Bộ26 được dùng debug NMS, không dùng chọn checkpoint hoặc train; kết quả sau sửa không gọi là test độc lập chưa xem. 91tests pass, firmware compile lại pass. Cách nạp firmware, Serial, dashboard và test treo một tai: [TEST_CAMERA_NOW](TEST_CAMERA_NOW.md). Hardware chưa upload/đo trực tiếp trong phiên.
