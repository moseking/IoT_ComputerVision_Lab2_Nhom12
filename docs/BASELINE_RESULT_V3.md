# Kết quả Lab 3 — checkpoint baseline v3

Hoàn thiện ngày 2026-10-11, trực tiếp tại `D:\PTIT\IoT_UD\Lab2_FaceMask`. Backend đang dùng v3. Dữ liệu gốc, dataset/run cũ và nhãn bạn đã sửa được giữ nguyên. Đây là baseline đã đánh giá offline; ESP32/camera chưa được kiểm chứng trên thiết bị thật.

## Dữ liệu đã chốt

| Nguồn | Ảnh nguồn | Ảnh giữ để train/eval | Box | Trạng thái |
|---|---:|---:|---:|---|
| Public | 853 PNG + 853 XML | 832 | 3725 | Audit strict, 0 lỗi/0 cảnh báo |
| Phone | 280 (235 HEIC + 45 JPEG) | 266 | 271 | Human verified, chỉ train |
| Multi-person | 1006 (5 HEIC + 1001 JPEG) | 0 | Draft | Chuyển 1006/1006; chưa xác minh |
| ESP32 | 0 | 0 | 0 | Pipeline riêng đã sẵn, chưa có ảnh thật |

| Folder phone | Nguồn | Hợp lệ xuất train |
|---|---:|---:|
| with_mask | 91 | 77 |
| without_mask | 59 | 59 |
| wore_incorrect | 130 | 130 |

276 ảnh phone có provenance human review; 4 ảnh bạn loại. Trong 276 ảnh, 10 ảnh có mặt người nền nhìn thấy nhưng chưa có box tương ứng, được giữ ngoài train sau kiểm tra trực quan. Các ảnh này đều ở folder with_mask; không sửa/ghi đè nhãn của bạn. Danh sách: `reports/baseline_v3/review_final.json`, manifest exclusions. 266 ảnh giữ có 271 box: 77 with_mask, 64 without_mask, 130 incorrect (có mặt nền khác class).

Public loại 11 ảnh chứa 12 box vượt biên và 10 ảnh thuộc 5 cặp pixel trùng với nhãn mâu thuẫn, tổng 21 ảnh. Raw 4072 box: 3232/717/123; strict 3725 box: 2934/674/117. Bộ legacy `dataset/` 843 ảnh được giữ nguyên. Kiểm tra convention VOC zero-based continuous, đối chiếu TXT và 24 ảnh mẫu GT trước train; xem `reports/baseline_v1/coordinate_verification.json`.

YAML cuối: `data/processed/facemask_baseline_v3/data.yaml`; mapping 0=with_mask, 1=without_mask, 2=mask_weared_incorrect. Tổng 1098 ảnh, 3996 box (3011/738/247).

| Split | Ảnh | Box | with_mask | without_mask | incorrect |
|---|---:|---:|---:|---:|---:|
| train | 850 | 2873 | 2126 | 535 | 212 |
| val | 124 | 560 | 442 | 101 | 17 |
| test | 124 | 563 | 443 | 102 | 18 |

Seed 42; public 584/124/124 giữ nguyên ở mọi version. Phone được đặt trong một nhóm chụp bảo thủ, chỉ vào train; không chia gần trùng qua split. Kiểm SHA/pixel, dHash<=4, raw/processed hash và provenance khi build. Phone phát hiện 99 cặp gần trùng trong 280 ảnh; không có exact duplicate. Public thiếu person ID và phone có mặt nền, chưa chứng minh tách danh tính tuyệt đối. Không có phone test độc lập; tỷ lệ split tổng khác 70/15/15 là chủ ý giữ holdout public.

## Model và training thật

GPU thật RTX 3050 Laptop 6 GB, CUDA 0; Python 3.12.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.174. Tensor CUDA và training thực tế đã chạy. Pin stack ở requirements-cuda-tested.txt. Giảng viên cho phép fine-tune pretrained.

| Run | Dataset | Epoch thực chạy | Khởi tạo |
|---|---|---:|---|
| v1 | Public 832 ảnh | 30 | Official COCO yolov8n.pt |
| v2 corrected | Public + 167 phone tại thời điểm chốt | 20 | v1 best, 355 tensor match |
| v3 | Public + 266 phone sau review hoàn tất | 9/10 | v2 best, 355 tensor match |

V3 batch 8, image 512, workers 0, seed 42, AdamW initial parameter-group LR 0.00015, lrf 0.2, patience 5, amp False. Early stopping sau 5 epoch không tăng validation fitness; best ở epoch 4. Run v2/v3 có lý do là thêm ảnh human reviewed ở hai lần chốt; không chạy dò test. LR trong CSV gồm warmup/schedule; metadata đã sửa trường từng ghi AdamW constructor default 0.001 bằng LR thật từ log, giữ backup và correction report.

Checkpoint gốc: `D:\PTIT\IoT_UD\Lab2_FaceMask\results\training\baseline_v3_2026-10-11T064852.371357_0000\weights\best.pt`; last.pt cùng thư mục. Backend: `models/face_mask_best.pt`, `face_mask_last.pt`, `model_metadata.json`; SHA best `d66dec3404992fe0c227b099ea47069c117e41739ac7e296219a857eeaf95764`. Model publish cũ được archive trước thay. Loss/metrics từng epoch trong results.csv, curves/args/metrics.json trong run; train log reports/baseline_v3/training.log. Chọn epoch bằng val fitness; giữa các candidate chọn standardized validation mAP50-95 v3 0.5672 > v2 0.5558 > v1 0.5480, không dùng test. Val trong trainer 0.5705 có thiết lập evaluator nội bộ; các bảng dưới dùng cùng evaluator offline NMS IoU 0.45.

Lượt v2 đầu tiên bị dừng sau 4 epoch do pretrained=False làm Ultralytics bỏ init weights. Không publish/evaluate test/dùng demo; giữ run và `reports/baseline_v2/initialization_failure.json`. Run corrected khắc phục và kiểm tensor equality trước optimizer.

## Evaluation khóa checkpoint

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validation | 0.8949 | 0.7421 | 0.8316 | 0.5672 |
| Test | 0.8852 | 0.7614 | 0.8688 | 0.5555 |

Test per class:

| Class | P | R | mAP50 | mAP50-95 | GT | Pred | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with_mask | 0.9625 | 0.8681 | 0.9496 | 0.6506 | 443 | 423 | 30 | 50 |
| without_mask | 0.7926 | 0.7493 | 0.8406 | 0.5266 | 102 | 113 | 29 | 18 |
| mask_weared_incorrect | 0.9005 | 0.6667 | 0.8162 | 0.4893 | 18 | 16 | 3 | 5 |

P/R là operating point Ultralytics; AP dùng min conf 0.001. Cột Pred/FP/FN ở conf vận hành 0.35, NMS IoU 0.45, matching class-aware IoU>=0.5. Không đồng nhất FP/FN với P/R tối ưu của framework. Incorrect chỉ có 17 GT val/18 GT test: recall test 0.6667, còn bỏ sót, chưa khẳng định generalization với camera.

Inference wall time test: mean 18.09 ms, median 12.70 ms, p95 62.38 ms trên 124 ảnh, warmup loại, CUDA synchronize, gồm preprocess/forward/NMS, không gồm decode. Đây là offline trên laptop, không phải latency ESP32/Wi-Fi.

JSON/CSV/Markdown từng split: `results/evaluation/baseline_v3/{validation,test}/`; predictions.json chứa GT/pred và FP/FN mỗi ảnh. 16 ảnh lỗi mỗi split trong error_examples (GT xanh, prediction đỏ), confusion/PR plots trong subfolder framework. Report gộp: reports/baseline_v3/final_summary.json và final_metrics.csv.

## Kiểm thử hệ thống

- 76/76 unit/regression tests pass, 8.138 s, dùng published v3; log reports/baseline_v3/regression-tests.log. Bao gồm VOC conversion, duplicate/leakage/class invalid, real HEIF/EXIF, review provenance, invalid boxes, camera source boundary, backend/latency/checkpoints. Không sửa/xóa test để ép pass.
- pip check: No broken requirements found. Node --check dashboard JS và review inline JS pass; giao diện vanilla JS không có build bundler.
- HTTP thật: health model loaded; corrupt400, unsupported415, oversized413, missing multipart422; missing model503; unreachable backend ConnectTimeout; flat synthetic JPEG200/0 detection; gửi metrics cùng request hai lần chỉ 1 row. Synthetic dùng kiểm transport/error, không train/không gán nhãn camera.
- Dashboard thật: API CONNECTED, YOLO LOADED, box/confidence/image/history/latency hiển thị; console error=[]; từng hiển thị 14 detections từ HTTP. Ảnh upload 852 có GT2 nhưng v3 chỉ phát hiện1, ghi nguyên kết quả. Frontend disconnected/reconnected test ở baseline_v1, source UI giữ nguyên. Reports baseline_v3/frontend_checks.json.
- Cold first request mới start server: inference 2380.6 ms, HTTP 2407.3 ms; không giấu vào số warm latency. DB test riêng results/verification/private/baseline_v3/latency.db; ảnh latest chỉ RAM, private GT/pred samples local; Telegram tắt.
- Firmware compile thật AI Thinker esp32:esp32:esp32cam, ESP32 core3.3.12, ArduinoJson7.4.3: flash1,178,201 B (37%), static RAM60,816 B (18%). Log baseline_v1/firmware-compile.log. Firmware giữ nguyên, không upload/hardware test.
- Hash bảo toàn: 3407 file gốc public/legacy/server/web/firmware không đổi; 280 phone raw không đổi; legacy best hash đúng metadata; public->v3 byte copies không khác. Source scripts/docs mới/chỉnh sửa là chủ ý; run/model mới giữ local; không commit/push/upload Git.

| Smoke public qua HTTP | GT | Pred | Inference ms | Roundtrip ms |
|---|---:|---:|---:|---:|
| one (public_maksssksksss132.png) | 1 | 1 | 92.8 | 212.6 |
| two (public_maksssksksss852.png) | 2 | 1 | 84.8 | 116.7 |
| three (public_maksssksksss522.png) | 3 | 2 | 44.9 | 75.9 |
| many (public_maksssksksss305.png) | 14 | 14 | 42.6 | 94.4 |

Đếm đủ 14 không có nghĩa 14 box/class đều đúng: có duplicate khác class ở cùng mặt và nhầm class. GT2->1, GT3->2 cho thấy còn bỏ sót. Không chỉnh threshold theo các ảnh test này.

## Chạy demo và phần cần làm trên thiết bị

```powershell
Set-Location 'D:\PTIT\IoT_UD\Lab2_FaceMask'
# Server đã chạy trên port8000 trong phiên này; không chạy thêm nếu còn hoạt động.
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
Invoke-RestMethod http://127.0.0.1:8000/health
```

Dashboard http://127.0.0.1:8000; .env MODEL_PATH=models/face_mask_best.pt, IMAGE_SIZE=512. Laptop LAN IP quan sát 192.168.1.5 (kiểm lại DHCP), firmware dùng http://<IP_LAPTOP>:8000/predict và /metrics/report. Secret chỉ config.h/.env local. Chi tiết install/train/evaluate/resume/compile/camera/Wi-Fi/firewall trong BASELINE_RUNBOOK.md.

1. Flash firmware lên AI Thinker, chọn đúng COM, nguồn5V; GPIO0-GND khi upload, tháo rồi reset. Serial115200 xem IP/status/errors. Cấu hình backend LAN IP thật, kiểm cùng Wi-Fi và AP isolation/firewall Private. Camera URL base http://<IP_ESP32> port80; /stream redirect81 sẵn, /capture80.
2. Chụp20 ảnh ESP32 bằng capture_dataset.py -> data/raw/esp32_eval_v1. custom_dataset.py --source-kind esp32 tạo JPEG/contact/quality/proposals; review_annotations.py --port8767 để gán box/class. Builder chặn esp32 vào baseline để giữ camera holdout. Chưa có ảnh thật, chưa đo capture/roundtrip/total Wi-Fi, không tuyên bố đã demo hardware.
3. Nếu muốn dùng 10 ảnh phone còn lại, review thêm box cho mọi mặt nền; giữ các manifest/dataset đã train, build bản v4 mới. Không chỉnh file verified thủ công để ép export.
4. Multi 1006 ảnh đã decode/contact sheets/YuNet proposals, 0 human verified; mở http://127.0.0.1:8766, chọn class riêng từng mặt. Proposal counts: 51 ảnh1 mặt, 794 ảnh2, 136 ảnh3, 25 ảnh4. Sau khi verified và group/overlap audit đủ, tạo dataset/run mới; không tự train draft.

Giới hạn: public nhiều ảnh nhỏ/mơ hồ; thiếu identity/session metadata; phone một nhóm không đo generalization độc lập; ảnh camera khác domain; incorrect holdout ít; vẫn FP/FN và duplicate khác class. Baseline chạy được offline/API, trạng thái demo_eligible=False cho đến khi có xác minh camera thật.

Automatic approval review từng từ chối bước tự động xác minh117 nhãn chưa được người review, vì có nguy cơ đưa nhãn chưa xác minh vào train. Bước đó không được chạy; lần này dùng nhãn bạn đã xác minh và các guard, không bỏ qua rejection.
