# File source và chức năng

| File | Chức năng |
|---|---|
| README.md | Hướng dẫn tiếng Việt, lệnh chạy, dataset, train, IP, camera, demo, screenshot |
| requirements.txt | Dependency trực tiếp của project |
| .env.example | Mẫu backend/client config, không secret thật |
| .env | Cấu hình local; Git bỏ qua |
| .gitignore | Bỏ qua secrets, dataset lớn, database, weights, môi trường |
| Dockerfile / .dockerignore | Backend/web cloud CPU, weights và SQLite mount riêng |
| scripts/common.py | Root, class mapping, device auto, metrics serialization |
| scripts/dataset_tools.py | VOC parser/audit: image/XML/box/hash/duplicates/distribution |
| scripts/audit_dataset.py | Audit read-only, JSON/TXT và Markdown tùy chọn |
| scripts/prepare_dataset.py | YOLO strict, split theo nhóm, manifest; clip chỉ khi chọn chế độ legacy rõ ràng |
| scripts/train.py | YOLOv8n scratch/optional fine-tuning, best/last/metadata/plots |
| scripts/evaluate.py | Test precision/recall/mAP từng lớp và plot export |
| scripts/test_inference.py | Multipart HTTP client, latency thật, lưu annotated image |
| scripts/test_multi_person.py | Ảnh test thật 1/2/3/nhiều mặt, giữ dự đoán thực tế |
| scripts/configure_esp32.py | Sinh config.h từ .env, không in secrets |
| server/config.py | .env, limits, model/database/URL paths |
| server/inference.py | Load một lần, predict BGR/PIL/bytes, vẽ box mỗi mặt |
| server/main.py | FastAPI, dashboard routes, upload guard, auth, latest result |
| server/latency_store.py | SQLite WAL/history/summary/report token và idempotency |
| web/index.html | Dashboard và offline test panel |
| web/style.css | Bố cục responsive |
| web/app.js | Poll kết quả/status/metrics, ảnh/charts/CSV |
| esp32_cam/esp32_cam.ino | Camera pin AI Thinker, WiFi, multipart, latency, web stream |
| esp32_cam/config.h.example | Mẫu WiFi/API/TLS/interval |
| esp32_cam/config.h | Placeholder cần chỉnh; Git bỏ qua |
| tests/test_project.py | Syntax/YAML/YOLO/leakage/API/checkpoint thật |
| docs/API.md | Endpoint/request/response/error/metric definitions |
| docs/ARCHITECTURE.md | Kiến trúc/clock/leakage/cloud |
| docs/DEMO.md | Flow demo 1/2/3/nhiều người, screenshot, troubleshooting |

__init__.py trong server/scripts/tests đánh dấu Python package.
Raw dataset do bạn cung cấp, không được tạo giả hay sửa.
dataset/data.yaml, split_manifest, images/labels là output prepare thật.
models/best/last/metadata được train đầy đủ sinh ra, chưa có model demo được publish.
results/training/verification_* chứa checkpoint thật một epoch, chỉ dùng test kỹ thuật.
results/verification/evaluation/ có số đo test model một epoch; metrics0 phản ánh chất lượng chưa đạt.
Report/log kiểm tra là bằng chứng kỹ thuật; không dùng chúng để thay kết quả live ESP32/cloud.

## Bổ sung ngày 2026-10-09

| File | Chức năng |
|---|---|
| scripts/audit_prepared_dataset.py | Đọc lại YOLO thực tế, box bounds/hash/leakage/count; train gọi trước khi chạy |
| scripts/checkpoint_tools.py | Chặn smoke publish, từ chối ghi đè ngầm, archive checkpoint cũ |
| scripts/capture_dataset.py | Thu JPEG/PNG ESP32, manifest SHA256/dimensions, không label/train |
| server/camera_client.py | HTTP capture sync/async có timeout/giới hạn/validate |
| server/telegram_service.py | Bot API HTTPS worker/queue/cooldown/commands/chat allowlist |
| tests/test_extensions.py | Mock Telegram/camera/model output contract; upload timeout/store/publish guards |
| docs/TELEGRAM.md | Setup bot mới, modes/cooldown/lệnh và phạm vi kiểm chứng |
| docs/DATA_COLLECTION.md | Thu/gán nhãn/group/prepare ảnh thật |
| docs/TROUBLESHOOTING.md | Connection refused, image/stream/TLS/nguồn/WiFi |
| docs/CLOUD.md | Cấu hình deploy sau local, HTTPS/API key/volumes |
| docs/AUDIT_2026-10-09.md | Audit nguồn/dataset/checkpoint thực tế |
| docs/VERIFICATION.md | Kết quả test/compile/browser và giới hạn bàn giao |

Source sửa: server/main.py/config.py/latency_store.py; scripts/train.py/evaluate.py/test_inference.py/test_multi_person.py;
web/index.html/app.js/style.css; esp32_cam/esp32_cam.ino/config.h.example;
README và tài liệu API/ARCHITECTURE/DEMO/FILES. .env/config.h chỉ bổ sung cấu hình còn thiếu,
giữ nguyên giá trị trước đó. Không chỉnh lab2/lab2.ino, raw dataset hoặc weights đã tồn tại.

## Bổ sung dataset ngày 2026-10-11

| File | Thay đổi |
|---|---|
| scripts/dataset_tools.py | Audit đủ raw object/classes, XML/filename/dimensions/bounds an toàn, nguồn SHA-256, strict YOLO lines |
| scripts/audit_dataset.py | Thêm report-name và Markdown; audit vẫn read-only |
| scripts/prepare_dataset.py | Giữ logic group/split; thêm plan/dry-run, no-overwrite, staging/provenance, skip toàn ảnh |
| scripts/convert_voc_to_yolo.py | Entry point strict tái sử dụng prepare; không clip, default error, skip/exclude rõ ràng |
| tests/test_dataset_conversion.py | 25 tests converter và raw/output/dry-run/group guards; không sửa test cũ |
| docs/DATASET_PUBLIC.md | Số liệu thực, lệnh PowerShell, phiên bản strict, giới hạn dataset |
| docs/CUSTOM_DATASET_WORKFLOW.md | Gợi ý bbox tự động, review final, custom audit/convert và thiết kế merge |
| reports/dataset_audit/ | Audit/conversion/validation JSON/Markdown, logs/tests và đối chiếu SHA-256 |
| data/processed/facemask_public_yolo/ | 832 ảnh/832 TXT/3725 box, data.yaml nc=3, split_manifest provenance |
| README.md, docs/DATA_COLLECTION.md | Trỏ bộ strict và hướng dẫn custom đã review |

Không thêm dependency hoặc chỉnh firmware/backend/dashboard/models. Raw và dataset cũ giữ nguyên.

## Baseline/review mới 2026-10-11

| File | Chức năng |
|---|---|
| scripts/custom_dataset.py | Actual HEIF/EXIF conversion, hashes/blur/dHash, YuNet drafts/contact sheets; no auto verify |
| scripts/review_annotations.py | Local nonce-protected per-face review/provenance; no training |
| web/annotation_review.html | Canvas box/class editor, accept/draft/exclude |
| scripts/build_baseline_dataset.py | Frozen public splits + hash-checked human-verified custom train, versioned manifest/validator |
| scripts/train.py | Scratch/COCO/checkpoint fine-tune, interrupted-only resume, new run/checkpoints |
| scripts/evaluate.py | Val/test, perclass/AP/confusion, FP/FN/GT/pred examples, perimage synchronized latency |
| tests/test_custom_dataset.py | HEIF codec/orientation/bbox/class/review gates/multiface/FPFN matching |
| docs/BASELINE_RUNBOOK.md | Environment, dataset/review/train/eval/server/ESP32/demo instructions |
| docs/BASELINE_RESULT.md | Actual final counts, metrics, artifacts, verification and limitations |

Reports nhỏ ở reports/baseline_v1; private images trong processed/ và results/verification/private. models/ có checkpoint đã train; chưa xác minh camera thật. Không đọc snapshot verification một epoch ngày 2026-10-08 như trạng thái model hiện tại.

## Bổ sung camera V8 và Telegram

| File | Vai trò |
|---|---|
| scripts/prepare_new_review.py | Import file mới thành nháp, không đổi review đã xác minh |
| tests/test_new_review.py | Kiểm import mới, nguồn/hash và dataset đã đóng băng |
| tests/test_context_boxes.py | Kiểm padding class 2 và provenance của bản export train |
| scripts/select_model.py | Chọn checkpoint đã lưu; restart backend để áp dụng |
| reports/camera_v8/ | Snapshot review, QC, bbox context, training/evaluation, kết quả thực tế |
| docs/TELEGRAM.md | Bot/token cũ đã kết nối, cảnh báo và lệnh camera |
