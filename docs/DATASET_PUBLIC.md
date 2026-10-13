# Dataset public VOC → YOLO — 2026-10-11

Project: `D:\PTIT\IoT_UD\Lab2_FaceMask`. Đây là dữ liệu thật đã có trên máy; không tải thêm, không sinh ảnh/nhãn, không train.

## Kết quả đọc toàn bộ nguồn

Nguồn `data/raw/facemask_detection/` có `images/` và `annotations/`.
Có 853 ảnh PNG, 853 XML; 853 XML parse được, 4.072 object. Không thiếu cặp ảnh/XML, không XML lỗi cú pháp, không class lạ, không mismatch kích thước. Tên XML không bắt buộc giống ảnh: trường filename trong XML quyết định ghép cặp sau kiểm tra đường dẫn.

| ID | Class | Object nguồn | Box đầu ra hợp lệ |
|---|---|---:|---:|
| 0 | with_mask | 3232 | 2934 |
| 1 | without_mask | 717 | 674 |
| 2 | mask_weared_incorrect | 123 | 117 |
| Tổng | Đủ cả 3 lớp | 4072 | 3725 |

Có 12 box vượt biên trong 11 ảnh. Converter strict bỏ toàn bộ 11 ảnh này: không cắt box và không coi chúng là background.
Có 5 cặp ảnh trùng pixel nhưng nhãn mâu thuẫn (13/737, 146/219, 410/428, 653/775, 656/690; prefix `maksssksksss`, đuôi `.png`). Loại cả 10 ảnh khỏi đầu ra, không tự chọn nhãn nào đúng. Tổng bỏ 21 ảnh, giữ 832 ảnh/832 TXT/3.725 box; không có ảnh nền hợp lệ.
Tên, tọa độ, nguyên nhân và hash cụ thể ở các báo cáo; ảnh/XML gốc giữ nguyên.

## Đầu ra và split

```text
data/processed/facemask_public_yolo/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
├── split_manifest.json
└── README.md
```

Dùng cấu trúc images/labels theo split vốn có của project để tương thích YOLO và validator. Không tạo output phẳng rồi nhân bản ảnh cho đủ thư mục.
`data.yaml` có path tuyệt đối tới output, train/val/test tương đối, nc=3 và mapping cố định 0/1/2 như bảng.

| Split | Ảnh | Box | with_mask | without_mask | mask_weared_incorrect |
|---|---:|---:|---:|---:|---:|
| train | 584 | 2602 | 2049 | 471 | 82 |
| val | 124 | 560 | 442 | 101 | 17 |
| test | 124 | 563 | 443 | 102 | 18 |

Seed 42, mục tiêu 70/15/15; chia theo 820 nhóm nên tỷ lệ thực tế xấp xỉ.
SHA-256 bytes và pixel được audit; ảnh pixel giống nhau hoặc dHash distance ≤4 được group cùng split.
Có `--groups groups.json` để nối thêm nhóm người/phiên. File này map tên ảnh duy nhất sang group ID; ảnh không có metadata chỉ dùng nhóm hash.
Không có danh tính trong public dataset: chưa chứng minh các người khác nhau nằm ở các split khác nhau. dHash có thể bỏ sót crop, góc chụp hoặc nén; kiểm tra thêm trước khi kết luận về leakage.
Chỉ dùng validation để chọn checkpoint/hyperparameter/threshold; test dành cho đánh giá cuối.

Bộ cũ `dataset/` có 843 ảnh/3.970 box do chế độ clip rõ ràng trước đây, giữ nguyên để bảo toàn kết quả cũ. Bộ strict hiện tại có 832 ảnh/3.725 box. Không so sánh hai số liệu như cùng phiên bản.
Default train.py/evaluate.py hiện trỏ `data/processed/facemask_baseline_v1/data.yaml`, snapshot strict public đã train 30 epoch ngày 2026-10-11. Truyền --data rõ ràng cho phiên bản muốn dùng. Xem [BASELINE_RESULT.md](BASELINE_RESULT.md) và [BASELINE_RUNBOOK.md](BASELINE_RUNBOOK.md).

## Lệnh PowerShell

```powershell
Set-Location 'D:\PTIT\IoT_UD\Lab2_FaceMask'

# Audit: exit 1 là các lỗi nguồn đã ghi nhận, không phải XML parser bị hỏng.
.\.venv\Scripts\python.exe scripts/audit_dataset.py --raw data/raw/facemask_detection --output reports/dataset_audit --report-name public_dataset_audit --markdown

# Chỉ lập kế hoạch: không ghi dataset hoặc report.
.\.venv\Scripts\python.exe scripts/convert_voc_to_yolo.py --raw data/raw/facemask_detection --output data/processed/facemask_public_yolo --invalid-policy skip --duplicate-policy exclude --dry-run

# Tạo bộ mới. Đầu ra hiện đã tồn tại nên chạy lại sẽ từ chối ghi đè.
.\.venv\Scripts\python.exe scripts/convert_voc_to_yolo.py --raw data/raw/facemask_detection --output data/processed/facemask_public_yolo --invalid-policy skip --duplicate-policy exclude

# Chỉ thêm --overwrite vào lệnh trên nếu chủ động tái tạo các file đầu ra.
# Không chỉ output vào raw. Converter chặn đường dẫn chồng lấn hoặc symlink thoát thư mục.

.\.venv\Scripts\python.exe scripts/audit_prepared_dataset.py --data data/processed/facemask_public_yolo/data.yaml --output reports/dataset_audit/public_yolo_validation.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dataset_conversion.py -v
```

Default converter có invalid-policy=error, duplicate-policy=error, không clip và không overwrite.
`skip`/ `exclude` là lựa chọn có báo cáo; không sửa raw. TXT được ghi mới, không append; chạy lại cùng seed/nguồn/chính sách giữ ảnh, split và nhãn như nhau (timestamp báo cáo/manifest thay đổi khi ghi thật).
Tọa độ mặc định zero-based continuous; không tự đoán. Nguồn VOC one-based inclusive khác dùng `--coordinates one-based-inclusive`; không đổi convention cho bộ hiện tại chỉ để hợp thức hóa box lỗi.
Hỗ trợ PNG/JPG/JPEG/BMP/WebP, XML UTF-8/UTF-8 BOM; từ chối DTD/entity, XML quá 4 MiB, filename thiếu/thoát thư mục, tên ảnh/label va chạm, box NaN/Inf/diện tích không dương/vượt biên và size không khớp.

## File và bằng chứng

- scripts/dataset_tools.py: audit nguồn, pairing an toàn, hash, raw counts và YOLO lines.
- scripts/audit_dataset.py: tái sử dụng audit, thêm Markdown và report-name.
- scripts/prepare_dataset.py: tái sử dụng group/split, dry-run/no-overwrite, staged output và provenance. --repair-boxes chỉ còn là tùy chọn legacy rõ ràng.
- scripts/convert_voc_to_yolo.py: entry point strict; không có tùy chọn repair.
- tests/test_dataset_conversion.py: 25 tests XML/box/mapping/pairing/overwrite/dry-run/duplicates/group.
- reports/dataset_audit/public_dataset_audit.{json,md}: 853 ảnh/853 XML/4.072 object và danh sách lỗi.
- reports/dataset_audit/public_dataset_conversion.{json,md}: 832 ảnh/832 TXT, các split, toàn bộ ảnh bỏ và lý do.
- reports/dataset_audit/public_yolo_validation.json: kiểm tra lại ảnh/TXT thực tế và trùng qua split.
- reports/dataset_audit/conversion-unit-tests.log, regression-tests.log: log kiểm thử.
- reports/dataset_audit/source_integrity_before.json, source_integrity_after.{json,md}: đối chiếu SHA-256 nguồn.
- reports/dataset_audit/source_before_2026-10-11.zip: backup ba script dataset trước chỉnh sửa.

Project không có Git repository nên không commit/push. Python 3.12 trong .venv và Pillow/PyYAML sẵn có; không thêm dependency.
Nguồn không có background, lớp incorrect chỉ 117 box ở đầu ra và còn domain shift từ Internet sang ESP32. Cần bổ sung ảnh camera đã review, background thật, báo metrics từng lớp và đánh giá riêng trên camera. Baseline v1 đã đánh giá public val/test; chưa có kết quả camera. Xem BASELINE_RESULT.md.
Quy trình ảnh tự chụp nằm ở [CUSTOM_DATASET_WORKFLOW.md](CUSTOM_DATASET_WORKFLOW.md).

## Kết quả kiểm chứng thực tế

25 unit tests converter và toàn bộ 55 tests đều qua (regression: 27.280 giây). Validator đọc lại dataset mới có 0 lỗi/0 cảnh báo. SHA-256 của 3.407 file bảo vệ trước/sau giống nhau, gồm 1.706 ảnh/XML raw, 1.688 file dataset cũ và code server/web/firmware. 832 ảnh output có checksum khớp nguồn. Dry-run trên dataset thật giữ nguyên inventory file raw/output/report và không tạo report directory; split plan khớp output.

Kết quả này xác nhận cấu trúc/nhãn/split theo các phép kiểm đã nêu, chưa xác nhận chất lượng model hay nhận diện camera.
