# Quy trình thu ảnh và annotation

Cập nhật 2026-10-11: dataset public đã được chuẩn hóa độc lập ở
data/processed/facemask_public_yolo/. Quy trình ảnh tự chụp theo folder
single_person/{with_mask,without_mask,mask_weared_incorrect} và multi_person,
gợi ý bbox tự động rồi review, export staging và thiết kế merge nằm ở
[CUSTOM_DATASET_WORKFLOW.md](CUSTOM_DATASET_WORKFLOW.md).
Folder class không thay cho annotation. Chưa có custom folder vẫn xử lý public được.

Các lệnh dưới đây dành cho công cụ capture hiện có (lưu images/ + manifest);
khi dùng cấu trúc folder bạn tự chụp, cần export reviewed VOC vào staging như guide mới
trước khi audit/convert. Không tự train sau khi capture.

# Thu ảnh ESP32 độc lập với inference

Mở http://IP_CAMERA/capture để xác nhận camera trả ảnh thật.
Có thể đặt INFERENCE_ENABLED=false trong config.h rồi nạp lại để chỉ thu dữ liệu;
trang /, /capture, /stream, /flash vẫn hoạt động.

```powershell
cd D:\PTIT\IoT_UD\Lab2_FaceMask
.\.venv\Scripts\python.exe scripts/capture_dataset.py --camera http://IP_CAMERA --count 100 --interval 2
```

Có thể bỏ --camera nếu ESP32_CAMERA_URL đã có trong .env.
--timeout mặc định 10 giây; --output mặc định data/raw/custom_capture.
Ảnh thành công được lưu bằng tên timestamp UTC + UUID, không ghi đè ảnh cũ.
manifest.jsonl lưu timestamp, camera_source, SHA-256 của bytes ảnh gốc,
width, height, image_size_bytes, label_status=unlabeled, included_in_training=false.
Request lỗi/timeout không tạo ảnh hoặc manifest thành công giả; script trả exit 1 nếu có lỗi.
JPEG/PNG được decode thật và kiểm tra giới hạn byte/pixel trước khi lưu.
Không cần backend chạy, không import YOLO và không tự train.

## Gán nhãn rồi đưa vào dữ liệu

1. Thu nhiều điều kiện sáng/khoảng cách, đủ ba lớp, có một và nhiều khuôn mặt.
   Thu thêm ảnh không có người để mô tả background âm.
2. Dùng công cụ gán nhãn bbox hỗ trợ Pascal VOC XML. Mỗi mặt có đúng nhãn:
   with_mask, without_mask, mask_weared_incorrect.
   Đeo sai: dưới mũi/miệng hoặc sai vị trí theo tiêu chí ghi trong báo cáo.
3. Lưu XML cùng basename vào data/raw/custom_capture/annotations.
   XML filename và size phải khớp ảnh thật. Ảnh không có mặt có XML hợp lệ không có object.
4. Nhóm ảnh theo người/phiên chụp để tránh frame cùng người lọt vào các tập khác nhau.
   groups.json có dạng {"image.jpg":"person_session_01"}.
5. Audit dữ liệu mới riêng:
   
   ```powershell
   .\.venv\Scripts\python.exe scripts/audit_dataset.py --raw data/raw/custom_capture --output results/custom_capture_audit
   ```

6. Chuẩn bị tập mới ở đường dẫn riêng bằng --raw, --output và --groups của prepare_dataset.py.
   Không chạy --overwrite trên dataset đang dùng chỉ để thêm ảnh.
   Ví dụ sau khi đã có đủ nhãn:
   
   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_dataset.py --raw data/raw/custom_capture --output dataset_custom --groups data/raw/custom_capture/groups.json
   .\.venv\Scripts\python.exe scripts/audit_prepared_dataset.py --data dataset_custom/data.yaml
   .\.venv\Scripts\python.exe scripts/train.py --data dataset_custom/data.yaml --pretrained --epochs 100 --imgsz 640 --batch 8 --no-publish
   ```

Muốn kết hợp Kaggle và ảnh camera: tạo nguồn VOC hợp nhất ở thư mục riêng, đổi basename
nếu trùng và cập nhật filename XML tương ứng, giữ groups theo người/phiên; audit rồi split lại.
Giữ một test camera độc lập để đánh giá domain shift. Không dùng test để chọn ngưỡng/checkpoint.
Script không tạo annotation, pseudo-label hoặc tự đưa capture vào training.
