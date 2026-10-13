# Workflow custom hiện có

Hướng dẫn đầy đủ và lệnh Windows: [BASELINE_RUNBOOK.md](BASELINE_RUNBOOK.md).

`data/raw/custom_capture/single_person/{with_mask,without_mask,wore_incorrect}` có 91/59/130 ảnh. Canonical IDs 0/1/2 cố định; `wore_incorrect`→`mask_weared_incorrect`. Đọc HEIF thật, EXIF transpose và tạo JPEG riêng tại `data/processed/single_person_review_v1`; 280/280 decode thành công. Raw không sửa. Manifest lưu source/processed hash, dimensions, orientation, blur, dHash, intended class, draft boxes, review state.

YuNet chỉ đề xuất box; folder chỉ class dự kiến. Nhiều ảnh có người nền hoặc box quá nhỏ. Review tại `http://127.0.0.1:8765`: nhập reviewer, kiểm box bao mặt/mũi/miệng và gán mỗi mặt đúng class. App chặn bỏ proposal chưa đủ box; ảnh ambiguous loại. Không tạo accepted bằng script từ confidence. Builder còn kiểm human_reviewed/reviewer/time/hash; bản verified cũ thiếu mặt nền bị loại.

Ảnh verified đóng snapshot bằng `build_baseline_dataset.py`; thêm train cùng nhóm chụp, giữ public val/test. Không tuyên bố phone test tách người vì chưa có identity/session đầy đủ. DHash heuristic phát hiện 99 cặp gần trùng; raw không bị xóa. Review thêm không cập nhật dataset frozen: tạo version mới.

Multi-person có snapshot riêng, class unknown/null từng proposal nếu folder không mang class. Sửa class từng box và xem các trạng thái trộn. Không đưa draft vào train. Camera giữ riêng `data/raw/esp32_eval_v1`, ưu tiên đánh giá domain; chưa có captures thật. Không dùng ảnh phone làm bằng chứng ESP32. Contact sheets và review dùng cùng pipeline, port khác để tránh thay manifest đang review.

Alternative CVAT local có thể dùng khi cần collaboration/session annotation, nhưng project hiện đã có editor bbox/class nhẹ, không cần cài thêm dependency nặng. Export phải giữ provenance/mapping và qua audit; labels không thể xác minh chỉ bằng syntax validator.

Chọn box bằng danh sách hoặc bấm vào mặt trên canvas; viền trắng đánh dấu box đang chọn. Dropdown class đồng bộ với box đó và giữ lựa chọn sau redraw. Có thể gán cùng một class cho nhiều box có sẵn (ví dụ cả hai người without_mask), không cần vẽ lại. Chỉ bấm Xác minh khi đã kiểm từng mặt. Sau cập nhật giao diện, lưu nháp trước khi reload nếu đang có sửa chưa lưu.
