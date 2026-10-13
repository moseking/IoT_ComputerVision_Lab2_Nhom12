# Triển khai Lab 2 từ repository đầy đủ

Repository đầy đủ chứa firmware, dashboard, backend, công cụ train, checkpoint
local và ONNX. Dockerfile ở root chọn riêng API ONNX CPU cho Render; dashboard,
camera LAN, SQLite và Telegram hoạt động qua backend local.

Chuẩn bị mã nguồn để push: `python scripts/prepare_repository.py`. Source nằm
trong `.runtime/deployment/full_lab2/source`, ZIP ở `reports/full_repository/`.
Ảnh dataset, `.env`, cấu hình Wi-Fi và môi trường Python không nằm trong gói.

Render Web Service dùng Dockerfile `./Dockerfile`, context `.`, branch `main`,
gói Free, health check `/health`. Đặt `CLOUD_API_KEY` và `CLOUD_MODEL_SHA256`
trùng file `.runtime/cloud/render_secrets.env`. Sau deploy, cấu hình URL qua
`scripts/configure_cloud.py --url https://TEN-SERVICE.onrender.com`, khởi động
lại backend local và chọn Cloud Render trên dashboard.

Dockerfile chỉ cài cloud/render/requirements.txt, không cài Torch. Model được
COPY từ models/cloud_v7/face_mask_v7.onnx và kiểm tra SHA-256 khi khởi động.
Dockerfile.local giữ cấu hình container backend local riêng.

Đo inference tại Render, RTT của laptop gửi ảnh/nhận hết response, end-to-end
từ dashboard yêu cầu chụp/test đến khi nhận kết quả. Kiểm tra HTTPS thật bằng
`scripts/verify_cloud.py --image "D:\images (3).jpg"` trước khi xác nhận deploy.

Render Free có thể ngủ sau thời gian không có request. Mở `/health` và chờ
model_loaded=true trước demo: https://render.com/docs/free.

Gói tối giản chỉ gồm API cloud vẫn có thể tạo bằng scripts/prepare_cloud.py.
