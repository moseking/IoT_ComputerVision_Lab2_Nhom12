# API Face Mask IoT

Base URL: API_BASE_URL trong .env. /docs là Swagger.
Header X-API-Key bắt buộc cho endpoint dữ liệu nếu API_KEY không rỗng.
GET /, /static/*, /health, /config, /docs public; /config không trả secrets.
**Các con số trong ví dụ chỉ minh họa schema, không phải kết quả thực nghiệm.**

## GET /

Dashboard HTML.

## GET /health

HTTP200 là process hoạt động; cần model_loaded=true mới inference được.

```json
{"status":"model_unavailable","model_loaded":false,"detail":"Trained checkpoint missing...","timestamp":"..."}
```

## GET /model/info

loaded, model_name, checkpoint filename, classes 0/1/2, confidence_threshold, image_size, device và metadata.
Metadata null khi chưa train. Metrics metadata là validation; test metrics ở results/evaluation/.

## POST /predict

Multipart/form-data:
- file: bắt buộc, JPEG/PNG.
- source: esp32|test_client|dashboard; mặc định test_client.
- capture_ms: tùy chọn, >=0, <=300000.
- include_image: false mặc định; true trả annotated_image_base64 cho CLI.

Giới hạn mặc định file 5MiB, tổng multipart +64KiB, ảnh 12 triệu pixel.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@test.jpg" -F "source=test_client" -F "include_image=true"
```

Schema response:

```json
{
  "success":true,"request_id":"server_generated_id","timestamp":"...",
  "source":"esp32","model_name":"YOLOv8n Face Mask Detection","checkpoint":"face_mask_best.pt",
  "detections":[{"class_id":0,"class_name":"with_mask","confidence":0.94,
    "bbox":{"x1":100,"y1":80,"x2":220,"y2":260}}],
  "detection_count":1,"inference_ms":35.2,"queue_ms":0.3,
  "server_processing_ms":40.5,"image_size_bytes":18304,
  "width":640,"height":480,"capture_ms":20.1,
  "roundtrip_ms":null,"total_ms":null,"end_to_end_ms":null,"upload_ms":null,"download_ms":null,
  "report_token":"unpredictable_token",
  "annotated_image_url":"/results/latest/image?request_id=server_generated_id"
}
```

Box là pixel tọa độ của ảnh decode, không tự xoay EXIF. 0..N detection, mỗi box có class riêng.
inference_ms gồm preprocess/forward/NMS, queue đo riêng.
processing gồm file read/decode/inference/vẽ, trước DB/serialize.
Headers X-Request-ID và Server-Timing để đối chiếu.
RTT/total chưa biết ở server trước response nên giữ null.

## POST /predict/image

Multipart file/source/capture_ms; trả image/jpeg annotated.
Headers: X-Request-ID, X-Report-Token, X-Inference-Ms, X-Server-Processing-Ms,
X-Detection-Count, Server-Timing. Client đo RTT sau khi đọc hết JPEG rồi report.
Để lấy JSON và ảnh khớp chính xác dùng /predict với include_image=true.

## POST /camera/predict

JSON `{"camera_url":"http://192.168.137.28"}`; cần X-API-Key khi được bật. Server tải ảnh `/capture` thật rồi gọi cùng pipeline YOLO, trả schema predict với `source="camera_pull"`, `capture_mode="server_pull"`, `capture_ms=null` và `annotated_image_base64` luôn có. Ảnh và detections thuộc cùng request_id, không cần tải latest/image riêng.

`camera_fetch_ms` là thời gian server HTTP `/capture` + đọc hết ảnh, có cả mạng; không phải thời gian acquisition/copy trên ESP32. RTT/total ở response ban đầu vẫn null; dashboard đo khi đọc hết response rồi gửi `/metrics/report`. Khi mode này total là dashboard request → kết quả YOLO, không khẳng định ESP32 capture/upload latency. Header X-Request-ID/Server-Timing của prediction giữ nguyên; processing_ms không bao gồm bước camera fetch.

Camera URL phải là HTTP(S) với IPv4 literal thuộc RFC1918 (10/8, 172.16/12, 192.168/16), không credentials/path/query/fragment, không localhost/link-local/DNS; không theo redirect. Timeout và giới hạn ảnh được áp dụng. Lỗi camera 502, URL sai 422, model chưa load 503. `/predict` công khai vẫn chỉ nhận source esp32/test_client/dashboard; client không thể gắn upload tùy ý thành camera_pull.

## POST /camera/flash

JSON `{"camera_url":"http://192.168.137.28","enabled":true}`; enabled phải là boolean. Cùng auth và giới hạn URL LAN như `/camera/predict`. Server gửi GET `/flash?on=1` khi bật, `on=0` khi tắt (lệnh đặt trạng thái, retry không toggle). Timeout 5s, JSON camera tối đa 4096 byte, không theo redirect.

Chỉ trả `{"flash":true,"confirmed":true}` khi camera xác nhận đúng boolean đã yêu cầu. Timeout, camera HTTP lỗi, JSON không đúng hoặc trạng thái khác đều trả 502; không trả thành công giả. Không tạo record inference/latency cho lệnh flash.

## POST /metrics/report

Sau khi nhận hết prediction response, client gửi JSON:

```json
{"request_id":"id_from_prediction","report_token":"token_from_prediction",
 "capture_ms":20.1,"roundtrip_ms":158.4,"total_ms":180.2,
 "upload_ms":null,"download_ms":8.4}
```

Trả {"success":true,"measurement":{...}}. ID không tồn tại404, token sai403.
Số đo không khả dụng null; total phải bao gồm capture+RTT (dung sai2ms),
mỗi số0..300000ms, không NaN/Infinity.
Retry đã report trả số đo đầu tiên, không thay đổi/tạo request.
network_overhead_estimated_ms=RTT-processing nếu không âm; residual ước lượng,
không phải mạng một chiều/upload tách biệt. Token không xuất trong latest/history.

## GET /metrics/history?limit=50&offset=0

```json
{"items":[{"id":1,"timestamp":"...","request_id":"...","source":"esp32","model_name":"YOLOv8n Face Mask Detection","checkpoint":"face_mask_best.pt",
 "detection_count":3,"inference_ms":35.2,"server_processing_ms":40.5,
 "roundtrip_ms":158.4,"total_ms":180.2,"capture_ms":20.1,
 "upload_ms":null,"download_ms":8.4,"network_overhead_estimated_ms":117.9,
 "image_size_bytes":18304,"reported":1}]}
```

Newest first; limit1..500, offset>=0. Chưa report: RTT/total=null.

## GET /metrics/summary

total_requests, average_inference_ms, average_roundtrip_ms, average_total_ms;
inference_ms/roundtrip_ms/total_ms/server_processing_ms có
{"samples":n,"average":...,"min":...,"max":...}.
Chỉ tính giá trị không null; chưa có mẫu thì avg/min/max=null.
Summary dùng toàn bộ DB, không giới hạn theo history.

## GET /results/latest; GET /results/latest/image

/latest trả {"result":null} khi chưa inference, sau đó trả kết quả và latency merge từ DB.
/image trả JPEG; request_id không còn là frame hiện tại trả409 để UI thử frame mới.
Latest chỉ một frame RAM, history chỉ số đo. Restart mất latest nhưng giữ SQLite.

## GET /device/status

online khi prediction source=esp32 hoặc camera_pull thành công gần đây: ngưỡng max(15s,3*SEND_INTERVAL_MS).
Trả last_seen/age_seconds/camera_url/basis. Không phải ping hoặc heartbeat WiFi; camera_pull ONLINE không chứng minh ESP32 đang tự push.

## GET /config

API/camera URL/interval và api_key_required; không trả API/WiFi secret.

## HTTP errors

| Status | Lỗi |
|---|---|
| 400 | File rỗng/corrupt, decode/pixel limit |
| 401 | API key thiếu/sai |
| 403 | Token report sai |
| 404 | Request/ảnh chưa có |
| 409 | latest frame đổi |
| 408 | Upload body quá UPLOAD_TIMEOUT_SECONDS |
| 413 | File/body quá lớn |
| 415 | Không JPEG/PNG/octet-stream |
| 422 | Thiếu file hoặc field không hợp lệ |
| 502 | Camera capture/flash không truy cập được hoặc response không hợp lệ |
| 503 | Model chưa train/load lỗi |
| 500 | Inference/encode/DB lỗi, log request_id |

Model load một lần trong lifespan mỗi process, restart khi thay checkpoint.

## GET /telegram/status

Cần X-API-Key nếu API_KEY được bật. Trả enabled/configured/state/verified,
last_error (tên loại lỗi, không chứa token), last_sent, sent_count, skipped_count,
queue_depth, cooldown_seconds, alert_mode và commands_enabled.
ready sau getMe thành công; sent_count/last_sent sau sendPhoto thành công.
Không trả bot token/chat ID hoặc URL Bot API chứa token.

## Alias và lịch sử lỗi

end_to_end_ms bằng total_ms ở predict/latest/report/history/summary.
Client tiếp tục gửi total_ms để tương thích firmware/report cũ.
Bản ghi history thêm status=success|failure, http_status, error_code.
Lỗi xử lý predict được lưu request_failures, tất cả số đo chưa biết là null.
Body/auth/schema bị từ chối trước handler có access log, không tạo phép đo inference.
summary.total_requests vẫn là số request thành công, failed_requests và total_events bổ sung.

## Ảnh gốc cùng frame từ camera

`POST /camera/predict` còn trả `original_image_base64`, `original_image_sha256`, `original_image_content_type`, `original_image_filename`. Đây là bytes JPEG/PNG của chính lần lấy camera tạo ra annotation/request_id này, không chụp lần thứ hai. Nút **Lưu ảnh gốc** dừng Live, giữ kết quả đang xem và download ảnh chưa vẽ box để review/train. Không lưu ảnh đã annotate vào dữ liệu train; các trường ảnh gốc chỉ nằm trong response lượt chụp, không thêm vào latest/global/SQLite.

## Gửi Telegram của lượt chụp thủ công

POST `/camera/predict` nhận `send_telegram` boolean nghiêm ngặt, mặc định false. Khi true, request thành công có `telegram_delivery.requested=true` và state `queued`, `not_queued` hoặc `unavailable`; queued không chứng minh đã gửi. Lượt này gửi chính annotation một lần, không qua lọc lớp/cooldown cảnh báo; vẫn giữ hàng đợi tối đa 2. Live không đặt flag. GET `/telegram/status` thêm `last_sent_request_id`, chỉ cập nhật sau sendPhoto thành công. Không trả secret hoặc ảnh gốc trên endpoint trạng thái.
