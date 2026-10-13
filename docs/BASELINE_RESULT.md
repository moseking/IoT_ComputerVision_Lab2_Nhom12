# Kết quả Lab 3 — hoàn tất review multi-person và fine-tune v6

Ngày 2026-10-11, làm trực tiếp tại `D:\PTIT\IoT_UD\Lab2_FaceMask`. Giữ v3 trên backend vì validation mAP50-95 của v6 không cao hơn; run v6 đã hoàn tất và giữ nguyên để đối chiếu. Model được chọn bằng validation, không dùng test. Target theo lựa chọn của bạn: chỉ khuôn mặt người thật; mặt in trên poster là background trong các ảnh custom đã audit.

## Dataset mới

| Nguồn | Ảnh nguồn | Verified | Dùng | Box dùng | Loại/chờ |
|---|---:|---:|---:|---:|---:|
| Public | 853 | Source annotation audit | 832 | 3725 | 21 |
| Phone | 280 | 276 | 266 | 271 | 14 |
| Multi | 1006 | 846 | 683 | 1370 | 323 |
| ESP32 | 26 | Đang review | 0 train | Chưa evaluate | Giữ độc lập |

Multi: 158 ảnh bạn loại, 2 còn nháp, 158 bản trùng loại, 3 ảnh chuyển trạng thái khó xác định và 2 ảnh cùng pixel có nhãn trái/phải xung đột (review index342/343), không tự sửa class. Phone: 4 bạn loại, 10 ảnh thiếu box người nền giữ ngoài train. Chỉ human verified có reviewer/time/hash hợp lệ được xuất.

Sau kiểm tra trực quan 80 ảnh unique có trên2 box, bỏ83 box poster ở76 ảnh; giữ box người thật phía sau trong cảnh lớp học. Ledger `reports/baseline_v5/non_person_box_exclusions.json` ghi ID, hash snapshot, original boxes, removed indices và evidence. Nhãn gốc không sửa, phần box thật vẫn dùng class/coordinate bạn xác minh. Lọc áp dụng đúng snapshot, từ chối nếu review thay đổi, không áp dụng theo kích thước box chung. `v4` là candidate audit có poster, không train/publish; `v5` dừng sau 7 epoch để audit nhãn, giữ checkpoint/log; `v6` loại thêm 3 ảnh chuyển trạng thái khó xác định (468–470), không tự đổi class. Ảnh255 đã có nhãn đúng [1,0] và được giữ. Quy ước: che kín mũi và miệng là đeo đúng; hở mũi/miệng hoặc kéo xuống cằm là đeo sai.

YAML: `data/processed/facemask_baseline_v6/data.yaml`; 1781 ảnh/5366 box; class0/1/2=3378/1054/934. Mapping0=with_mask,1=without_mask,2=mask_weared_incorrect (folder wore_incorrect→2).

| Split | Ảnh | Box | with_mask | without_mask | incorrect |
|---|---:|---:|---:|---:|---:|
| train | 1533 | 4243 | 2493 | 851 | 899 |
| val | 124 | 560 | 442 | 101 | 17 |
| test | 124 | 563 | 443 | 102 | 18 |

Dataset audit0 lỗi/0 cảnh báo. Public val/test496 file image+label byte-identical với v3. Custom phone/multi chỉ train, exact duplicate/cross-source conflict và near matches public holdout bị loại. Không có camera train; không chứng minh identity-disjoint, không có custom test độc lập. Dataset cũ đóng băng; review thêm không tự đổi model đã train.

## Training thật

RTX3050 Laptop6GB, CUDA0, torch2.11.0+cu128, Ultralytics8.4.174. YOLOv8n, init từ checkpoint best đã lưu của v5 sau 7 epoch (v5 khởi tạo từ v3, chuỗi fine-tune official COCO);355 tensor equality trước optimizer được xác nhận. Image512/batch8/workers0/seed42/AdamW initialLR0.00008/lrf0.2/patience5/ampFalse. Chạy 8/12 epoch, best epoch3 theo validation fitness. Loss/metrics trong `results.csv`, plots/args/init verification/metadata ở run; log `reports/baseline_v6/training.log`.

Run: `D:\PTIT\IoT_UD\Lab2_FaceMask\results\training\baseline_v6_2026-10-11T114225.941386_0000`. Best: `D:\PTIT\IoT_UD\Lab2_FaceMask\results\training\baseline_v6_2026-10-11T114225.941386_0000\weights\best.pt`. Last: `D:\PTIT\IoT_UD\Lab2_FaceMask\results\training\baseline_v6_2026-10-11T114225.941386_0000\weights\last.pt`. Best SHA256: `b95b35539dd06039da11846facbb903a92fff1a23bb45b7614dc47b7a55f8f9b`. Published model: `models/face_mask_best.pt` với metadata/hash của model đã chọn.

## Evaluation sau khi khóa checkpoint

| Model/split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| v3 validation | 0.8949 | 0.7421 | 0.8316 | 0.5672 |
| v6 validation | 0.9302 | 0.6850 | 0.8100 | 0.5487 |
| v6 test | 0.8230 | 0.7922 | 0.8417 | 0.5304 |

Chọn `baseline_v3`; model_selection.json ghi thời điểm khóa trước test. Standardized evaluator image512/NMSIoU0.45/AP minconf0.001. Operational count dùng conf0.35, matching class-aware IoU0.5; P/R framework có operating point riêng.

| Test class | P | R | mAP50 | mAP50-95 | GT | Pred | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with_mask | 0.9391 | 0.8962 | 0.9374 | 0.6375 | 443 | 421 | 24 | 46 |
| without_mask | 0.7988 | 0.8137 | 0.8575 | 0.5274 | 102 | 104 | 22 | 20 |
| mask_weared_incorrect | 0.7311 | 0.6667 | 0.7301 | 0.4263 | 18 | 17 | 5 | 6 |

Test offline warm latency mean21.84/median16.32/p9564.05ms,124ảnh, CUDA synchronize, decodeexcluded. Không phải thời gian ESP32/Wi-Fi. Metrics/predictions/CSV/Markdown/confusion/PR/16 error examples mỗi split tại `results/evaluation/baseline_v6/{validation,test}`.

## Kiểm thử, chạy demo và việc còn lại

87 ca regression đều pass: suite đầu84pass/3skip; chạy riêng3ca với TEST_MODEL_PATH thật đều pass. Logs `reports/baseline_v6/regression-tests.log` và `real-model-tests.log`; kiểm giữ raw/manifest/holdout và cases filter stale/missing-face. API/dashboard lỗi/0/1/multi đã test ở v3; báo cáo smoke model hiện dùng tại `reports/baseline_v6/runtime_smoke.json`. Frontend source giữ bản đã sửa chọn từng box, cho phép nhiều box cùng class. Firmware AIThinker đã compile lại sau sửa URL laptop192.168.1.5 (flash1178185byte/37%,RAM60816/18%). Kết quả v3 đầy đủ được giữ trong `docs/BASELINE_RESULT_V3.md` và reports cũ.

```powershell
Set-Location 'D:\PTIT\IoT_UD\Lab2_FaceMask'
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Không chạy thêm server nếu port8000 đang dùng. Mở `http://127.0.0.1:8000`. Server kiểm thử phiên dùng DB riêng/tắtTelegram; launch bình thường dùng .env local. Laptop IP kiểm lại DHCP; ESP32 gửi `http://<IP_LAPTOP>:8000/predict`, không dùng localhost. Camera base `http://<IP_ESP32>` port80, stream redirect81, capture80.

26JPEG tại `data/raw/camera32_capture` đã được bạn xác minh hoàn tất:33box người thật, 7/12/14 theo class0/1/2. Đã đánh giá V3/V6, nới8box ngữ cảnh trên bản riêng, sửa NMS đếm trùng. Raw và manifest human giữ byte-identical; camera không vào train. Qua API JPEG gốc, class2 khớp6/14, tổng18/33box; xem [báo cáo camera](../reports/esp32_v1/FINAL_RESULT.md). Chưa kiểm hardware capture/send/Wi-Fi; camera26 đã dùng debug hậu xử lý nên cần lượt mới để test độc lập. Detector2D có thể bỏ sót/nhầm class/đếm poster; lọc nhãn training không bảo đảm phân biệt người thật tuyệt đối. Incorrect val17/test18box nên còn hạn chế. Không upload private data, commit hoặc push.

### Kiểm tra ảnh mẫu nhiều người và camera mới

Checkpoint v6 được kiểm riêng trên7ảnh thuộc train: tổng19GT,18TP,0FP,1FN (class-aware IoU0.5,conf0.35); chưa phải test độc lập. Ảnh index1000 có4người, v6 đếm4, v3 đang phục vụ đếm5. Ảnh1002 có3người, v6 bỏ sót1. Không thấy detection khớp các box poster đã lọc trong những ảnh mẫu được chọn; không khẳng định loại mọi poster. Báo cáo reports/baseline_v6/custom_candidate_smoke.json; không dùng smoke này chọn model.

Review26camera đã hoàn tất bởi Mose. Hai proposal poster từng lọc khi còn nháp vẫn có snapshot; mọi nhãn human hiện giữ nguyên. Các box ngữ cảnh assistant sửa ghi provenance riêng, không tự human verified. Hướng dẫn [TEST_CAMERA_NOW](TEST_CAMERA_NOW.md). Runtime NMS class-agnostic đã làm public validation mAP50-95 giảm từ baseline0.5672 xuống0.5125; sửa đếm trùng không đồng nghĩa tăng AP/class accuracy. Camera runtime processed mAP50-95=0.2624, rawHTTPclass2=6/14.
