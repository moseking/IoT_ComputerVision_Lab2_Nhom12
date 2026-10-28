"use strict";
const $ = id => document.getElementById(id);
let apiBase = location.origin, cameraBase = "", apiKey = "", rows = [], lastRequest = null;
let preview = false, polling = false, imageObjectUrl = null, cameraPreviewError = false;
let liveDetection = false, liveTimer = null, liveBusy = false, liveGeneration = 0;
let liveMessage = "", liveFailure = false, flashState = null, flashBusy = false;
let activeFrameResult = null, frozenResult = null, singleCaptureBusy = false;
let inferenceTarget = "local", cloudConfigured = false, cloudTimeoutMs = 200000;
let cloudProviderLabel = "Cloud";
const LIVE_INTERVAL_MS = 50;
let cameraSource = "esp32", webcamErrorMessage = "";
const webcam = new WebcamCapture($("webcamVideo"));
function configureDashboard(config) {
  cloudConfigured = config.cloud_configured === true;
  cloudProviderLabel = typeof config.cloud_provider === "string" && config.cloud_provider ? config.cloud_provider : "Cloud";
  $("cloudOption").textContent = cloudProviderLabel;
  $("cloudOption").disabled = !cloudConfigured;
  $("cloudHint").textContent = cloudConfigured ? "Ảnh được gửi tới " + cloudProviderLabel : "Cloud chưa cấu hình; chọn Máy tính nếu muốn chạy local.";
  if (["local", "cloud"].includes(config.default_inference_target)) inferenceTarget = config.default_inference_target;
  if (["esp32", "webcam"].includes(config.default_camera_source)) cameraSource = config.default_camera_source;
  // A requested cloud default is retained even when unconfigured: never silently run local.
  $("inferenceTarget").value = inferenceTarget;
  $("cameraSource").value = cameraSource;
  $("inferenceLocationLabel").textContent = "MODEL INFERENCE · " + inferenceTarget.toUpperCase();
  setCamera();
}
function stopWebcam() { webcam.stop(); syncWebcamControls(); }
function syncWebcamControls() {
  if (cameraSource === "webcam") {
    $("deviceStatus").textContent = webcam.ready ? "WEBCAM · ĐÃ BẬT" : "WEBCAM · ĐÃ TẮT";
    $("deviceStatus").style.color = webcam.ready ? "#087e61" : "#617085";
    $("lastSeen").textContent = "Camera laptop qua trình duyệt";
  }
  $("webcamToggle").textContent = webcam.starting ? "Hủy bật webcam" : webcam.ready ? "Tắt webcam" : "Bật webcam";
  $("webcamStatus").textContent = webcam.ready ? "Webcam đã bật · Bấm Chụp hoặc Live để nhận diện" : webcam.starting ? "Chờ bạn cấp quyền camera…" : webcamErrorMessage || "Webcam chưa bật · Không cần Camera URL";
}
$("cameraSource").onchange = () => {
  stopLiveDetection(); stopWebcam(); preview = false;
  webcamErrorMessage = ""; cameraSource = $("cameraSource").value;
  activeFrameResult = null; frozenResult = null; lastRequest = null;
  selectView("resultButton"); setCamera();
  pendingFrame(cameraSource === "webcam" ? "WEBCAM LAPTOP · SẴN SÀNG" : "ESP32-CAM · SẴN SÀNG", "Bấm Chụp + nhận diện hoặc Live để lấy ảnh từ nguồn vừa chọn.");
};
$("webcamToggle").onclick = async () => {
  if (webcam.ready || webcam.starting) { stopLiveDetection(); stopWebcam(); selectView("resultButton"); return; }
  const generation = liveGeneration;
  webcamErrorMessage = ""; const work = webcam.start(); syncWebcamControls();
  try { await work; }
  catch (error) { if (generation === liveGeneration && error.name !== "AbortError") { webcamErrorMessage = error.message; notice(error.message, true); } }
  finally { syncWebcamControls(); }
};
window.addEventListener("pagehide", () => { stopLiveDetection(); stopWebcam(); });
document.addEventListener("visibilitychange", () => {
  if (document.hidden && cameraSource === "webcam") { stopLiveDetection(); stopWebcam(); selectView("resultButton"); }
});
async function webcamPrediction(manual, generation) {
  const started = performance.now();
  let capture;
  try { capture = await webcam.capture(); } catch (error) { if (error.name === "WebcamError") webcamErrorMessage = error.message; throw error; }
  syncWebcamControls();
  if (generation !== liveGeneration || cameraSource !== "webcam") throw webcam.cancelled();
  const data = new FormData();
  data.append("file", capture.blob, "webcam.jpg"); data.append("source", "webcam");
  data.append("capture_ms", String(capture.capture_ms)); data.append("include_image", "true");
  data.append("inference_target", inferenceTarget); data.append("send_telegram", String(manual));
  const requestStarted = performance.now();
  const response = await request("/predict", {method: "POST", body: data, timeout: inferenceTarget === "cloud" ? cloudTimeoutMs : 60000});
  const body = await response.text(), received = performance.now(), result = JSON.parse(body);
  attachClientTimings(result, received - requestStarted, received - started);
  // Preserve the exact JPEG submitted, never a later webcam frame.
  const bytes = new Uint8Array(await capture.blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  result.original_image_base64 = btoa(binary); result.original_image_content_type = "image/jpeg";
  result.original_image_filename = "webcam_" + result.request_id + ".jpg";
  return result;
}

function attachClientTimings(result, roundtrip, total) {
  result.roundtrip_ms = result.inference_location === "cloud" ? result.cloud_roundtrip_ms : roundtrip;
  result.total_ms = total;
}
$("inferenceTarget").onchange = () => {
  const target = $("inferenceTarget").value;
  if (target === "cloud" && !cloudConfigured) { $("inferenceTarget").value = inferenceTarget; return; }
  stopLiveDetection(); stopWebcam(); inferenceTarget = target; preview = false;
  activeFrameResult = null; frozenResult = null; lastRequest = null;
  selectView("resultButton");
  pendingFrame(target === "cloud" ? cloudProviderLabel.toUpperCase() + " · SẴN SÀNG" : "LOCAL · SẴN SÀNG", "Bấm Chụp + nhận diện hoặc Live để nhận diện tại nơi đã chọn.");
  $("emptyFrame").querySelector("h3").textContent = "Sẵn sàng nhận diện";
  $("inferenceLocationLabel").textContent = "MODEL INFERENCE · " + target.toUpperCase();
};
let liveMetricsBusy = false, liveMetricsPending = null, liveMetricsError = "";

function queueLiveMetrics(result, generation) {
  // One report in flight and one newest pending report: a slow DB cannot stall Live.
  liveMetricsPending = {generation, values: {request_id: result.request_id, report_token: result.report_token,
    capture_ms: result.capture_ms ?? null, roundtrip_ms: result.roundtrip_ms, total_ms: result.total_ms, upload_ms: null, download_ms: null}};
  void flushLiveMetrics();
}
async function flushLiveMetrics() {
  if (liveMetricsBusy) return;
  liveMetricsBusy = true;
  try {
    while (liveMetricsPending) {
      const pending = liveMetricsPending; liveMetricsPending = null;
      try {
        await json("/metrics/report", {method: "POST", timeout: 3000,
          headers: {"Content-Type": "application/json"}, body: JSON.stringify(pending.values)});
        if (pending.generation === liveGeneration) liveMetricsError = "";
      } catch (error) {
        if (liveDetection && pending.generation === liveGeneration) liveMetricsError = "Chưa lưu được số đo client.";
      }
    }
  } finally { liveMetricsBusy = false; }
}
const classLabels = ["WITH MASK", "WITHOUT MASK", "INCORRECT MASK"];
const classColors = ["#087e61", "#ca4053", "#ad6900"];
const countKeys = ["with_mask", "without_mask", "mask_weared_incorrect"];
const countLabels = ["Đeo đúng", "Không đeo", "Đeo sai"];
const countStyles = ["correct", "absent", "incorrect"];
function labelCounts(row) {
  const counts = row.class_counts;
  if (counts && countKeys.every(key => Number.isInteger(counts[key]) && counts[key] >= 0)) return counts;
  return null;
}
function sourceLabel(row) {
  const source = {esp32:"ESP32-CAM", camera_pull:"ESP32-CAM", webcam:"Webcam laptop", dashboard:"Ảnh kiểm thử", test_client:"API"}[row.source] || row.source;
  return source + (row.inference_location === "cloud" ? " · Cloud" : " · Máy tính");
}
function countChips(counts, target) {
  countKeys.forEach((key, index) => {
    const chip = document.createElement("span"); chip.className = "count-chip " + countStyles[index] + (counts[key] === 0 ? " zero" : "");
    const value = document.createElement("b"); value.textContent = counts[key];
    chip.append(countLabels[index] + " ", value); target.append(chip);
  });
}
function recognitionRows() {
  return rows.filter(row => row.status !== "failure");
}
function renderRecognitionHistory() {
  const body = $("recognitionBody"); body.replaceChildren();
  const items = recognitionRows();
  $("recognitionRowsNote").textContent = `${items.length} lượt hiển thị · trong 50 lượt gần nhất`;
  if (!items.length) {
    const tr = document.createElement("tr"), td = document.createElement("td"); td.colSpan = 4;
    td.className = "recognition-empty"; td.textContent = "Chưa có kết quả đã lưu. Chụp hoặc bật Live để bắt đầu.";
    tr.append(td); body.append(tr); return;
  }
  items.forEach(row => {
    const tr = document.createElement("tr");
    [date(row.timestamp), sourceLabel(row), row.detection_count ?? "—"].forEach(text => {
      const td = document.createElement("td"); td.textContent = text; tr.append(td);
    });
    const td = document.createElement("td"), counts = labelCounts(row);
    if (counts) {
      const chips = document.createElement("div"); chips.className = "count-chips"; countChips(counts, chips); td.append(chips);
      if (row.detection_count === 0) { const hint = document.createElement("small"); hint.className = "row-hint"; hint.textContent = "Không phát hiện mặt"; td.append(hint); }
    } else { const hint = document.createElement("span"); hint.className = "legacy-labels"; hint.textContent = "Chưa lưu nhãn"; td.append(hint); }
    tr.append(td); body.append(tr);
  });
}
function renderRecognitionSummary(recognition) {
  [["correct", "with_mask"], ["absent", "without_mask"], ["incorrect", "mask_weared_incorrect"]].forEach(([id, key]) => {
    const values = recognition?.classes?.[key];
    $(id + "Total").textContent = values ? values.detections.toLocaleString("vi-VN") : "—";
    $(id + "Requests").textContent = values ? `${values.requests.toLocaleString("vi-VN")} lượt có nhãn này` : "Chưa có số liệu";
  });
  $("recognitionSummaryNote").textContent = recognition
    ? `Đã lưu nhãn ${recognition.labeled_requests.toLocaleString("vi-VN")} lượt${recognition.unlabeled_requests ? ` · ${recognition.unlabeled_requests.toLocaleString("vi-VN")} lượt cũ chưa lưu nhãn` : ""}. Cộng theo lượt nhận diện; một người trong nhiều frame Live được tính nhiều lần.`
    : "Chưa nhận được lịch sử nhãn. Khởi động lại backend để áp dụng phần lưu nhãn mới.";
}
$("recognitionExportButton").onclick = () => {
  const keys = ["timestamp", "request_id", "source", "inference_location", "detection_count", "with_mask_count", "without_mask_count", "incorrect_mask_count"];
  const csv = [keys.join(","), ...recognitionRows().map(row => keys.map(key => JSON.stringify(row[key] ?? "")).join(","))].join("\r\n");
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob(["\ufeff" + csv], {type: "text/csv;charset=utf-8"}));
  link.download = "recognition_history.csv"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
};
const fmt = value => value == null ? "—" : Number(value).toFixed(1);
const date = value => value ? new Date(value).toLocaleString("vi-VN") : "—";
function notice(text, error = false) { $("notice").textContent = text; $("notice").classList.toggle("error", error); }
function normalizeUrl(value) {
  const url = new URL(value, location.origin);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("URL phải dùng http hoặc https");
  return url.href.replace(/\/$/, "");
}
async function request(path, options = {}) {
  const response = await fetch(apiBase + path, { ...options,
    headers: { ...(apiKey ? {"X-API-Key": apiKey} : {}), ...options.headers },
    signal: AbortSignal.timeout(options.timeout || 15000), cache: "no-store" });
  if (!response.ok) {
    let error; try { error = await response.json(); } catch { error = {}; }
    throw new Error(`HTTP ${response.status}: ${JSON.stringify(error.detail || "Request failed")}`);
  }
  return response;
}
async function json(path, options) { return (await request(path, options)).json(); }
function setCamera() {
  const laptop = cameraSource === "webcam";
  $("cameraDeviceLabel").textContent = laptop ? "WEBCAM LAPTOP" : "THIẾT BỊ ESP32-CAM";
  $("cameraAddressLabel").textContent = laptop ? "NGUỒN CAMERA" : "CAMERA IP / URL";
  $("cameraAddress").textContent = laptop ? "Camera của trình duyệt" : cameraBase || "Chưa cấu hình";
  $("cameraLink").hidden = laptop || !cameraBase;
  if (cameraBase) $("cameraLink").href = cameraBase;
  $("flashButton").hidden = laptop; $("flashStatus").hidden = laptop;
  $("flashButton").disabled = laptop || !cameraBase;
  $("webcamControls").hidden = !laptop; syncWebcamControls();
}
function selectView(name) {
  for (const id of ["resultButton", "liveButton"]) $(id).className = id === name ? "selected" : "secondary";
}
function stopLiveDetection() {
  liveDetection = false; liveGeneration++;
  if (liveTimer !== null) clearTimeout(liveTimer);
  liveTimer = null;
}
function unloadFrame() {
  $("frame").onload = null; $("frame").removeAttribute("src");
  if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
  imageObjectUrl = null;
  $("frame").hidden = true; $("emptyFrame").hidden = false;
}
function pendingFrame(source, message) {
  unloadFrame(); $("frameSource").textContent = source; $("frameTime").textContent = "—";
  $("emptyFrame").querySelector("h3").textContent = "Đang chờ ảnh nhận diện";
  $("emptyFrame").querySelector("p").textContent = message;
  clearFrameResult(message);
}
function clearFrameResult(message) {
  $("saveRawButton").disabled = true;
  $("captureTelegramStatus").textContent = "";
  $("currentCounts").replaceChildren(); $("currentCounts").hidden = true;
  $("peopleCount").textContent = "—"; $("detections").replaceChildren();
  const p = document.createElement("p"); p.className = "empty-list"; p.textContent = message; $("detections").append(p);
  for (const id of ["inferenceMs", "roundtripMs", "totalMs", "captureMs"]) $(id).textContent = "—";
}
async function liveCameraTick(generation) {
  if (!liveDetection || generation !== liveGeneration) return;
  if (liveBusy || flashBusy) { liveTimer = setTimeout(() => liveCameraTick(generation), LIVE_INTERVAL_MS); return; }
  liveBusy = true;
  try {
    const started = performance.now();
    let result;
    if (cameraSource === "webcam") result = await webcamPrediction(false, generation);
    else {
      const response = await request("/camera/predict", {method: "POST", timeout: inferenceTarget === "cloud" ? cloudTimeoutMs : 20000,
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({camera_url: cameraBase, inference_target: inferenceTarget})});
      const body = await response.text(); const received = performance.now(); result = JSON.parse(body);
      attachClientTimings(result, received - started, received - started);
    }
    if (!liveDetection || generation !== liveGeneration) return;
    await renderResult(result, true);
    liveMessage = cameraSource === "webcam" ? `Webcam Live · Chụp JPEG ${fmt(result.capture_ms)} ms · YOLO ${fmt(result.inference_ms)} ms.` : `Live YOLO · Lấy ảnh ${fmt(result.camera_fetch_ms)} ms · YOLO ${fmt(result.inference_ms)} ms.`;
    if (liveMetricsError) liveMessage += " " + liveMetricsError;
    liveFailure = false; notice(liveMessage);
    queueLiveMetrics(result, generation);
  } catch (error) {
    if (liveDetection && generation === liveGeneration) {
      if (cameraSource === "webcam" && ["WebcamError", "AbortError"].includes(error.name)) { stopLiveDetection(); stopWebcam(); selectView("resultButton"); notice(error.message, true); return; }
      liveMessage = "Live nhận diện lỗi: " + error.message + (cameraSource === "webcam" ? ". Kiểm tra kết nối backend/cloud." : ". Kiểm tra Camera URL/Wi-Fi.");
      liveFailure = true; notice(liveMessage, true);
      if (activeFrameResult && !$("frame").hidden) {
        $("frameSource").textContent = "LIVE YOLO · GIỮ ẢNH CŨ · ĐANG THỬ LẠI";
        liveMessage += " Đang giữ ảnh gần nhất; xem thời gian trên ảnh."; notice(liveMessage, true);
      } else {
        $("frame").hidden = true; $("emptyFrame").hidden = false;
        $("frameSource").textContent = "LIVE YOLO · CHƯA CÓ FRAME MỚI";
        clearFrameResult("Chưa nhận được frame mới để nhận diện.");
      }
    }
  } finally {
    liveBusy = false;
    if (liveDetection && generation === liveGeneration) liveTimer = setTimeout(() => liveCameraTick(generation), liveFailure ? 1000 : LIVE_INTERVAL_MS);
  }
}
function renderDetections(result) {
  $("peopleCount").textContent = result.detection_count;
  const counts = Object.fromEntries(countKeys.map((key, index) => [key, result.detections.filter(d => d.class_id === index).length]));
  $("currentCounts").replaceChildren(); countChips(counts, $("currentCounts")); $("currentCounts").hidden = false;
  const list = $("detections");
  list.replaceChildren();
  if (!result.detections.length) {
    const p = document.createElement("p"); p.className = "empty-list"; p.textContent = "Model không phát hiện khuôn mặt ở frame này."; list.append(p);
  }
  result.detections.forEach((d, index) => {
    const row = document.createElement("div"); row.className = "detection";
    const top = document.createElement("div"); top.className = "detection-top";
    const label = document.createElement("span"); label.textContent = `#${index + 1}  ${classLabels[d.class_id]}`; label.style.color = classColors[d.class_id];
    const confidence = document.createElement("strong"); confidence.textContent = (d.confidence * 100).toFixed(1) + "%";
    top.append(label, confidence);
    const small = document.createElement("small"); small.textContent = `BBox: ${Object.values(d.bbox).map(v => Math.round(v)).join(", ")}`;
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("span"); fill.style.background = classColors[d.class_id]; fill.style.width = Math.max(0, Math.min(100, d.confidence * 100)) + "%";
    bar.append(fill); row.append(top, small, bar); list.append(row);
  });
}
async function renderResult(result, fromLive = false) {
  if (!result || preview || (liveDetection && !fromLive)) return;
  const generation = liveGeneration;
  if (result.request_id !== lastRequest || $("frame").hidden) {
    let nextUrl;
    try {
      nextUrl = result.annotated_image_base64 ? "data:image/jpeg;base64," + result.annotated_image_base64 :
        URL.createObjectURL(await (await request(result.annotated_image_url)).blob());
    } catch (error) {
      if (generation === liveGeneration) lastRequest = null;
      return;
    }
    if (generation !== liveGeneration || preview || (liveDetection && !fromLive)) {
      if (!result.annotated_image_base64) URL.revokeObjectURL(nextUrl);
      return;
    }
    const oldUrl = imageObjectUrl;
    $("frame").onload = () => { if (oldUrl) URL.revokeObjectURL(oldUrl); };
    $("frame").src = nextUrl; imageObjectUrl = result.annotated_image_base64 ? null : nextUrl;
    $("frame").hidden = false; $("emptyFrame").hidden = true;
    renderDetections(result); lastRequest = result.request_id;
  }
  $("inferenceLocationLabel").textContent = "MODEL INFERENCE · " + (result.inference_location === "cloud" ? "CLOUD" : "LOCAL");
  if (result.inference_location === "cloud") { $("modelStatus").textContent = "CLOUD · ĐÃ NHẬN KẾT QUẢ"; $("modelDetails").textContent = `V7 ONNX · ${cloudProviderLabel} CPU · 3 classes`; }
  $("roundtripDefinition").textContent = result.inference_location === "cloud" ? "Máy tính gửi ảnh → nhận hết kết quả Cloud" : "Gửi request → nhận hết response";
  $("inferenceMs").textContent = fmt(result.inference_ms);
  $("roundtripMs").textContent = fmt(result.roundtrip_ms);
  $("totalMs").textContent = fmt(result.end_to_end_ms ?? result.total_ms);
  $("captureMs").textContent = fmt(result.capture_ms);
  $("captureDefinition").textContent = result.source === "webcam" ? "Chụp và mã hóa JPEG trên trình duyệt" : "Lấy và sao chép frame trên ESP32";
  $("totalDefinition").textContent = result.source === "esp32" ? "Capture → nhận hết kết quả" : result.source === "camera_pull" ? "Dashboard yêu cầu → nhận ảnh YOLO" : result.source === "webcam" ? "Webcam: yêu cầu chụp → nhận kết quả" : "Ảnh offline: chuẩn bị → nhận kết quả";
  activeFrameResult = result;
  if (!result.telegram_delivery?.requested) $("captureTelegramStatus").textContent = "";
  $("saveRawButton").disabled = !(["camera_pull", "webcam"].includes(result.source) && result.original_image_base64);
  $("frameSource").textContent = result.source === "esp32" ? "ESP32-CAM · INFERENCE" : result.source === "camera_pull" ? (fromLive ? "ESP32-CAM · LIVE YOLO" : "ESP32-CAM · ẢNH ĐÃ NHẬN DIỆN") : result.source === "webcam" ? (fromLive ? "WEBCAM LAPTOP · LIVE YOLO" : "WEBCAM LAPTOP · ẢNH ĐÃ NHẬN DIỆN") : "ẢNH KIỂM THỬ · INFERENCE";
  $("frameSource").textContent += result.inference_location === "cloud" ? " · CLOUD" : " · LOCAL";
  $("frameTime").textContent = date(result.timestamp);
}
function renderHistory(items) {
  rows = items;
  const body = $("historyBody"); body.replaceChildren();
  if (!items.length) {
    const tr = document.createElement("tr"), td = document.createElement("td");
    td.colSpan = 7; td.textContent = "Chưa có request nào."; tr.append(td); body.append(tr);
  }
  items.forEach(row => {
    const tr = document.createElement("tr");
    [date(row.timestamp), row.source === "esp32" ? "ESP32-CAM · " + (row.inference_location || "local") : row.source === "camera_pull" ? "ESP32 · " + (row.inference_location || "local") : (row.source === "webcam" ? "Webcam laptop" : row.source) + (row.inference_location === "cloud" ? " · Cloud" : " · Local"), row.status === "failure" ? `Lỗi ${row.http_status} · ${row.error_code}` : "Thành công", row.detection_count ?? "—",
      fmt(row.inference_ms) + " ms", fmt(row.roundtrip_ms) + " ms", fmt(row.total_ms) + " ms"].forEach(text => {
        const td = document.createElement("td"); td.textContent = text; tr.append(td);
      });
    body.append(tr);
  });
  renderRecognitionHistory();
  drawChart();
}
function renderSummary(summary) {
  renderRecognitionSummary(summary.recognition);
  $("requestCount").textContent = `${summary.total_requests} thành công / ${summary.failed_requests || 0} lỗi đã lưu trong SQLite`;
  const target = $("summaryValues"); target.replaceChildren();
  [["inference_ms", "AVG. INFERENCE"], ["roundtrip_ms", "AVG. ROUND TRIP"], ["total_ms", "AVG. END-TO-END"]].forEach(([key, label]) => {
    const metric = summary[key];
    const cell = document.createElement("div"); cell.className = "summary-stat";
    const title = document.createElement("span"); title.className = "label"; title.textContent = label;
    const average = document.createElement("strong"); average.textContent = fmt(metric.average) + " ms";
    const range = document.createElement("small"); range.textContent = `Min ${fmt(metric.min)} / Max ${fmt(metric.max)} ms · ${metric.samples} mẫu đo`;
    cell.append(title, average, range); target.append(cell);
  });
}
function drawChart() {
  const canvas = $("latencyChart"), dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = 190;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  const chronological = rows.slice(0, 40).reverse();
  const metrics = [["inference_ms", "#087e61"], ["roundtrip_ms", "#3265d6"], ["total_ms", "#ad6900"]];
  const values = chronological.flatMap(r => metrics.map(([key]) => r[key]).filter(v => v != null));
  const maximum = Math.max(100, ...values) * 1.1;
  const left = 76, right = w - 16, top = 14, bottom = h - 30;
  ctx.font = "12px Segoe UI"; ctx.strokeStyle = "#e4e9f0"; ctx.fillStyle = "#617085";
  for (let i = 0; i <= 4; i++) {
    const y = top + (bottom - top) * i / 4;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillText(Math.round(maximum * (1 - i / 4)) + " ms", 2, y + 3);
  }
  if (!values.length) { ctx.fillText("Chưa có số đo latency", left + 15, h / 2); return; }
  const x = i => left + (right - left) * i / Math.max(1, chronological.length - 1);
  metrics.forEach(([key, color]) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let connected = false;
    chronological.forEach((row, i) => {
      if (row[key] == null) { connected = false; return; }
      const y = bottom - row[key] / maximum * (bottom - top);
      if (connected) ctx.lineTo(x(i), y); else ctx.moveTo(x(i), y);
      connected = true;
    });
    ctx.stroke();
    chronological.forEach((row, i) => {
      if (row[key] == null) return;
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x(i), bottom - row[key] / maximum * (bottom - top), 2, 0, Math.PI * 2); ctx.fill();
    });
  });
  ctx.fillStyle = "#617085";
  ctx.fillText(new Date(chronological[0].timestamp).toLocaleTimeString("vi-VN"), left, h - 7);
  ctx.textAlign = "right"; ctx.fillText(new Date(chronological.at(-1).timestamp).toLocaleTimeString("vi-VN"), right, h - 7);
}
async function poll() {
  if (polling || flashBusy) return;
  polling = true;
  try {
    const [health, device, history, summary, telegram, model] = await Promise.all([
      json("/health"), json("/device/status"), json("/metrics/history?limit=50"), json("/metrics/summary"),
      json("/telegram/status"), json("/model/info")]);
    const smoke = model.metadata?.epochs_completed <= 1;
    const telegramLabels = {disabled: "Đã tắt", not_configured: "Chưa cấu hình", starting: "Đang kết nối", ready: "Đã xác nhận Bot API", error: "Lỗi kết nối / gửi"};
    $("telegramStatus").textContent = telegramLabels[telegram.state] || "Chưa xác nhận";
    $("telegramStatus").style.color = telegram.state === "ready" ? "#087e61" : telegram.state === "error" ? "#ca4053" : "#617085";
    $("telegramDetails").textContent = telegram.last_error || (telegram.last_sent ? "Gửi gần nhất: " + date(telegram.last_sent) : `Chế độ ${telegram.alert_mode} · cooldown ${telegram.cooldown_seconds}s`);
    $("modelDetails").textContent = `${model.metadata?.version || model.checkpoint} · ${model.device} · 3 classes`;

    if (inferenceTarget === "cloud") {
      $("modelDetails").textContent = `V7 ONNX · ${cloudProviderLabel} CPU · 3 classes`;
      $("modelStatus").textContent = activeFrameResult?.inference_location === "cloud" ? "CLOUD · ĐÃ NHẬN KẾT QUẢ" : "CLOUD · CHỜ NHẬN DIỆN";
    }
    const apiReady = inferenceTarget === "cloud" || health.model_loaded;
    $("apiBadge").textContent = apiReady ? "API CONNECTED" : "API · MODEL CHƯA SẴN SÀNG";
    $("apiBadge").className = "pill" + (apiReady ? "" : " warn");
    $("deviceStatus").textContent = device.status === "online" ? "ONLINE · Có frame mới" : (!liveDetection && !singleCaptureBusy ? "CHỜ CHỤP / LIVE" : "Chưa có frame gần đây");
    $("deviceStatus").style.color = device.status === "online" ? "#087e61" : "#617085";
    $("lastSeen").textContent = device.last_seen ? "Frame gần nhất: " + date(device.last_seen) : "Chờ frame đầu tiên từ ESP32-CAM";
    if (cameraSource === "webcam") { $("deviceStatus").textContent = webcam.ready ? "WEBCAM · ĐÃ BẬT" : "WEBCAM · ĐÃ TẮT"; $("lastSeen").textContent = "Camera laptop qua trình duyệt"; syncWebcamControls(); }
    if (inferenceTarget === "local") $("modelStatus").textContent = health.model_loaded ? (smoke ? "MODEL KIỂM THỬ · 1 EPOCH" : "YOLOv8n · LOADED") : "Checkpoint chưa khả dụng";
    if (inferenceTarget === "local" && !health.model_loaded) notice("API đã chạy. Train model bằng scripts/train.py rồi khởi động lại server.", true);
    else if (inferenceTarget === "local" && smoke) notice("Checkpoint kiểm thử 1 epoch: chỉ xác minh pipeline, chưa dùng để demo chất lượng.", true);
    else if (cameraPreviewError && preview) notice("Camera stream lỗi. Mở Camera URL để kiểm tra /capture và port 81.", true);
    else if (liveDetection) notice(liveMessage || "Đang lấy frame camera để nhận diện…", liveFailure);
    else if (preview) notice("Camera gốc: không chạy YOLO trong stream này. Chọn Live nhận diện để xem box cập nhật.");
    else if (inferenceTarget === "cloud" && !activeFrameResult) notice("Cloud có thể cần thời gian khởi động. Bấm Chụp + nhận diện để thử.");
    else notice(activeFrameResult ? "Ảnh đang giữ nguyên để kiểm tra. Bấm Chụp + nhận diện để lấy ảnh mới, hoặc Live nhận diện để cập nhật liên tục." : "Bấm Chụp + nhận diện để kiểm tra một ảnh, hoặc Live nhận diện để xem liên tục.");
    if (!liveDetection && !singleCaptureBusy && !$("frame").hidden && activeFrameResult?.telegram_delivery?.state === "queued" && telegram.last_sent_request_id === activeFrameResult.request_id) {
      $("captureTelegramStatus").textContent = "Đã gửi ảnh này lên Telegram";
    }
    renderHistory(history.items); renderSummary(summary);
  } catch (error) {
    $("apiBadge").textContent = "API DISCONNECTED"; $("apiBadge").className = "pill off";
    $("modelStatus").textContent = "Không lấy được trạng thái model";
    $("telegramStatus").textContent = "Không lấy được trạng thái Telegram";
    $("telegramStatus").style.color = "#617085";
    $("telegramDetails").textContent = "Chờ backend kết nối lại";
    $("modelDetails").textContent = "Chờ backend kết nối lại";
    $("deviceStatus").textContent = "Backend chưa xác nhận frame";
    $("deviceStatus").style.color = "#617085";
    notice("Không lấy được dữ liệu: " + error.message, true);
  } finally { polling = false; }
}
$("saveRawButton").onclick = async () => {
  const result = activeFrameResult;
  if (!result || !["camera_pull", "webcam"].includes(result.source) || !result.original_image_base64 || singleCaptureBusy) return;
  stopLiveDetection(); stopWebcam(); preview = false; frozenResult = {...result}; selectView("resultButton");
  // Start download within the click gesture, before awaiting rendering.
  const bytes = Uint8Array.from(atob(result.original_image_base64), value => value.charCodeAt(0));
  const rawUrl = URL.createObjectURL(new Blob([bytes], {type: result.original_image_content_type}));
  const link = document.createElement("a"); link.href = rawUrl; link.download = result.original_image_filename;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(rawUrl), 30000);
  await renderResult(result);
  notice("Đã yêu cầu tải ảnh gốc của frame đang xem, chưa vẽ box/chữ. Kiểm tra mục tải xuống rồi chuyển file vào dataset để review.");
};
$("settingsButton").onclick = () => { $("settingsPanel").hidden = !$("settingsPanel").hidden; };
$("saveSettings").onclick = () => {
  try {
    apiBase = normalizeUrl($("apiInput").value);
    cameraBase = $("cameraInput").value ? normalizeUrl($("cameraInput").value) : "";
    apiKey = $("keyInput").value; stopLiveDetection(); stopWebcam(); preview = false; flashState = null; activeFrameResult = null; frozenResult = null;
    $("flashButton").textContent = "Bật flash"; $("flashButton").setAttribute("aria-pressed", "false"); $("flashStatus").textContent = "Bấm để bật đèn camera";
    try { localStorage.setItem("face_mask_camera_url", cameraBase); } catch (_) {}
    lastRequest = null; selectView("resultButton"); setCamera();
    pendingFrame("CAMERA ĐÃ CẤU HÌNH", "Bấm Chụp + nhận diện để lấy một ảnh, hoặc Live nhận diện để cập nhật liên tục.");
    $("emptyFrame").querySelector("h3").textContent = "Sẵn sàng chụp hoặc xem Live"; poll();
  } catch (error) { notice(error.message, true); }
};
$("resultButton").onclick = () => {
  const held = liveDetection ? activeFrameResult : (frozenResult || activeFrameResult);
  stopLiveDetection(); stopWebcam(); preview = false; selectView("resultButton");
  if (held) {
    frozenResult = held; lastRequest = null; renderResult(held);
    notice("Đã dừng Live và giữ nguyên ảnh này để kiểm tra box/nhãn.");
  } else {
    pendingFrame("ẢNH CHỤP · CHƯA CÓ ẢNH", "Bấm Chụp + nhận diện để lấy một ảnh từ camera.");
    $("emptyFrame").querySelector("h3").textContent = "Chưa có ảnh chụp";
  }
};
$("liveButton").onclick = () => {
  if (cameraSource === "esp32" && !cameraBase) return notice("Cấu hình Camera URL trước khi mở live nhận diện.", true);
  if (singleCaptureBusy) return;
  stopLiveDetection(); preview = false; liveDetection = true; liveFailure = false;
  liveMessage = cameraSource === "webcam" ? "Đang lấy webcam và chạy YOLO…" : "Đang lấy frame ESP32-CAM và chạy YOLO…";
  lastRequest = null; pendingFrame("LIVE YOLO · ĐANG NHẬN DIỆN", cameraSource === "webcam" ? "Đang lấy JPEG từ webcam và chạy YOLO." : "Đang lấy JPEG mới từ ESP32-CAM và chạy YOLO.");
  selectView("liveButton"); notice(liveMessage);
  liveCameraTick(liveGeneration);
};

$("flashButton").onclick = async () => {
  if (cameraSource !== "esp32" || !cameraBase || flashBusy) return;
  const base = cameraBase, enabled = flashState !== true, started = performance.now();
  flashBusy = true;
  $("flashButton").disabled = true; $("flashButton").textContent = enabled ? "Đang bật…" : "Đang tắt…";
  $("flashStatus").textContent = "Đang chờ camera xác nhận…";
  try {
    const result = await json("/camera/flash", {method: "POST", timeout: 10000,
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({camera_url: base, enabled})});
    if (base !== cameraBase) return;
    if (result.confirmed !== true || result.flash !== enabled) throw new Error("Camera chưa xác nhận trạng thái flash");
    flashState = result.flash; $("flashButton").textContent = flashState ? "Tắt flash" : "Bật flash";
    $("flashButton").setAttribute("aria-pressed", String(flashState));
    const elapsed = Math.round(performance.now() - started);
    $("flashStatus").textContent = (flashState ? "Flash đang bật" : "Flash đã tắt") + ` · camera xác nhận · ${elapsed} ms`;
  } catch (error) {
    if (base === cameraBase) { flashState = null; $("flashStatus").textContent = "Flash chưa xác nhận: " + error.message; $("flashButton").textContent = "Bật flash"; $("flashButton").setAttribute("aria-pressed", "false"); }
  } finally { flashBusy = false; $("flashButton").disabled = cameraSource !== "esp32" || !cameraBase; }
};
$("captureButton").onclick = async () => {
  if (cameraSource === "esp32" && !cameraBase) return notice("Cấu hình Camera URL trước khi chụp.", true);
  if (singleCaptureBusy) return;
  stopLiveDetection(); preview = false; selectView("resultButton");
  const generation = liveGeneration, started = performance.now();
  singleCaptureBusy = true; $("captureButton").disabled = true; $("captureButton").textContent = "Đang chụp…";
  $("captureTelegramStatus").textContent = "";
  pendingFrame("ẢNH CHỤP · ĐANG NHẬN DIỆN", cameraSource === "webcam" ? "Đang chụp một JPEG từ webcam và chạy YOLO." : "Đang lấy một JPEG từ ESP32-CAM và chạy YOLO.");
  try {
    let result;
    if (cameraSource === "webcam") result = await webcamPrediction(true, generation);
    else {
      const response = await request("/camera/predict", {method: "POST", timeout: inferenceTarget === "cloud" ? cloudTimeoutMs : 20000,
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({camera_url: cameraBase, send_telegram: true, inference_target: inferenceTarget})});
      const body = await response.text(), received = performance.now(); result = JSON.parse(body);
      attachClientTimings(result, received - started, received - started);
    }
    if (generation !== liveGeneration || liveDetection) return;
    if (cameraSource === "webcam") stopWebcam();
    frozenResult = result; lastRequest = null; await renderResult(result);
    $("captureTelegramStatus").textContent = result.telegram_delivery?.state === "queued"
      ? "Ảnh này đang chờ gửi Telegram"
      : "Ảnh chưa xếp hàng Telegram; kiểm tra kết nối/cấu hình hoặc hàng đợi";
    notice("Ảnh đã chụp và nhận diện. Giữ nguyên đến khi bạn chụp ảnh khác hoặc bật Live.");
    try {
      await json("/metrics/report", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({request_id: result.request_id, report_token: result.report_token,
          capture_ms: result.capture_ms ?? null, roundtrip_ms: result.roundtrip_ms, total_ms: result.total_ms, upload_ms: null, download_ms: null})});
    } catch (error) { notice("Ảnh đã nhận diện; chưa lưu được số đo client: " + error.message, true); }
  } catch (error) {
    if (generation === liveGeneration) {
      pendingFrame("ẢNH CHỤP · LỖI", cameraSource === "webcam" ? "Kiểm tra quyền webcam hoặc kết nối backend/cloud." : "Không lấy được ảnh mới; kiểm tra Camera URL/Wi-Fi.");
      $("emptyFrame").querySelector("h3").textContent = "Chụp ảnh chưa thành công";
      notice("Chụp + nhận diện lỗi: " + error.message, true);
    }
  } finally { if (generation === liveGeneration && cameraSource === "webcam") stopWebcam(); singleCaptureBusy = false; $("captureButton").disabled = false; $("captureButton").textContent = "Chụp + nhận diện"; }
};
$("testButton").onclick = async () => {
  const file = $("testFile").files[0];
  if (!file) return notice("Chọn ảnh JPEG/PNG để kiểm thử.", true);
  const data = new FormData(); data.append("file", file); data.append("source", "dashboard"); data.append("include_image", "true"); data.append("inference_target", inferenceTarget);
  const totalStart = performance.now(); $("testButton").disabled = true;
  try {
    const started = performance.now();
    const response = await request("/predict", {method: "POST", body: data, timeout: inferenceTarget === "cloud" ? cloudTimeoutMs : 60000});
    // Read complete body before taking the roundtrip timestamp.
    const text = await response.text(); const received = performance.now();
    const result = JSON.parse(text);
    attachClientTimings(result, received - started, received - totalStart);
    await json("/metrics/report", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({request_id: result.request_id, report_token: result.report_token,
        capture_ms: result.capture_ms ?? null, roundtrip_ms: result.roundtrip_ms, total_ms: result.total_ms,
        upload_ms: null, download_ms: null})});
    stopLiveDetection(); stopWebcam(); preview = false; selectView("resultButton"); lastRequest = null;
    result.annotated_image_base64 = result.annotated_image_base64 || null; frozenResult = result; await renderResult(result);
  } catch (error) { notice(error.message, true); }
  finally { $("testButton").disabled = false; }
};
$("exportButton").onclick = () => {
  const keys = ["timestamp", "request_id", "source", "inference_location", "cloud_model_sha256", "cloud_roundtrip_ms", "status", "http_status", "error_code", "detection_count", "with_mask_count", "without_mask_count", "incorrect_mask_count", "inference_ms", "roundtrip_ms", "end_to_end_ms"];
  const csv = [keys.join(","), ...rows.map(row => keys.map(key => JSON.stringify(row[key] ?? "")).join(","))].join("\r\n");
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob(["\ufeff" + csv], {type: "text/csv;charset=utf-8"}));
  link.download = "latency_history.csv"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
};
window.addEventListener("resize", drawChart);
$("frame").addEventListener("error", () => {
  cameraPreviewError = true;
  if (preview) { $("frameSource").textContent = "CAMERA STREAM · LỖI"; }
  if (preview) notice("Không mở được camera stream. Kiểm tra WiFi/IP; nếu dashboard dùng HTTPS, mở Camera URL trong tab riêng để xem camera HTTP.", true);
});
$("frame").addEventListener("load", () => { cameraPreviewError = false; });
async function init() {
  try {
    const config = await (await fetch("/config")).json();
    configureDashboard(config);
    cloudTimeoutMs = (config.cloud_timeout_seconds || 90) * 2000 + 20000;
    apiBase = normalizeUrl(config.api_base_url || location.origin);
    cameraBase = config.camera_url ? normalizeUrl(config.camera_url) : "";
    try { const saved = localStorage.getItem("face_mask_camera_url"); if (saved) cameraBase = normalizeUrl(saved); } catch (_) {}
    if (config.api_key_required) { $("settingsPanel").hidden = false; notice("Nhập API key để xem kết quả.", true); }
  } catch (error) { notice("Dùng API cùng địa chỉ dashboard: " + error.message, true); }
  $("apiInput").value = apiBase; $("cameraInput").value = cameraBase; setCamera();
  await poll(); setInterval(poll, 2000);
  const updateClock = () => { $("clock").textContent = new Date().toLocaleString("vi-VN"); };
  updateClock(); setInterval(updateClock, 1000);
}
init();
