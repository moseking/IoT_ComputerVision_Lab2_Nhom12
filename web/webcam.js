"use strict";
let laptopStream = null, laptopBusy = false;
function showTelegramDelivery(result) {
  const delivery = result.telegram_delivery;
  $("telegramDelivery").textContent = delivery ? (delivery.state === "queued" ? "Đã đưa ảnh vào hàng đợi Telegram. Kiểm tra tin nhắn trong chat của bot." : "Chưa gửi Telegram. Kiểm tra cấu hình bot, chat ID hoặc hàng đợi đang bận.") : "";
}
function stopLaptopCamera() {
  if (laptopStream) laptopStream.getTracks().forEach(track => track.stop());
  laptopStream = null; $("webcamVideo").srcObject = null; $("webcamVideo").hidden = true;
  $("webcamStart").disabled = false; $("webcamCapture").disabled = true; $("webcamStop").disabled = true;
}
$("webcamStart").onclick = async () => {
  $("webcamStart").disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Mở dashboard bằng http://localhost:8000 hoặc http://127.0.0.1:8000 để dùng camera laptop.");
    stopLiveDetection();
    laptopStream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal:1280}, height:{ideal:720}}, audio:false});
    $("webcamVideo").srcObject = laptopStream; $("webcamVideo").hidden = false;
    await $("webcamVideo").play();
    $("webcamStart").disabled = true; $("webcamCapture").disabled = false; $("webcamStop").disabled = false;
    notice("Camera laptop đã mở. Bấm Chụp & nhận diện để xử lý một ảnh.");
  } catch (error) {
    stopLaptopCamera();
    notice(error.name === "NotAllowedError" ? "Chưa được cấp quyền camera. Cho phép camera trong trình duyệt rồi thử lại." : error.name === "NotFoundError" ? "Không tìm thấy camera laptop." : error.message, true);
  }
};
$("webcamStop").onclick = stopLaptopCamera;
$("webcamCapture").onclick = async () => {
  if (laptopBusy || !laptopStream) return;
  const video = $("webcamVideo");
  if (!video.videoWidth) return notice("Camera chưa có hình, thử lại sau.", true);
  laptopBusy = true; $("webcamCapture").disabled = true;
  const totalStart = performance.now();
  try {
    stopLiveDetection(); preview = false;
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video,0,0);
    const blob = await new Promise(resolve => canvas.toBlob(resolve,"image/jpeg",.9));
    if (!blob) throw new Error("Không tạo được ảnh từ camera.");
    const data = new FormData(); data.append("file",blob,"laptop.jpg"); data.append("source","dashboard");
    data.append("include_image","true"); data.append("inference_target",inferenceTarget);
    data.append("send_telegram",$("sendTelegram").checked ? "true" : "false");
    const started = performance.now();
    const response = await request("/predict",{method:"POST",body:data,timeout:inferenceTarget === "cloud" ? cloudTimeoutMs : 60000});
    const text = await response.text(); const received = performance.now(); const result = JSON.parse(text);
    attachClientTimings(result,received-started,received-totalStart);
    selectView("resultButton"); lastRequest = null; frozenResult = result;
    await renderResult(result); $("frameSource").textContent = "CAMERA LAPTOP"; showTelegramDelivery(result);
    try { await json("/metrics/report",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({request_id:result.request_id,report_token:result.report_token,capture_ms:null,roundtrip_ms:result.roundtrip_ms,total_ms:result.total_ms,upload_ms:null,download_ms:null})}); }
    catch (_) { notice("Đã nhận diện; chưa lưu được số đo thời gian phía trình duyệt.",true); }
  } catch (error) { notice(error.message,true); }
  finally { laptopBusy = false; $("webcamCapture").disabled = !laptopStream; }
};
window.addEventListener("pagehide",stopLaptopCamera);
