"use strict";
// Capture the browser's webcam only after an explicit user action.
class WebcamCapture {
  constructor(video, options = {}) {
    this.video = video;
    this.mediaDevices = options.mediaDevices ?? globalThis.navigator?.mediaDevices;
    this.secure = options.secureContext ?? globalThis.isSecureContext;
    this.makeCanvas = options.makeCanvas ?? (() => document.createElement("canvas"));
    this.now = options.now ?? (() => performance.now());
    this.stream = null; this.pending = null; this.generation = 0;
  }
  get starting() { return this.pending !== null; }
  get ready() { return !!this.stream && this.stream.getVideoTracks().some(t => t.readyState === "live"); }
  error(message) { const error = new Error(message); error.name = "WebcamError"; return error; }
  cancelled() { const error = new Error("Đã dừng yêu cầu webcam."); error.name = "AbortError"; return error; }
  stop() {
    this.generation++;
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    this.stream = null; this.pending = null;
    this.video.pause(); this.video.srcObject = null;
  }
  start() {
    if (this.ready) return Promise.resolve(this.stream);
    if (this.pending) return this.pending;
    if (!this.secure || !this.mediaDevices?.getUserMedia) {
      return Promise.reject(this.error("Mở dashboard bằng http://127.0.0.1:8000 hoặc localhost để dùng webcam; địa chỉ IP qua HTTP không hỗ trợ quyền camera."));
    }
    const generation = ++this.generation;
    const task = this.open(generation);
    this.pending = task;
    task.then(() => { if (this.pending === task) this.pending = null; }, () => { if (this.pending === task) this.pending = null; });
    return task;
  }
  async open(generation) {
    let stream;
    try {
      stream = await this.mediaDevices.getUserMedia({audio: false, video: {width: {ideal: 640}, height: {ideal: 480}, facingMode: "user"}});
      if (generation !== this.generation) throw this.cancelled();
      this.stream = stream; this.video.srcObject = stream;
      await this.video.play();
      if (!this.video.videoWidth || !this.video.videoHeight) {
        await new Promise((resolve, reject) => {
          const finish = error => {
            clearTimeout(timer); this.video.removeEventListener("loadeddata", loaded);
            this.video.removeEventListener("error", failed);
            error ? reject(error) : resolve();
          };
          const loaded = () => finish();
          const failed = () => finish(this.error("Không đọc được frame webcam."));
          const timer = setTimeout(() => finish(this.error("Webcam chưa có frame. Đóng app khác đang dùng camera rồi thử lại.")), 8000);
          this.video.addEventListener("loadeddata", loaded, {once: true});
          this.video.addEventListener("error", failed, {once: true});
        });
      }
      if (generation !== this.generation) throw this.cancelled();
      if (!this.ready || !this.video.videoWidth || !this.video.videoHeight) throw this.error("Webcam đã dừng hoặc chưa có ảnh.");
      return stream;
    } catch (error) {
      if (generation !== this.generation) { if (stream) stream.getTracks().forEach(t => t.stop()); throw this.cancelled(); }
      this.stop();
      if (error.name === "WebcamError") throw error;
      const messages = {
        NotAllowedError: "Chưa được cấp quyền camera. Chọn Cho phép camera trong trình duyệt rồi bật webcam lại.",
        NotFoundError: "Không tìm thấy webcam. Kiểm tra camera laptop/USB và công tắc che camera.",
        NotReadableError: "Webcam đang bận hoặc bị Windows chặn. Đóng Camera/Teams và kiểm tra quyền camera của trình duyệt.",
      };
      throw this.error(messages[error.name] || "Không mở được webcam: " + error.message);
    }
  }
  async capture() {
    await this.start();
    const generation = this.generation, started = this.now();
    const canvas = this.makeCanvas();
    const scale = Math.min(1, 1280 / Math.max(this.video.videoWidth, this.video.videoHeight));
    canvas.width = Math.max(1, Math.round(this.video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(this.video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) throw this.error("Trình duyệt không tạo được ảnh webcam.");
    context.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.85));
    if (generation !== this.generation || !this.ready) throw this.cancelled();
    if (!blob || blob.type !== "image/jpeg" || !blob.size) throw this.error("Không mã hóa được ảnh webcam thành JPEG.");
    return {blob, capture_ms: Math.max(0, this.now() - started), width: canvas.width, height: canvas.height};
  }
}
if (typeof module !== "undefined") module.exports = {WebcamCapture};
else globalThis.WebcamCapture = WebcamCapture;
