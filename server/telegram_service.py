"""Optional Telegram Bot API worker: bounded queue, real annotated bytes, no inference blocking."""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
import httpx
from .camera_client import CameraError, camera_endpoint, capture_async

logger = logging.getLogger("face_mask.telegram")


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool = False
    token: str = field(default="", repr=False)
    chat_id: str = field(default="", repr=False)
    cooldown: float = 30
    mode: str = "violations"
    metrics_wait: float = 3
    commands: bool = True
    camera_url: str = ""
    timeout: float = 10


class TelegramError(RuntimeError):
    pass


class TelegramService:
    def __init__(self, config, store, latest_snapshot, client=None, clock=time.monotonic):
        self.config, self.store, self.latest_snapshot = config, store, latest_snapshot
        self.client, self.clock = client, clock
        self._owns_client = client is None
        self._lock = threading.Lock()
        self._loop = None
        self._jobs = set()
        self._queue = asyncio.Queue(maxsize=2)
        self._last_reserved = None
        self._command_times = {}
        self._pending = 0
        self._verified = False
        self._state = "disabled" if not config.enabled else ("not_configured" if not config.token or not config.chat_id else "starting")
        self._error = None
        self._last_sent = None
        self._last_sent_request_id = None
        self._sent = self._skipped = 0

    def status(self):
        with self._lock:
            return {"enabled": self.config.enabled, "configured": bool(self.config.token and self.config.chat_id),
                    "state": self._state, "verified": self._verified, "last_error": self._error,
                    "last_sent": self._last_sent, "last_sent_request_id": self._last_sent_request_id,
                    "sent_count": self._sent, "skipped_count": self._skipped,
                    "queue_depth": self._pending, "cooldown_seconds": self.config.cooldown,
                    "alert_mode": self.config.mode, "commands_enabled": self.config.commands}

    def _set_error(self, exc):
        # Exception URLs/Telegram descriptions can contain tokens: never log str(exc).
        kind = type(exc).__name__
        with self._lock:
            self._state, self._error = "error", kind
        logger.warning("Telegram operation failed (%s); inference continues", kind)

    async def start(self):
        if self._state in {"disabled", "not_configured"}:
            return
        self._loop = asyncio.get_running_loop()
        # httpx/httpcore DEBUG URLs contain the bot token.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.config.timeout, trust_env=False, follow_redirects=False)
        self._jobs.add(asyncio.create_task(self._worker()))
        if self.config.commands:
            self._jobs.add(asyncio.create_task(self._poll_commands()))
        # Startup does not wait on Telegram connectivity.

    async def stop(self):
        for task in self._jobs:
            task.cancel()
        await asyncio.gather(*self._jobs, return_exceptions=True)
        self._jobs.clear()
        self._loop = None
        with self._lock:
            self._pending = 0
            self._verified = False
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        if self._owns_client and self.client:
            await self.client.aclose()

    async def _api(self, method, data=None, files=None):
        response = await self.client.post(f"https://api.telegram.org/bot{self.config.token}/{method}",
                                         data=data or {}, files=files)
        try:
            body = response.json()
        except ValueError:
            raise TelegramError("Invalid Bot API response") from None
        if response.status_code != 200 or not isinstance(body, dict) or not body.get("ok"):
            raise TelegramError(f"Bot API HTTP {response.status_code}")
        return body.get("result")

    def submit(self, result, annotated, *, manual=False):
        # Called from FastAPI's inference thread. Reserve before scheduling to bound memory.
        if not self.config.enabled or not self.config.token or not self.config.chat_id or self._loop is None:
            return False
        if not manual and self.config.mode == "violations" and not any(d["class_id"] in (1, 2) for d in result["detections"]):
            return False
        with self._lock:
            now = self.clock()
            if self._pending >= 2 or (not manual and self._last_reserved is not None and now-self._last_reserved < self.config.cooldown):
                self._skipped += 1
                return False
            self._last_reserved = now
            self._pending += 1
        event = ({k: v for k, v in result.items() if k not in {"report_token", "annotated_image_base64"}}, annotated)
        def enqueue():
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                with self._lock:
                    self._pending -= 1
                    self._skipped += 1
        try:
            self._loop.call_soon_threadsafe(enqueue)
        except RuntimeError as exc:
            with self._lock:
                self._pending -= 1
            self._set_error(exc)
            return False
        return True

    async def enriched(self, result):
        deadline = self.clock() + self.config.metrics_wait
        # Wait in the background for the client report; no /predict delay.
        while True:
            row = await asyncio.to_thread(self.store.get, result["request_id"])
            if row:
                result = {**result, **{k: row.get(k) for k in ("capture_ms", "roundtrip_ms", "total_ms", "end_to_end_ms")}}
            if (row and row.get("reported")) or self.clock() >= deadline:
                return result
            await asyncio.sleep(0.1)

    @staticmethod
    def caption(result):
        detections = result["detections"]
        labels = ["Đeo đúng", "Không đeo", "Đeo sai"]
        counts = [sum(d["class_id"] == cls for d in detections) for cls in range(3)]
        def ms(key):
            return "chưa đo" if result.get(key) is None else f"{result[key]:.1f} ms"
        lines = ["ESP32-CAM • YOLO Face Mask", f"Sự kiện: {result['timestamp']}",
                 f"Request: {result['request_id']}", f"Nguồn: {result['source']}",
                 f"Khuôn mặt: {len(detections)} | Đúng {counts[0]} / Không {counts[1]} / Sai {counts[2]}",
                 f"Inference: {ms('inference_ms')}", f"RTT: {ms('roundtrip_ms')} | Toàn trình: {ms('total_ms')}"]
        # Leave headroom below Bot API's 1024-character caption limit.
        for index, detection in enumerate(detections, 1):
            line = f"#{index}: {labels[detection['class_id']]} {detection['confidence']:.1%}"
            if len("\n".join(lines + [line])) > 920:
                lines.append(f"... {len(detections)-index+1} mặt còn lại xem dashboard.")
                break
            lines.append(line)
        return "\n".join(lines)

    async def send_result(self, result, annotated, wait_metrics=True):
        if wait_metrics:
            result = await self.enriched(result)
        # Uses the exact annotation generated for this request, not a later camera capture.
        await self._api("sendPhoto", {"chat_id": self.config.chat_id, "caption": self.caption(result)},
                        {"photo": ("annotated.jpg", annotated, "image/jpeg")})
        with self._lock:
            self._sent += 1
            self._last_sent = datetime.now(timezone.utc).isoformat()
            self._last_sent_request_id = result["request_id"]
            self._state, self._error = "ready", None

    async def _worker(self):
        while True:
            if not self._verified:
                try:
                    await self._api("getMe")
                    with self._lock:
                        self._verified, self._state, self._error = True, "ready", None
                except Exception as exc:
                    self._set_error(exc)
                    await asyncio.sleep(max(5, self.config.cooldown))
                    continue
            result, annotated = await self._queue.get()
            try:
                await self.send_result(result, annotated)
            except Exception as exc:
                self._set_error(exc)
            finally:
                self._queue.task_done()
                with self._lock:
                    self._pending -= 1

    async def handle_command(self, message):
        if str(message.get("chat", {}).get("id", "")) != self.config.chat_id:
            return False
        text = message.get("text", "").strip()
        if not text.startswith("/"):
            return False
        command = text.split()[0].split("@")[0].lower()
        if command not in {"/start", "/photo", "/flash", "/result"}:
            return False
        now = self.clock()
        previous = self._command_times.get(self.config.chat_id)
        if previous is not None and now-previous < 2:
            return False
        self._command_times[self.config.chat_id] = now
        async def say(value):
            await self._api("sendMessage", {"chat_id": self.config.chat_id, "text": value})
        try:
            if command == "/start":
                await say("Lab 3: /photo ảnh camera gốc; /result ảnh YOLO gần nhất; /flash [on|off] điều khiển LED. Backend xử lý Telegram, ESP32 chỉ phục vụ camera/flash.")
            elif command == "/photo":
                data, _ = await capture_async(self.client, self.config.camera_url)
                await self._api("sendPhoto", {"chat_id": self.config.chat_id,
                               "caption": "Ảnh camera gốc • " + datetime.now(timezone.utc).isoformat() + " • chưa chạy inference"},
                               {"photo": ("camera.jpg", data, "image/jpeg")})
            elif command == "/flash":
                argument = text.split()[1].lower() if len(text.split()) > 1 else ""
                if argument not in {"", "on", "off"}:
                    await say("Dùng /flash, /flash on hoặc /flash off")
                else:
                    params = {"on": "1" if argument == "on" else "0"} if argument else None
                    response = await self.client.get(camera_endpoint(self.config.camera_url, "/flash"), params=params)
                    if response.status_code != 200:
                        raise CameraError(f"Flash HTTP {response.status_code}")
                    await say("Flash ON" if response.json()["flash"] else "Flash OFF")
            elif command == "/result":
                result, image = self.latest_snapshot()
                if result is None or image is None:
                    await say("Chưa có kết quả YOLO. Chạy checkpoint đã train và gửi ảnh trước.")
                else:
                    await self.send_result(result, image)
        except (CameraError, httpx.RequestError):
            await say("Không truy cập được camera: kiểm tra ESP32_CAMERA_URL, WiFi, port 80 và /capture.")
        return True

    async def _poll_commands(self):
        offset, initial = None, True
        while True:
            try:
                if not self._verified:
                    await asyncio.sleep(1)
                    continue
                data = {"timeout": "10", "allowed_updates": '["message"]'}
                # Discard pre-start backlog so old /flash commands cannot replay after restart.
                data["offset"] = "-1" if initial else str(offset or 0)
                updates = await self._api("getUpdates", data)
                for update in updates or []:
                    offset = max(offset or 0, int(update["update_id"])+1)
                    if not initial:
                        await self.handle_command(update.get("message", {}))
                initial = False
                await asyncio.sleep(0.2)
            except Exception as exc:
                self._set_error(exc)
                await asyncio.sleep(5)
