"""FastAPI service and dashboard; a missing model produces explicit 503 responses."""
import asyncio
import base64
import hashlib
import json
import logging
import secrets
import threading
import time
import uuid
import io
import httpx
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from pydantic import BaseModel, Field, StrictBool, model_validator

from .config import ROOT, settings
from .inference import CLASSES, InvalidImage, MaskDetectionModel
from .latency_store import LatencyStore, recognition_counts
from .cloud_client import CloudConfig, CloudInferenceClient, CloudError
from . import camera_control, camera_client
from .telegram_service import TelegramConfig, TelegramService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("face_mask")
latest_lock = threading.Lock()
latest_result = None
latest_image = None
store = None
telegram = None
cloud = CloudInferenceClient(CloudConfig(settings.cloud_url, settings.cloud_key, settings.cloud_sha256, settings.cloud_timeout))
model = MaskDetectionModel(settings.model_path, settings.confidence, settings.iou,
                           settings.image_size, settings.device, settings.max_image_pixels)


@asynccontextmanager
async def lifespan(app):
    global store, telegram, latest_result, latest_image
    store = LatencyStore(settings.db_path)
    try:
        model.load_model()
        logger.info("model_ready path=%s device=%s warmup=%.3fms", model.path, model.device, model.warmup_ms)
    except Exception as exc:
        model.load_error = str(exc)
        logger.warning("model_unavailable: %s", exc)
    with latest_lock:
        latest_result, latest_image = None, None
    telegram = TelegramService(TelegramConfig(
        enabled=settings.telegram_enabled, token=settings.telegram_token, chat_id=settings.telegram_chat_id,
        cooldown=settings.telegram_cooldown, mode=settings.telegram_mode, metrics_wait=settings.telegram_metrics_wait,
        commands=settings.telegram_commands, camera_url=settings.camera_url, timeout=settings.telegram_timeout),
        store, latest_snapshot)
    await telegram.start()
    try:
        yield
    finally:
        await telegram.stop()
        cloud.close()


def authorize(x_api_key: Annotated[str | None, Header()] = None):
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(401, "Missing or invalid X-API-Key")


app = FastAPI(title="Face Mask Detection – IoT Lab 3", version="1.0.0", lifespan=lifespan)



class UploadLimitMiddleware:
    """Bound the full body before multipart parsing, including chunked uploads."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        limit = settings.max_upload_bytes + 65536 if scope["path"] in {"/predict", "/predict/image"} else 65536
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if declared > limit:
            return await JSONResponse({"detail": "Request exceeds upload limit"}, 413)(scope, receive, send)
        chunks, size = [], 0
        deadline = time.perf_counter() + settings.upload_timeout_seconds
        while True:
            try:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                message = await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                return await JSONResponse({"detail": "Request body upload timed out"}, 408)(scope, receive, send)
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            size += len(part)
            if size > limit:
                return await JSONResponse({"detail": "Request exceeds upload limit"}, 413)(scope, receive, send)
            chunks.append(part)
            if not message.get("more_body", False):
                break
        replayed = False
        async def bounded_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, bounded_receive, send)


app.add_middleware(UploadLimitMiddleware)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/health")
def health():
    return {"status": "ok" if model.ready else "model_unavailable", "model_loaded": model.ready,
            "detail": model.load_error, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/config")
def public_config():
    # No secrets are exposed to the browser.
    return {"api_base_url": settings.api_base_url, "camera_url": settings.camera_url,
            "send_interval_ms": settings.send_interval_ms, "api_key_required": bool(settings.api_key),
            "cloud_configured": cloud.configured, "cloud_timeout_seconds": settings.cloud_timeout}


@app.get("/model/info", dependencies=[Depends(authorize)])
def model_info():
    candidates = [model.path.parent / "model_metadata.json", model.path.parent.parent / "model_metadata.json"]
    path = next((p for p in candidates if p.is_file()), None)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError):
        metadata = None
    return {"loaded": model.ready, "model_name": "YOLOv8n Face Mask Detection", "checkpoint": model.path.name, "classes": CLASSES, "confidence_threshold": settings.confidence, "nms_class_agnostic": True, "nms_iou": settings.iou,
            "image_size": settings.image_size, "device": model.device, "warmup_ms": model.warmup_ms, "metadata": metadata}


@app.get("/cloud/status", dependencies=[Depends(authorize)])
def cloud_status():
    return cloud.status()


@app.get("/cloud/model/info", dependencies=[Depends(authorize)])
def cloud_model_info():
    if not cloud.configured:raise HTTPException(503, "Cloud chưa cấu hình")
    try:return cloud.verify()
    except CloudError as exc:raise HTTPException(502, str(exc)) from None


def run_prediction(file, source, capture_ms, include_image, response_kind, *, manual_telegram=False, inference_target="local"):
    global latest_result, latest_image
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    image_bytes = b""
    def failed(status, code):
        store.record_failure(request_id, source, status, code, len(image_bytes) or None, inference_target)
    try:
        if inference_target == "cloud" and not cloud.configured:
            raise HTTPException(503, "Cloud chưa cấu hình URL/key/model SHA-256")
        if inference_target == "local" and not model.ready:
            raise HTTPException(503, "Model unavailable. Train the project, then restart the server.")
        if file.content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
            raise HTTPException(415, "Upload a JPEG or PNG image")
        image_bytes = file.file.read(settings.max_upload_bytes + 1)
        if len(image_bytes) > settings.max_upload_bytes:
            raise HTTPException(413, "Image exceeds MAX_UPLOAD_BYTES")
        image = model.decode(image_bytes)
        remote = None
        if inference_target == "cloud":
            remote, annotated = cloud.predict(image_bytes, image.shape[1], image.shape[0])
            prediction = {k: remote[k] for k in ("detections", "inference_ms", "queue_ms", "inference_definition")}
        else:
            prediction = model.predict(image)
            annotated = model.annotate(image, prediction["detections"])
        gateway_processing = round((time.perf_counter() - started) * 1000, 3)
        processing = remote["server_processing_ms"] if remote else gateway_processing
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
                  "source": source, "detection_count": len(prediction["detections"]),
                  "inference_ms": prediction["inference_ms"], "server_processing_ms": processing,
                  "capture_ms": capture_ms, "image_size_bytes": len(image_bytes),
                  "inference_location": inference_target,
                  "cloud_roundtrip_ms": remote["cloud_roundtrip_ms"] if remote else None,
                  "cloud_model_sha256": remote["model_sha256"] if remote else None,
                  **recognition_counts(prediction["detections"])}
        token = store.insert(record)
        result = {"success": True, **record, **prediction, "detections": prediction["detections"],
                  "class_counts": {"with_mask": record["with_mask_count"], "without_mask": record["without_mask_count"],
                                   "mask_weared_incorrect": record["incorrect_mask_count"]},
                  "model_name": "YOLOv8n Face Mask Detection", "checkpoint": "face_mask_v7.onnx" if remote else model.path.name,
                  "roundtrip_ms": None, "total_ms": None, "end_to_end_ms": None, "upload_ms": None, "download_ms": None,
                  "report_token": token, "width": image.shape[1], "height": image.shape[0],
                  "annotated_image_url": f"/results/latest/image?request_id={request_id}",
                  "server_processing_definition": "cloud_file_read_decode_inference_annotation; excludes upload_serialization" if remote else "file_read_decode_inference_annotation; excludes upload_DB_serialization",
                  "gateway_processing_ms": gateway_processing,
                  "runtime": remote["runtime"] if remote else "pytorch",
                  "cloud_request_id": remote["request_id"] if remote else None}
        with latest_lock:
            latest_result = {k: v for k, v in result.items() if k != "report_token"}
            latest_image = annotated
        if manual_telegram:
            result["telegram_delivery"] = {"requested": True, "state": "unavailable"}
        if telegram:
            try:
                accepted = telegram.submit(result, annotated, manual=True) if manual_telegram else telegram.submit(result, annotated)
                if manual_telegram:
                    result["telegram_delivery"]["state"] = "queued" if accepted else "not_queued"
            except Exception as exc:
                logger.warning("Telegram enqueue failed (%s); prediction still succeeds", type(exc).__name__)
        logger.info("request=%s source=%s bytes=%s detections=%s inference=%.3fms processing=%.3fms",
                    request_id, source, len(image_bytes), len(prediction["detections"]),
                    prediction["inference_ms"], processing)
        headers = {"X-Request-ID": request_id,
                   "Server-Timing": f"inference;dur={prediction['inference_ms']}, processing;dur={processing}"}
        if response_kind == "image":
            headers.update({"X-Report-Token": token, "X-Inference-Ms": str(prediction["inference_ms"]),
                            "X-Server-Processing-Ms": str(processing), "X-Detection-Count": str(record["detection_count"])})
            return Response(annotated, media_type="image/jpeg", headers=headers)
        if include_image:
            result["annotated_image_base64"] = base64.b64encode(annotated).decode("ascii")
        return JSONResponse(result, headers=headers)
    except CloudError as exc:
        failed(502, "cloud_inference_error")
        raise HTTPException(502, str(exc), headers={"X-Request-ID": request_id}) from None
    except InvalidImage as exc:
        failed(400, "invalid_image")
        raise HTTPException(400, str(exc), headers={"X-Request-ID": request_id}) from exc
    except HTTPException as exc:
        failed(exc.status_code, f"http_{exc.status_code}")
        exc.headers = {**(exc.headers or {}), "X-Request-ID": request_id}
        raise
    except Exception as exc:
        failed(500, "inference_error")
        logger.error("inference_error request=%s type=%s", request_id, type(exc).__name__)
        raise HTTPException(500, f"Inference failed; request_id={request_id}", headers={"X-Request-ID": request_id}) from exc
    finally:
        file.file.close()


Source = Literal["esp32", "test_client", "dashboard"]


class CameraRequest(BaseModel):
    camera_url: str = Field(min_length=1, max_length=256)


class CameraPredictRequest(CameraRequest):
    send_telegram: StrictBool = False
    inference_target: Literal["local", "cloud"] = "local"


class FlashRequest(CameraRequest):
    enabled: StrictBool


def control_camera_base(value):
    try:
        return camera_control.lan_camera_url(value)
    except camera_client.CameraError as exc:
        raise HTTPException(422,str(exc)) from exc


@app.post('/camera/flash', dependencies=[Depends(authorize)])
def camera_flash(command: FlashRequest):
    base=control_camera_base(command.camera_url)
    try:
        with httpx.Client(timeout=httpx.Timeout(5,connect=3),trust_env=False,follow_redirects=False) as client:
            return camera_control.flash(client,base,command.enabled)
    except camera_client.CameraError as exc:
        raise HTTPException(502,str(exc)) from exc


@app.post('/camera/predict', dependencies=[Depends(authorize)])
def predict_camera(command: CameraPredictRequest):
    base=control_camera_base(command.camera_url)
    if command.inference_target == "cloud" and not cloud.configured:raise HTTPException(503, "Cloud chưa cấu hình")
    if command.inference_target == "local" and not model.ready:raise HTTPException(503,'Model unavailable')
    started=time.perf_counter()
    try:
        with httpx.Client(timeout=httpx.Timeout(8,connect=3),trust_env=False,follow_redirects=False) as client:
            image_bytes,original_shape=camera_client.capture(client,base,max_bytes=settings.max_upload_bytes)
    except camera_client.CameraError as exc:
        raise HTTPException(502,str(exc)) from exc
    fetch_ms=(time.perf_counter()-started)*1000
    # Server capture+download includes network; ESP32 acquisition timing is unknown.
    upload=UploadFile(file=io.BytesIO(image_bytes),filename='esp32_capture.jpg',
        headers=Headers({'content-type':'image/jpeg'}))
    response=run_prediction(upload,'camera_pull',None,True,'json',manual_telegram=command.send_telegram, inference_target=command.inference_target)
    payload=json.loads(response.body)
    payload['camera_fetch_ms']=round(fetch_ms,3)
    payload['camera_fetch_definition']='server HTTP /capture and download; includes network, not ESP32 capture_ms'
    payload['capture_mode']='server_pull'
    # Exact fetched bytes from this request; never a recapture or annotated image.
    payload['original_image_base64']=base64.b64encode(image_bytes).decode('ascii')
    payload['original_image_sha256']=hashlib.sha256(image_bytes).hexdigest()
    payload['original_image_content_type']='image/jpeg' if original_shape[2]=='JPEG' else 'image/png'
    payload['original_image_filename']='cam32_'+payload['request_id']+('.jpg' if original_shape[2]=='JPEG' else '.png')
    with latest_lock:
        if latest_result and latest_result['request_id']==payload['request_id']:
            latest_result.update({k:payload[k] for k in ['camera_fetch_ms','camera_fetch_definition','capture_mode']})
    headers={k:v for k,v in response.headers.items() if k.lower() not in {'content-length','content-type'}}
    return JSONResponse(payload,headers=headers)


@app.post("/predict", dependencies=[Depends(authorize)])
def predict(file: Annotated[UploadFile, File()], source: Annotated[Source, Form()] = "test_client",
            capture_ms: Annotated[float | None, Form(ge=0, le=300000)] = None,
            include_image: Annotated[bool, Form()] = False,
            send_telegram: Annotated[bool, Form()] = False,
            inference_target: Annotated[Literal["local", "cloud"], Form()] = "local"):
    return run_prediction(file, source, capture_ms, include_image, "json", manual_telegram=send_telegram, inference_target=inference_target)


@app.post("/predict/image", dependencies=[Depends(authorize)])
def predict_image(file: Annotated[UploadFile, File()], source: Annotated[Source, Form()] = "test_client",
                  capture_ms: Annotated[float | None, Form(ge=0, le=300000)] = None,
                  inference_target: Annotated[Literal["local", "cloud"], Form()] = "local"):
    return run_prediction(file, source, capture_ms, False, "image", inference_target=inference_target)


class ClientMetrics(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    report_token: str = Field(min_length=1, max_length=128)
    capture_ms: float | None = Field(default=None, ge=0, le=300000, allow_inf_nan=False)
    roundtrip_ms: float = Field(ge=0, le=300000, allow_inf_nan=False)
    total_ms: float = Field(ge=0, le=300000, allow_inf_nan=False)
    upload_ms: float | None = Field(default=None, ge=0, le=300000, allow_inf_nan=False)
    download_ms: float | None = Field(default=None, ge=0, le=300000, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self):
        if self.total_ms + 2 < self.roundtrip_ms + (self.capture_ms or 0):
            raise ValueError("total_ms must cover capture and full roundtrip")
        if self.download_ms is not None and self.download_ms > self.roundtrip_ms:
            raise ValueError("download_ms must be <= roundtrip_ms")
        return self


@app.post("/metrics/report", dependencies=[Depends(authorize)])
def report_latency(metrics: ClientMetrics):
    try:
        row = store.report(metrics.request_id, metrics.report_token, metrics.model_dump())
        return {"success": True, "measurement": row}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@app.get("/metrics/history", dependencies=[Depends(authorize)])
def history(limit: Annotated[int, Query(ge=1, le=500)] = 50,
            offset: Annotated[int, Query(ge=0)] = 0):
    return {"items": store.history(limit, offset)}


@app.get("/metrics/summary", dependencies=[Depends(authorize)])
def summary():
    return store.summary()


@app.get("/device/status", dependencies=[Depends(authorize)])
def device_status():
    timestamp = store.latest_device_timestamp()
    age = ((datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
           if timestamp else None)
    threshold = max(15, settings.send_interval_ms / 1000 * 3)
    return {"status": "online" if age is not None and age < threshold else "offline_or_not_seen",
            "last_seen": timestamp, "age_seconds": age, "camera_url": settings.camera_url,
            "basis": "recent successful camera predictions tagged source=esp32 or camera_pull",
            "last_failure": store.latest_failure("esp32")}


@app.get("/telegram/status", dependencies=[Depends(authorize)])
def telegram_status():
    return telegram.status()


def latest_snapshot():
    """Copy one matching result/image pair before looking up its client metrics."""
    with latest_lock:
        result = latest_result.copy() if latest_result else None
        image = latest_image
    if result:
        row = store.get(result["request_id"])
        if row:
            for key in ("roundtrip_ms", "total_ms", "end_to_end_ms", "capture_ms", "upload_ms",
                        "download_ms", "network_overhead_estimated_ms"):
                result[key] = row[key]
    return result, image


@app.get("/results/latest", dependencies=[Depends(authorize)])
def latest():
    result, _ = latest_snapshot()
    return {"result": result}


@app.get("/results/latest/image", dependencies=[Depends(authorize)])
def result_image(request_id: str | None = None):
    with latest_lock:
        if latest_image is None:
            raise HTTPException(404, "No image has been inferred yet")
        if request_id and request_id != latest_result["request_id"]:
            raise HTTPException(409, "A newer frame is available; refresh the latest result")
        return Response(latest_image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=settings.host, port=settings.port, workers=1)
