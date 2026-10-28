"""Thread-safe real YOLO detection. No fallback predictions or implicit downloads."""
import io
import threading
import time
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

CLASSES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
_IMAGE_OPEN = Image.open  # Capture original decoder before Ultralytics monkey-patches PIL.
LABELS = {0: "WITH MASK", 1: "WITHOUT MASK", 2: "INCORRECT MASK"}
COLORS = {0: (90, 195, 40), 1: (70, 70, 230), 2: (0, 180, 245)}


class InvalidImage(ValueError):
    pass


class MaskDetectionModel:
    def __init__(self, path, confidence=0.35, iou=0.45, image_size=640, device="auto", max_pixels=12000000):
        self.path = Path(path)
        self.confidence, self.iou = confidence, iou
        self.image_size, self.device, self.max_pixels = image_size, device, max_pixels
        self._model = None
        self._lock = threading.Lock()
        self.load_error = None
        self.warmup_ms = None

    @property
    def ready(self):
        return self._model is not None

    def load_model(self):
        with self._lock:
            if self._model is not None:
                return
            if not self.path.is_file():
                raise FileNotFoundError(f"Trained checkpoint missing: {self.path}. Run scripts/train.py.")
            import torch
            from ultralytics import YOLO
            self.device = ("0" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            model = YOLO(str(self.path))
            actual = {int(k): v for k, v in model.names.items()}
            if actual != CLASSES:
                raise ValueError(f"Checkpoint classes mismatch: {actual}; need {CLASSES}")
            model.to("cuda:" + self.device if self.device.isdigit() else self.device)
            # Initialize predictor/CUDA before reporting ready. This synthetic frame
            # is discarded: never persisted, rendered, counted or sent to Telegram.
            started = time.perf_counter()
            model.predict(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8),
                          conf=self.confidence, iou=self.iou, imgsz=self.image_size,
                          device=self.device, verbose=False, agnostic_nms=True)
            self.warmup_ms = round((time.perf_counter() - started) * 1000, 3)
            self._model = model

    def decode(self, image_bytes):
        if not image_bytes:
            raise InvalidImage("Empty image")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with _IMAGE_OPEN(io.BytesIO(image_bytes)) as image:
                    if image.format not in {"JPEG", "PNG"}:
                        raise InvalidImage("Only JPEG/PNG are accepted")
                    if image.width * image.height > self.max_pixels:
                        raise InvalidImage("Image exceeds MAX_IMAGE_PIXELS")
                    image.load()
                    rgb = np.asarray(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InvalidImage(f"Cannot decode image: {exc}") from exc
        # Preserve raw image orientation: boxes and annotated image use identical pixels.
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def predict_bytes(self, image_bytes):
        return self.predict(self.decode(image_bytes))

    def predict(self, image):
        if not self.ready:
            raise RuntimeError("Model is not ready")
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImage("Expected PIL RGB image or OpenCV BGR image")
        if image.shape[0] * image.shape[1] > self.max_pixels:
            raise InvalidImage("Image exceeds MAX_IMAGE_PIXELS")
        queued_at = time.perf_counter()
        with self._lock:
            acquired_at = time.perf_counter()
            import torch
            model_device = next(self._model.model.parameters()).device
            cuda = model_device.type == "cuda"
            if cuda and torch.cuda.is_available():
                torch.cuda.synchronize(model_device)
            start = time.perf_counter()
            result = self._model.predict(image, conf=self.confidence, iou=self.iou,
                                         imgsz=self.image_size, device=self.device, verbose=False,
                                         agnostic_nms=True)[0]
            if cuda and torch.cuda.is_available():
                torch.cuda.synchronize(model_device)
            # Inference includes preprocessing, model forward and NMS/postprocessing.
            elapsed = (time.perf_counter() - start) * 1000
            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                coords = box.xyxy[0].detach().cpu().tolist()
                x1, y1, x2, y2 = coords
                detections.append({"class_id": class_id, "class_name": CLASSES[class_id],
                                   "confidence": round(float(box.conf.item()), 6),
                                   "bbox": dict(zip(("x1", "y1", "x2", "y2"), (round(v, 2) for v in coords)))})
        return {"detections": detections, "inference_ms": round(elapsed, 3),
                "queue_ms": round((acquired_at - queued_at) * 1000, 3),
                "inference_definition": "wall_time_preprocess_forward_NMS; excludes queue/decode/annotation"}

    @staticmethod
    def annotate(image, detections):
        output = image.copy()
        for detection in detections:
            class_id, bbox = detection["class_id"], detection["bbox"]
            x1, y1, x2, y2 = (round(bbox[k]) for k in ("x1", "y1", "x2", "y2"))
            label = f"{LABELS[class_id]} {detection['confidence']:.0%}"
            cv2.rectangle(output, (x1, y1), (x2, y2), COLORS[class_id], 2)
            cv2.putText(output, label, (max(0, x1), max(15, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Could not encode annotated image")
        return encoded.tobytes()
