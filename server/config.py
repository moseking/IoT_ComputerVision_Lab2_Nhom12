"""Resolve environment configuration relative to the project, independent of cwd."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".runtime/ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))
os.environ["YOLO_AUTOINSTALL"] = "false"
load_dotenv(ROOT / ".env")


def path_setting(name, default):
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    model_path: Path = path_setting("MODEL_PATH", "models/face_mask_best.pt")
    db_path: Path = path_setting("LATENCY_DB_PATH", "data/latency.db")
    confidence: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    iou: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "640"))
    device: str = os.getenv("DEVICE", "auto")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "5242880"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "12000000"))
    camera_url: str = os.getenv("ESP32_CAMERA_URL", "").rstrip("/")
    send_interval_ms: int = int(os.getenv("SEND_INTERVAL_MS", "2000"))
    upload_timeout_seconds: float = float(os.getenv("UPLOAD_TIMEOUT_SECONDS", "30"))
    telegram_enabled: bool = os.getenv("TELEGRAM_ENABLED", "false").lower() in {"1", "true", "yes"}
    telegram_token: str = field(default=os.getenv("TELEGRAM_BOT_TOKEN", ""), repr=False)
    telegram_chat_id: str = field(default=os.getenv("TELEGRAM_CHAT_ID", ""), repr=False)
    telegram_cooldown: float = float(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "30"))
    telegram_mode: str = os.getenv("TELEGRAM_ALERT_MODE", "violations")
    telegram_metrics_wait: float = float(os.getenv("TELEGRAM_METRICS_WAIT_SECONDS", "3"))
    telegram_commands: bool = os.getenv("TELEGRAM_COMMANDS_ENABLED", "true").lower() in {"1", "true", "yes"}
    telegram_timeout: float = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "15"))
    cloud_url: str = os.getenv("CLOUD_API_URL", "").rstrip("/")
    cloud_key: str = field(default=os.getenv("CLOUD_API_KEY", ""), repr=False)
    cloud_sha256: str = os.getenv("CLOUD_MODEL_SHA256", "")
    cloud_timeout: float = float(os.getenv("CLOUD_TIMEOUT_SECONDS", "90"))
    api_key: str = field(default=os.getenv("API_KEY", ""), repr=False)
    cors_origins: tuple = tuple(x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip())


settings = Settings()
if not 0 <= settings.confidence <= 1 or not 0 <= settings.iou <= 1:
    raise ValueError("CONFIDENCE_THRESHOLD/IOU_THRESHOLD must be in [0,1]")
if min(settings.max_upload_bytes, settings.max_image_pixels, settings.image_size) <= 0:
    raise ValueError("Upload/image limits must be positive")

if settings.telegram_mode not in {"all", "violations"}:
    raise ValueError("TELEGRAM_ALERT_MODE must be all or violations")
if min(settings.telegram_cooldown, settings.telegram_metrics_wait) < 0 or min(settings.telegram_timeout, settings.upload_timeout_seconds) <= 0:
    raise ValueError("Invalid Telegram/upload timeout or cooldown")
