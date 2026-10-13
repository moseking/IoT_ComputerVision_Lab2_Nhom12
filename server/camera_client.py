"""Bounded camera HTTP access shared by capture CLI and Telegram commands."""
import io
from urllib.parse import urlsplit, urlunsplit
import httpx
from PIL import Image, UnidentifiedImageError

_IMAGE_OPEN = Image.open


class CameraError(RuntimeError):
    pass


def camera_endpoint(base, path):
    parts = urlsplit(base.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise CameraError("Configure ESP32_CAMERA_URL as http(s)://camera-IP without credentials")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def validate_capture(data, max_pixels=12000000):
    try:
        with _IMAGE_OPEN(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise CameraError("Camera returned neither JPEG nor PNG")
            if image.width * image.height > max_pixels:
                raise CameraError("Camera image exceeds pixel limit")
            image.load()
            return image.width, image.height, image.format
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CameraError("Camera returned an invalid image") from exc


def capture(client, base, max_bytes=5242880):
    try:
        with client.stream("GET", camera_endpoint(base, "/capture")) as response:
            if response.status_code != 200:
                raise CameraError(f"Camera /capture HTTP {response.status_code}")
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise CameraError("Camera image exceeds byte limit")
                chunks.append(chunk)
        data = b"".join(chunks)
        return data, validate_capture(data)
    except httpx.TimeoutException as exc:
        raise CameraError("Camera capture timed out") from exc
    except httpx.RequestError as exc:
        raise CameraError("Camera unreachable; check WiFi/IP/port 80") from exc


async def capture_async(client, base, max_bytes=5242880):
    try:
        async with client.stream("GET", camera_endpoint(base, "/capture")) as response:
            if response.status_code != 200:
                raise CameraError(f"Camera /capture HTTP {response.status_code}")
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise CameraError("Camera image exceeds byte limit")
                chunks.append(chunk)
        data = b"".join(chunks)
        return data, validate_capture(data)
    except httpx.TimeoutException as exc:
        raise CameraError("Camera capture timed out") from exc
    except httpx.RequestError as exc:
        raise CameraError("Camera unreachable; check WiFi/IP/port 80") from exc
