"""Cloud Run-only preflight, entry point and read-only inference verifier.

Does not read .env, upload models, modify Render, or write reports.
JWT claims are inspected for expiry policy, NOT cryptographically authenticated.
Supabase HTTPS and the pinned artifact SHA-256 validate the actual download.
"""
import argparse
import base64
import hashlib
import io
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import runpy
import statistics
import sys
import time
from urllib.parse import parse_qs, quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
V7_SHA256 = "f1fc04ca98bace9f1a1a5537d83fc9c52574514c0641959084c94a9f0750cbf4"
CLASSES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
MAX_URL_LIFETIME = 7 * 24 * 3600
MIN_STARTUP_VALIDITY = 3600
MAX_RESPONSE_BYTES = 12 * 1024 * 1024


class CheckError(ValueError):
    """Only fixed, non-secret diagnostic messages may be exposed."""


def preflight(env=None, now=None):
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    if env.get("CLOUD_MODEL_SHA256") != V7_SHA256:
        raise CheckError("CLOUD_MODEL_SHA256 must pin the verified V7 ONNX artifact")
    key = env.get("CLOUD_API_KEY", "")
    if len(key) < 24 or not key.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise CheckError("CLOUD_API_KEY must contain at least 24 printable ASCII characters")
    try:
        project = urlsplit(env.get("SUPABASE_URL", ""))
        signed = urlsplit(env.get("CLOUD_MODEL_DOWNLOAD_URL", ""))
        valid_host = project.hostname and re.fullmatch(r"[a-z0-9-]+\.supabase\.co", project.hostname)
        if not valid_host or project.scheme != "https" or project.username or project.password or project.port or project.path not in ("", "/") or project.query or project.fragment:
            raise CheckError("SUPABASE_URL must be the exact HTTPS project origin")
        if signed.scheme != "https" or signed.hostname != project.hostname or signed.username or signed.password or signed.port or signed.fragment:
            raise CheckError("Signed URL must belong to the configured Supabase HTTPS project")
        bucket = env.get("SUPABASE_MODEL_BUCKET", "facemask-models")
        obj = env.get("SUPABASE_MODEL_OBJECT", "v7/face_mask_v7_f1fc04ca98bace9f.onnx")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", bucket):
            raise CheckError("Invalid private bucket name")
        if not obj or len(obj) > 512 or any(part in ("", ".", "..") for part in obj.split("/")):
            raise CheckError("Invalid model object path")
        expected_path = "/storage/v1/object/sign/" + quote(bucket + "/" + obj, safe="/")
        if signed.path != expected_path:
            raise CheckError("Signed URL bucket/object does not match configuration")
        query = parse_qs(signed.query, keep_blank_values=True)
        if set(query) - {"token", "download"} or len(query.get("token", [])) != 1:
            raise CheckError("Signed URL must contain exactly one token")
        token = query["token"][0]
        if not token.isascii() or len(token) > 8192:
            raise CheckError("Invalid signed URL token")
        parts = token.split(".")
        if len(parts) != 3 or not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts):
            raise CheckError("Expected a three-part Supabase signed URL JWT")
        def decode(part):
            return json.loads(base64.b64decode(part + "=" * (-len(part) % 4), altchars=b"-_", validate=True))
        header, claims = decode(parts[0]), decode(parts[1])
        if not isinstance(header, dict) or header.get("alg", "").lower() in ("", "none"):
            raise CheckError("Unsigned model URL token is not allowed")
        if not isinstance(claims, dict) or any(type(claims.get(k)) is not int for k in ("iat", "exp")):
            raise CheckError("Signed URL must expose integer iat and exp claims")
        issued, expires = claims["iat"], claims["exp"]
        if not 0 < expires - issued <= MAX_URL_LIFETIME:
            raise CheckError("Signed URL lifetime must be positive and at most 7 days")
        if issued > now + 300:
            raise CheckError("Signed URL issue time is in the future; check the clock")
        if expires - now < MIN_STARTUP_VALIDITY:
            raise CheckError("Signed URL is expired or has less than 1 hour left; renew before startup")
        if claims.get("url") != bucket + "/" + obj:
            raise CheckError("Signed token object claim does not match configuration")
    except CheckError:
        raise
    except Exception:
        raise CheckError("Invalid signed URL format or JWT claims") from None
    remaining = int(expires - now)
    return {
        "model_sha256": V7_SHA256, "project_host": project.hostname,
        "bucket": bucket, "object": obj,
        "url_expires_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires)),
        "remaining_seconds": remaining, "renew_before_demo": remaining < 86400,
        "jwt_signature_verified_locally": False,
    }


def serve():
    """Validate Cloud Run configuration before the unchanged server starts."""
    metadata = preflight()
    path = Path(os.environ.get("CLOUD_MODEL_PATH", "/tmp/facemask/face_mask_v7.onnx")).resolve()
    cache_root = Path("/tmp/facemask").resolve()
    if path.parent != cache_root or path.suffix != ".onnx":
        raise CheckError("Cloud Run model cache must be an ONNX file directly inside /tmp/facemask")
    os.environ["CLOUD_MODEL_PATH"] = str(path)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    print(json.dumps({"event": "cloudrun_model_preflight", **metadata}), flush=True)
    if metadata["renew_before_demo"]:
        print("cloudrun_notice: renew signed URL before the next demo; less than 24 hours remain", file=sys.stderr, flush=True)
    sys.path.insert(0, str(ROOT))
    runpy.run_module("server.cloud_entry", run_name="__main__")


def checked_origin(value, allow_local_http=False):
    try:
        p = urlsplit(value)
        if not p.hostname or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment:
            raise CheckError("API URL must be an origin without credentials, path or query")
        try:
            address = ipaddress.ip_address(p.hostname)
        except ValueError:
            address = None
        local = p.hostname == "localhost" or bool(address and address.is_loopback)
        if p.scheme != "https" and not (allow_local_http and p.scheme == "http" and local):
            raise CheckError("Use HTTPS; local HTTP requires --allow-local-http and a loopback address")
        return value.rstrip("/")
    except CheckError:
        raise
    except Exception:
        raise CheckError("Invalid API origin") from None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_prediction(body, data, width, height):
    from PIL import Image
    if not isinstance(body, dict) or body.get("success") is not True:
        raise CheckError("Prediction was not successful")
    if body.get("model_sha256") != V7_SHA256 or body.get("runtime") != "onnxruntime" or body.get("inference_location") != "cloud":
        raise CheckError("Wrong artifact, runtime or inference location")
    if body.get("input_sha256") != hashlib.sha256(data).hexdigest() or (body.get("width"), body.get("height")) != (width, height):
        raise CheckError("Response does not match the exact input image")
    detections = body.get("detections")
    if not isinstance(detections, list) or len(detections) > 300 or type(body.get("detection_count")) is not int or body["detection_count"] != len(detections):
        raise CheckError("Invalid detection count")
    for d in detections:
        if not isinstance(d, dict):
            raise CheckError("Invalid detection")
        cls, confidence = d.get("class_id"), d.get("confidence")
        if type(cls) is not int or cls not in CLASSES or d.get("class_name") != CLASSES[cls] or not finite_number(confidence) or confidence > 1:
            raise CheckError("Invalid detection class/confidence")
        box = d.get("bbox")
        coords = [box.get(k) for k in ("x1", "y1", "x2", "y2")] if isinstance(box, dict) else []
        if len(coords) != 4 or not all(finite_number(x) for x in coords):
            raise CheckError("Invalid bounding box")
        x1, y1, x2, y2 = coords
        if not (x1 <= x2 <= width and y1 <= y2 <= height):
            raise CheckError("Bounding box lies outside the input")
    if not all(finite_number(body.get(k)) for k in ("inference_ms", "server_processing_ms", "queue_ms")):
        raise CheckError("Invalid elapsed-time metadata")
    if body["server_processing_ms"] + 2 < body["inference_ms"]:
        raise CheckError("Inconsistent processing/inference times")
    if not isinstance(body.get("inference_definition"), str) or not body["inference_definition"]:
        raise CheckError("Missing inference timing definition")
    try:
        annotation = base64.b64decode(body["annotated_image_base64"], validate=True)
        with Image.open(io.BytesIO(annotation)) as image:
            if image.format != "JPEG" or image.size != (width, height):
                raise ValueError()
            image.verify()
    except Exception:
        raise CheckError("Invalid annotation or wrong image dimensions") from None


def verify_api(url, image_path, key, repeat=3, timeout=120, allow_local_http=False, client=None):
    """Print-friendly report only; never writes an image, DB, .env or report file."""
    import httpx
    from PIL import Image
    origin = checked_origin(url, allow_local_http)
    if not key or not key.isascii():
        raise CheckError("Set the dedicated Cloud Run API key in the selected environment variable")
    if not 1 <= repeat <= 30 or not 0 < timeout <= 300:
        raise CheckError("Repeat must be 1..30 and timeout must be <=300 seconds")
    owned = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 15)), trust_env=False, follow_redirects=False)
    headers = {"X-API-Key": key}
    try:
        health = client.get(origin + "/health")
        if health.status_code != 200 or health.json().get("model_loaded") is not True:
            raise CheckError("Health probe failed or model is unavailable")
        info_response = client.get(origin + "/model/info", headers=headers)
        if info_response.status_code != 200:
            raise CheckError("Model info request failed")
        info = info_response.json()
        if info.get("loaded") is not True or info.get("model_sha256") != V7_SHA256 or info.get("runtime") != "onnxruntime" or info.get("device") != "cpu":
            raise CheckError("Model info does not match V7 ONNX CPU")
        if info.get("classes") != {str(k): v for k, v in CLASSES.items()} or info.get("image_size") != 512 or info.get("confidence_threshold") != .35 or info.get("nms_iou") != .45 or info.get("nms_class_agnostic") is not True:
            raise CheckError("Model class/preprocessing/NMS contract mismatch")
        rows = []
        for _ in range(repeat):
            total_start = time.perf_counter()
            data = Path(image_path).read_bytes()
            if not data or len(data) > 5 * 1024 * 1024:
                raise CheckError("Test image must be nonempty and <=5 MiB")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                    if image.format not in ("JPEG", "PNG") or width * height > 4000000:
                        raise ValueError()
                    image.verify()
            except Exception:
                raise CheckError("Test image must be valid JPEG/PNG and <=4 million pixels") from None
            request_start = time.perf_counter()
            with client.stream("POST", origin + "/predict", headers=headers,
                               files={"file": ("frame", data, "application/octet-stream")}) as response:
                if response.status_code != 200:
                    raise CheckError("Prediction HTTP " + str(response.status_code))
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise CheckError("Prediction response exceeds 12 MiB")
            received = time.perf_counter()
            try:
                result = json.loads(raw)
            except Exception:
                raise CheckError("Prediction JSON is invalid") from None
            validate_prediction(result, data, width, height)
            ready = time.perf_counter()
            rows.append({
                "request_id": result["request_id"], "input_sha256": result["input_sha256"],
                "image_bytes": len(data), "detections": result["detections"],
                "class_counts": {name: sum(d["class_id"] == cls for d in result["detections"]) for cls, name in CLASSES.items()},
                "inference_ms": result["inference_ms"], "queue_ms": result["queue_ms"],
                "server_processing_ms": result["server_processing_ms"],
                "roundtrip_ms": round((received - request_start) * 1000, 3),
                "end_to_end_ms": round((ready - total_start) * 1000, 3),
            })
        def percentiles(key):
            values = sorted(row[key] for row in rows)
            return {"p50": round(statistics.median(values), 3),
                    "p95": values[max(0, math.ceil(.95 * len(values)) - 1)]}
        return {
            "test_location": "local_loopback" if allow_local_http and urlsplit(origin).scheme == "http" else "https_endpoint",
            "cloud_run_deployment_verified": False,
            "model_sha256": V7_SHA256, "runtime": "onnxruntime", "samples": len(rows),
            "cold_start_measured": False, "probe_order": "health_and_model_info_before_predictions",
            "definitions": {
                "inference_ms": "server monotonic preprocess+ONNX_forward+NMS; excludes decode/queue/annotation/upload",
                "roundtrip_ms": "client monotonic POST start through full response body; excludes health/model-info",
                "end_to_end_ms": "client monotonic local file read through JSON and JPEG verification; no camera capture or browser rendering",
            },
            "timing_summary_ms": {key: percentiles(key) for key in ("inference_ms", "roundtrip_ms", "end_to_end_ms")},
            "rows": rows,
        }
    except CheckError:
        raise
    except httpx.HTTPError as exc:
        raise CheckError("API network failure: " + type(exc).__name__) from None
    except Exception:
        raise CheckError("API verification failed; inspect non-secret server diagnostics") from None
    finally:
        if owned:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Check env and signed URL claims without any network or file writes")
    commands.add_parser("serve", help="Container entry point; download/load once at startup")
    verify = commands.add_parser("verify", help="Read-only API verification; JSON report goes to stdout")
    verify.add_argument("--url", required=True)
    verify.add_argument("--image", required=True)
    verify.add_argument("--key-env", default="CLOUDRUN_API_KEY")
    verify.add_argument("--repeat", type=int, default=3)
    verify.add_argument("--timeout", type=float, default=120)
    verify.add_argument("--allow-local-http", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            serve()
            return
        if args.command == "preflight":
            result = preflight()
        else:
            result = verify_api(args.url, args.image, os.getenv(args.key_env, ""),
                                args.repeat, args.timeout, args.allow_local_http)
        print(json.dumps(result, indent=2, ensure_ascii=True))
    except CheckError as exc:
        print("Cloud Run check: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
