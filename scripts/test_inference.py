"""Send a real test image; measure client roundtrip and save the matched annotated image."""
import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import httpx
from common import ROOT, write_json


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--api-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/inference")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    headers = {"X-API-Key": os.getenv("API_KEY", "")}
    end_to_end_start = time.perf_counter()
    image = args.image.read_bytes()
    read_ms = (time.perf_counter() - end_to_end_start) * 1000
    # No connection preflight is hidden outside roundtrip.
    with httpx.Client(timeout=args.timeout, headers=headers, trust_env=False) as client:
        start = time.perf_counter()
        response = client.post(args.api_url.rstrip("/") + "/predict",
            files={"file": (args.image.name, image, mimetypes.guess_type(args.image.name)[0] or "image/jpeg")},
            data={"source": "test_client", "include_image": "true"})
        received = time.perf_counter()
        roundtrip = (received - start) * 1000
        total = (received - end_to_end_start) * 1000
        if response.is_error:
            print(f"HTTP {response.status_code}: {response.text}")
            return 1
        result = response.json()
        metrics = {"request_id": result["request_id"], "report_token": result.pop("report_token"),
                   "capture_ms": None, "roundtrip_ms": roundtrip, "total_ms": total,
                   "upload_ms": None, "download_ms": None}
        report = client.post(args.api_url.rstrip("/") + "/metrics/report", json=metrics)
        report.raise_for_status()
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = result.pop("annotated_image_base64")
    annotated = args.output / f"{result['request_id']}.jpg"
    annotated.write_bytes(base64.b64decode(encoded, validate=True))
    result.update({"roundtrip_ms": roundtrip, "total_ms": total, "end_to_end_ms": total, "file_read_ms": read_ms,
                   "total_definition": "file_read + client_setup + request/response; this client is not ESP32 capture"})
    write_json(annotated.with_suffix(".json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Annotated image:", annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
