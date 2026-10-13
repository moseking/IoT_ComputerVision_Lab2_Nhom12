"""Collect real ESP32 captures independently of YOLO; do not label or train."""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import httpx
from dotenv import load_dotenv
from server.camera_client import CameraError, camera_endpoint, capture


def collect_one(client, camera_url, output):
    output = Path(output)
    data, (width, height, image_format) = capture(client, camera_url)
    moment = datetime.now(timezone.utc)
    filename = moment.strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]
    filename += ".jpg" if image_format == "JPEG" else ".png"
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / filename
    # Exclusive creation: no capture may overwrite another image.
    with path.open("xb") as stream:
        stream.write(data)
    record = {"timestamp": moment.isoformat(), "camera_source": camera_endpoint(camera_url, "/capture"),
              "image": "images/" + filename, "sha256": hashlib.sha256(data).hexdigest(),
              "width": width, "height": height, "image_size_bytes": len(data),
              "label_status": "unlabeled", "included_in_training": False}
    with (output / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=os.getenv("ESP32_CAMERA_URL", ""))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/custom_capture")
    args = parser.parse_args()
    if not args.camera or args.count <= 0 or args.interval < 0 or args.timeout <= 0:
        parser.error("Set --camera (or ESP32_CAMERA_URL), count>0, interval>=0, timeout>0")
    try:
        camera_endpoint(args.camera, "/capture")
    except CameraError as exc:
        parser.error(str(exc))
    successes = 0
    with httpx.Client(timeout=args.timeout, trust_env=False, follow_redirects=False) as client:
        for index in range(args.count):
            try:
                record = collect_one(client, args.camera, args.output)
                successes += 1
                print(f"[{index+1}/{args.count}] Saved {record['image']} {record['width']}x{record['height']}")
            except (CameraError, OSError) as exc:
                print(f"[{index+1}/{args.count}] Failed: {exc}", file=sys.stderr)
            if index + 1 < args.count:
                time.sleep(args.interval)
    print(f"{successes}/{args.count} real images saved. Label bounding boxes manually before preparing a dataset.")
    return 0 if successes == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
