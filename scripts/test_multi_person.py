"""API smoke tests on real held-out images with 1/2/3/many annotated faces."""
import argparse
import base64
import json
import mimetypes
import yaml
import os
import time
from pathlib import Path
import httpx
from dotenv import load_dotenv
from common import ROOT, write_json


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/multi_person")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/facemask_baseline_v1/data.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data_root = Path(config["path"])
    if not data_root.is_absolute(): data_root = args.data.resolve().parent / data_root
    manifest = data_root / "split_manifest.json"
    if not manifest.is_file():
        parser.error("Prepare the dataset first")
    samples = json.loads(manifest.read_text(encoding="utf-8"))["splits"]["test"]
    report = {"mode": "real_test_images_not_live_ESP32", "cases": []}
    headers = {"X-API-Key": os.getenv("API_KEY", "")}
    with httpx.Client(timeout=60, headers=headers, trust_env=False) as client:
        for name, count in (("one", 1), ("two", 2), ("three", 3), ("many", 4)):
            item = next((r for r in samples if r["objects"] == count or (name == "many" and r["objects"] >= 4)), None)
            if item is None:
                report["cases"].append({"case": name, "status": "no_matching_test_image"})
                continue
            image = data_root / config["test"] / item["image"]
            total_start = time.perf_counter()
            contents = image.read_bytes()
            start = time.perf_counter()
            response = client.post(args.api_url.rstrip("/") + "/predict",
                files={"file": (image.name, contents, mimetypes.guess_type(image.name)[0] or "image/png")},
                data={"source": "test_client", "include_image": "true"})
            received = time.perf_counter()
            response.raise_for_status()
            result = response.json()
            metrics = {"request_id": result["request_id"], "report_token": result.pop("report_token"),
                "roundtrip_ms": (received-start)*1000, "total_ms": (received-total_start)*1000,
                "capture_ms": None, "upload_ms": None, "download_ms": None}
            client.post(args.api_url.rstrip("/") + "/metrics/report", json=metrics).raise_for_status()
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"{name}.jpg").write_bytes(base64.b64decode(result.pop("annotated_image_base64"), validate=True))
            result.update({"case": name, "image": image.name, "ground_truth_count": item["objects"],
                "ground_truth_classes": item["class_counts"], "roundtrip_ms": metrics["roundtrip_ms"],
                "total_ms": metrics["total_ms"], "end_to_end_ms": metrics["total_ms"], "status": "measured"})
            report["cases"].append(result)
            print(f"{name}: ground_truth={item['objects']} model_detected={result['detection_count']}")
    write_json(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
