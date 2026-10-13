"""Data integrity and API checks; optional real model tests use TEST_MODEL_PATH."""
import ast
import base64
import io
import json
import os
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
temporary_db = tempfile.TemporaryDirectory(prefix="face-mask-api-tests-")
os.environ["LATENCY_DB_PATH"] = str(Path(temporary_db.name) / "latency.db")
checkpoint = os.getenv("TEST_MODEL_PATH", "")
os.environ["MODEL_PATH"] = checkpoint or str(ROOT / "models/__unavailable_test_checkpoint__.pt")
# Tests run isolated from any production API_KEY/CORS configuration.
os.environ["API_KEY"] = ""
os.environ["TELEGRAM_ENABLED"] = "false"
os.environ["CORS_ORIGINS"] = "https://dashboard.example.test"

from server.main import app, model
from scripts.common import CLASSES


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = ROOT / "dataset/split_manifest.json"
        if not manifest.is_file():
            raise unittest.SkipTest("Prepared dataset absent")
        cls.manifest = json.loads(manifest.read_text(encoding="utf-8"))

    def test_python_syntax(self):
        for folder in ("server", "scripts", "tests"):
            for path in (ROOT / folder).glob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_yaml_classes_and_paths(self):
        config = yaml.safe_load((ROOT / "dataset/data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["names"], CLASSES)
        for split in ("train", "val", "test"):
            self.assertTrue((Path(config["path"]) / config[split]).is_dir())

    def test_yolo_labels_for_every_image(self):
        total = 0
        for split, records in self.manifest["splits"].items():
            seen = Counter()
            image_dir, label_dir = ROOT / "dataset/images" / split, ROOT / "dataset/labels" / split
            self.assertEqual({p.stem for p in image_dir.iterdir()}, {p.stem for p in label_dir.iterdir()})
            for record in records:
                image_path = image_dir / record["image"]
                with Image.open(image_path) as image:
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)
                lines = (label_dir / (image_path.stem + ".txt")).read_text().splitlines()
                self.assertEqual(len(lines), record["objects"])
                for line in lines:
                    parts = line.split()
                    self.assertEqual(len(parts), 5)
                    class_id = int(parts[0]); values = list(map(float, parts[1:]))
                    self.assertIn(class_id, CLASSES)
                    self.assertTrue(all(0 <= v <= 1 for v in values))
                    self.assertGreater(values[2], 0)
                    self.assertGreater(values[3], 0)
                    seen[class_id] += 1
                total += 1
            self.assertEqual(set(seen), set(CLASSES), f"Missing class in {split}")
        self.assertGreater(total, 0)

    def test_no_exact_or_perceptual_leakage(self):
        hashes, items = {}, []
        threshold = self.manifest["near_duplicate_distance"]
        for split, records in self.manifest["splits"].items():
            for record in records:
                pixel_hash = record["pixel_hash"]
                if pixel_hash in hashes:
                    self.assertEqual(hashes[pixel_hash], split)
                hashes[pixel_hash] = split
                items.append((split, record["dhash"]))
        if threshold >= 0:
            for index, (split, digest) in enumerate(items):
                for other_split, other_digest in items[:index]:
                    if split != other_split:
                        self.assertGreater((digest ^ other_digest).bit_count(), threshold)

    def test_real_multi_face_test_cases_exist(self):
        counts = [r["objects"] for r in self.manifest["splits"]["test"]]
        for number in (1, 2, 3):
            self.assertIn(number, counts)
        self.assertTrue(any(number >= 4 for number in counts))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.sample = next((ROOT / "dataset/images/test").glob("*.png"), None)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        temporary_db.cleanup()

    def test_startup_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_loaded"], bool(checkpoint))
        self.assertEqual(self.client.get("/model/info").json()["classes"], {str(k): v for k, v in CLASSES.items()})

    def test_dashboard_static_assets(self):
        for route in ("/", "/static/style.css", "/static/app.js", "/docs", "/config"):
            self.assertEqual(self.client.get(route).status_code, 200, route)
        self.assertIn("Face Mask Detection", self.client.get("/").text)

    def test_cors_headers_on_upload_rejection(self):
        response = self.client.post("/predict", content=b"x", headers={
            "Content-Length": "999999999", "Origin": "https://dashboard.example.test"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://dashboard.example.test")
    def test_missing_file_and_body_limit(self):
        self.assertEqual(self.client.post("/predict", data={"source": "test_client"}).status_code, 422)
        self.assertEqual(self.client.post("/predict", content=b"x",
                         headers={"Content-Length": "999999999", "Content-Type": "application/octet-stream"}).status_code, 413)
        response = self.client.post("/metrics/report", json={"request_id": "unknown",
            "report_token": "unknown", "roundtrip_ms": 1, "total_ms": 0, "capture_ms": 100})
        self.assertEqual(response.status_code, 422)

    def test_unavailable_or_corrupt_image(self):
        response = self.client.post("/predict", files={"file": ("bad.jpg", b"corrupt", "image/jpeg")})
        self.assertEqual(response.status_code, 400 if checkpoint else 503)

    def test_history_has_no_report_tokens(self):
        response = self.client.get("/metrics/history?limit=10")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all("report_token" not in row for row in response.json()["items"]))
        self.assertEqual(self.client.get("/metrics/history?limit=0").status_code, 422)
        self.assertEqual(self.client.get("/results/latest/image?request_id=unknown").status_code in (404, 409), True)

    def test_real_prediction_reporting_and_load_once(self):
        if not checkpoint or self.sample is None:
            self.skipTest("Real trained checkpoint required via TEST_MODEL_PATH")
        model_identity = id(model._model)
        started = time.perf_counter()
        response = self.client.post("/predict", files={"file": (self.sample.name, self.sample.read_bytes(), "image/png")},
                                   data={"source": "test_client", "include_image": "true"})
        received = time.perf_counter()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIsNone(result["roundtrip_ms"])
        self.assertIsNone(result["total_ms"])
        self.assertEqual(result["detection_count"], len(result["detections"]))
        self.assertGreater(result["inference_ms"], 0)
        self.assertGreaterEqual(result["server_processing_ms"], result["inference_ms"])
        for detection in result["detections"]:
            self.assertIn(detection["class_id"], CLASSES)
            self.assertGreaterEqual(detection["confidence"], 0)
            self.assertLessEqual(detection["confidence"], 1)
        with Image.open(io.BytesIO(base64.b64decode(result["annotated_image_base64"]))) as annotated:
            self.assertEqual(annotated.size, (result["width"], result["height"]))
        milliseconds = (received - started) * 1000
        metrics = {"request_id": result["request_id"], "report_token": result["report_token"],
                   "roundtrip_ms": milliseconds, "total_ms": milliseconds, "capture_ms": None}
        invalid_token = {**metrics, "report_token": "wrong"}
        self.assertEqual(self.client.post("/metrics/report", json=invalid_token).status_code, 403)
        first = self.client.post("/metrics/report", json=metrics)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(self.client.post("/metrics/report", json=metrics).json(), first.json())
        latest = self.client.get("/results/latest").json()["result"]
        self.assertEqual(latest["roundtrip_ms"], milliseconds)
        self.assertEqual(self.client.get(latest["annotated_image_url"]).status_code, 200)
        self.assertNotIn("report_token", latest)
        second = self.client.post("/predict/image", files={"file": (self.sample.name, self.sample.read_bytes(), "image/png")})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers["content-type"], "image/jpeg")
        self.assertEqual(id(model._model), model_identity)
        summary = self.client.get("/metrics/summary").json()
        self.assertEqual(summary["roundtrip_ms"]["samples"], 1)
        self.assertEqual(summary["total_requests"], 2)


if __name__ == "__main__":
    unittest.main()
