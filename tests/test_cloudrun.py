"""Cloud Run preparation tests. No cloud calls, uploads or production database access.

Native tests use a mocked Storage transport and the original V7 ONNX bytes.
--container-smoke also runs a real loopback HTTP server inside Linux.
"""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# The same module is /app/scripts/verify_cloudrun.py in the Linux image.
VERIFY_PATH = Path(os.getenv("CLOUDRUN_VERIFY_SCRIPT", str(ROOT / "scripts/verify_cloudrun.py")))
sys.path.insert(0, str(VERIFY_PATH.parents[1]))
spec = importlib.util.spec_from_file_location("cloudrun_verify_under_test", VERIFY_PATH)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
MODEL = Path(os.getenv("CLOUDRUN_TEST_MODEL", str(ROOT / "models/cloud_v7/face_mask_v7.onnx")))
SAMPLES = Path(os.getenv("CLOUDRUN_TEST_IMAGES", str(ROOT / "data/processed/facemask_camera_v7/images/val")))
KEY = "cloudrun-isolated-test-key-not-production"
PROJECT = "unit-fixture.supabase.co"
OBJECT = "v7/face_mask_v7_f1fc04ca98bace9f.onnx"


def fixture_env(now=None, lifetime=604800, remaining=None, **claims):
    now = int(time.time()) if now is None else int(now)
    issued = now if remaining is None else now + remaining - lifetime
    payload = {"url": "facemask-models/" + OBJECT, "iat": issued, "exp": issued + lifetime}
    payload.update(claims)
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    # Deliberately NOT a valid Supabase signature: transport is always mocked.
    token = part({"alg": "HS256", "typ": "JWT"}) + "." + part(payload) + ".fixture_signature"
    return {"SUPABASE_URL": "https://" + PROJECT,
            "SUPABASE_MODEL_BUCKET": "facemask-models", "SUPABASE_MODEL_OBJECT": OBJECT,
            "CLOUD_MODEL_DOWNLOAD_URL": "https://" + PROJECT + "/storage/v1/object/sign/facemask-models/" + OBJECT + "?token=" + token,
            "CLOUD_MODEL_SHA256": verify.V7_SHA256, "CLOUD_API_KEY": KEY}


def sample_paths():
    return sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


@contextlib.contextmanager
def isolated_api(directory, fail_status=None):
    from server import cloud_app
    from server.cloud_bootstrap import ensure_model
    from server.inference_onnx import OnnxMaskDetectionModel
    destination = Path(directory) / "downloaded.onnx"
    env = fixture_env()
    verify.preflight(env)
    downloaded = []
    original = MODEL.read_bytes()
    def storage(request):
        if request.method != "GET" or request.url.host != PROJECT:
            raise AssertionError("Unexpected network operation; only mocked Storage GET is permitted")
        downloaded.append(request.method)
        return httpx.Response(fail_status or 200, content=original if fail_status is None else b"")
    with httpx.Client(transport=httpx.MockTransport(storage)) as transport:
        def load(path, sha, url):
            return ensure_model(path, sha, url, client=transport)
        instance = OnnxMaskDetectionModel(destination, expected_sha256=verify.V7_SHA256, max_pixels=4000000)
        with patch.dict(os.environ, env), patch.object(cloud_app, "KEY", KEY), \
             patch.object(cloud_app, "SHA", verify.V7_SHA256), patch.object(cloud_app, "MODEL_PATH", destination), \
             patch.object(cloud_app, "model", instance), patch.object(cloud_app, "ensure_model", side_effect=load):
            with TestClient(cloud_app.app) as client:
                yield client, downloaded, destination


class SignedUrlPolicyTests(unittest.TestCase):
    def test_seven_day_url_has_safe_metadata_without_token(self):
        env = fixture_env(now=2000000000)
        result = verify.preflight(env, now=2000000000)
        self.assertEqual(result["remaining_seconds"], 604800)
        self.assertFalse(result["jwt_signature_verified_locally"])
        self.assertNotIn("token=", json.dumps(result))
        self.assertNotIn(KEY, json.dumps(result))

    def test_expired_or_less_than_one_hour_refuses_startup(self):
        for remaining in (-1, 0, 3599):
            with self.subTest(remaining=remaining), self.assertRaisesRegex(verify.CheckError, "renew"):
                verify.preflight(fixture_env(now=2000000000, remaining=remaining), now=2000000000)
        self.assertEqual(verify.preflight(fixture_env(now=2000000000, remaining=3600), now=2000000000)["remaining_seconds"], 3600)

    def test_more_than_seven_days_is_rejected(self):
        with self.assertRaisesRegex(verify.CheckError, "7 days"):
            verify.preflight(fixture_env(now=2000000000, lifetime=604801), now=2000000000)

    def test_renewal_warning_below_24_hours(self):
        result = verify.preflight(fixture_env(now=2000000000, remaining=7200), now=2000000000)
        self.assertTrue(result["renew_before_demo"])

    def test_invalid_time_claims_future_issue_and_wrong_object_are_rejected(self):
        for change in ({"iat": None}, {"exp": True}, {"exp": 1}, {"iat": 2000001000, "exp": 2000005000}, {"url": "other/model.onnx"}):
            with self.subTest(change=change), self.assertRaises(verify.CheckError):
                verify.preflight(fixture_env(now=2000000000, **change), now=2000000000)

    def test_exact_project_path_and_https_are_required(self):
        good = fixture_env(now=2000000000)
        mutations = [
            good["CLOUD_MODEL_DOWNLOAD_URL"].replace(PROJECT, "other.supabase.co"),
            good["CLOUD_MODEL_DOWNLOAD_URL"].replace("https:", "http:"),
            good["CLOUD_MODEL_DOWNLOAD_URL"].replace("/object/sign/", "/object/public/"),
            good["CLOUD_MODEL_DOWNLOAD_URL"] + "&token=extra",
            good["CLOUD_MODEL_DOWNLOAD_URL"] + "#fragment",
            good["CLOUD_MODEL_DOWNLOAD_URL"].replace("https://", "https://user@"),
        ]
        for url in mutations:
            with self.subTest(url_kind=mutations.index(url)), self.assertRaises(verify.CheckError) as exc:
                verify.preflight({**good, "CLOUD_MODEL_DOWNLOAD_URL": url}, now=2000000000)
            self.assertNotIn("fixture_signature", str(exc.exception))
            self.assertNotIn("token=", str(exc.exception))

    def test_missing_signed_url_or_wrong_model_pin_and_key_fail_closed(self):
        good = fixture_env(now=2000000000)
        for updates in ({"CLOUD_MODEL_DOWNLOAD_URL": ""}, {"CLOUD_MODEL_SHA256": "0"*64}, {"CLOUD_API_KEY": ""}):
            with self.subTest(fields=list(updates)), self.assertRaises(verify.CheckError):
                verify.preflight({**good, **updates}, now=2000000000)

    def test_startup_reuses_existing_entry_and_does_not_log_credentials(self):
        output = io.StringIO()
        with patch.dict(os.environ, fixture_env()), patch.object(verify.runpy, "run_module") as run, \
             contextlib.redirect_stdout(output):
            verify.serve()
        run.assert_called_once_with("server.cloud_entry", run_name="__main__")
        self.assertNotIn(KEY, output.getvalue())
        self.assertNotIn("fixture_signature", output.getvalue())

    def test_model_cache_cannot_target_the_original_model(self):
        with patch.dict(os.environ, {**fixture_env(), "CLOUD_MODEL_PATH": str(MODEL)}):
            with self.assertRaisesRegex(verify.CheckError, "cache"):
                verify.serve()


class PackagingTests(unittest.TestCase):
    def test_dockerfile_uses_dedicated_requirements_entry_and_nonroot_without_model(self):
        text = (ROOT / "cloud/cloudrun/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY cloud/cloudrun/requirements.txt", text)
        self.assertIn("scripts/verify_cloudrun.py", text)
        self.assertIn("USER appuser", text)
        copies = [line for line in text.splitlines() if line.startswith(("COPY ", "ADD "))]
        self.assertFalse(any("models/" in line or ".env" in line or "COPY . " in line for line in copies))
        self.assertIn("pip check", text)

    def test_dockerignore_default_deny_does_not_allow_artifacts_or_private_configuration(self):
        text = (ROOT / "cloud/cloudrun/Dockerfile.dockerignore").read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        self.assertEqual(lines[0], "**")
        for rule in lines[1:]:
            self.assertTrue(rule.startswith("!"))
            self.assertNotIn("models", rule)
            self.assertNotIn(".env", rule)
            self.assertNotIn("data/", rule)
            self.assertNotIn(".runtime", rule)

    def test_dependencies_are_cpu_only_and_pinned(self):
        lines = (ROOT / "cloud/cloudrun/requirements.txt").read_text().splitlines()
        requirements = [line for line in lines if line and not line.startswith("#")]
        self.assertTrue(all("==" in line for line in requirements))
        self.assertFalse(any("torch" in line or "ultralytics" in line or "cuda" in line for line in requirements))
        self.assertTrue(any(line.startswith("opencv-python-headless==") for line in requirements))


class VerifierTests(unittest.TestCase):
    def test_http_is_only_allowed_on_explicit_loopback(self):
        self.assertEqual(verify.checked_origin("http://127.0.0.1:8080", True), "http://127.0.0.1:8080")
        for origin, local in (("http://127.0.0.1", False), ("http://public.example", True), ("https://user:pass@host", False), ("https://host/path", False)):
            with self.assertRaises(verify.CheckError):
                verify.checked_origin(origin, local)

    def test_request_failures_do_not_expose_key_or_url(self):
        def fail(request):
            raise httpx.ReadTimeout("secret token=" + KEY, request=request)
        with httpx.Client(transport=httpx.MockTransport(fail)) as client:
            with self.assertRaisesRegex(verify.CheckError, "ReadTimeout") as error:
                verify.verify_api("https://unit.example", "unused", KEY, client=client)
            self.assertNotIn(KEY, str(error.exception))

    def test_stdout_cli_has_no_default_file_writes_or_env_loading(self):
        text = VERIFY_PATH.read_text(encoding="utf-8")
        self.assertNotIn("write_text(", text)
        self.assertNotIn("write_bytes(", text)
        self.assertNotIn("load_dotenv(", text)
        self.assertNotIn("dotenv_values(", text)


@unittest.skipUnless(MODEL.exists() and SAMPLES.is_dir(), "Original V7 model and validation images required")
class RealOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="cloudrun-v7-tests-")
        cls.context = isolated_api(cls.temporary.name)
        cls.client, cls.downloads, cls.cached = cls.context.__enter__()
        cls.headers = {"X-API-Key": KEY}
        cls.sample = sample_paths()[0]
        cls.data = cls.sample.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        cls.temporary.cleanup()

    def test_original_sha_real_download_load_and_three_class_contract(self):
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(), verify.V7_SHA256)
        self.assertEqual(hashlib.sha256(self.cached.read_bytes()).hexdigest(), verify.V7_SHA256)
        self.assertEqual(self.downloads, ["GET"])
        info = self.client.get("/model/info", headers=self.headers).json()
        self.assertTrue(info["loaded"])
        self.assertEqual(info["classes"], {str(k):v for k,v in verify.CLASSES.items()})

    def test_multiple_inferences_do_not_download_again_and_exact_frame_is_returned(self):
        for _ in range(2):
            response = self.client.post("/predict", headers=self.headers, files={"file": ("frame.png", self.data, "image/png")})
            self.assertEqual(response.status_code, 200)
            with Image.open(io.BytesIO(self.data)) as image:
                verify.validate_prediction(response.json(), self.data, *image.size)
        self.assertEqual(self.downloads, ["GET"])

    def test_real_predictions_cover_all_three_labels_on_available_validation_images(self):
        observed = set()
        checked = 0
        for path in sample_paths():
            response = self.client.post("/predict", headers=self.headers, files={"file": (path.name, path.read_bytes(), "application/octet-stream")})
            self.assertEqual(response.status_code, 200)
            observed.update(d["class_id"] for d in response.json()["detections"])
            checked += 1
            if observed == {0, 1, 2}:
                break
        self.assertEqual(observed, {0, 1, 2}, "All three labels must be observed; this is a smoke test, not an accuracy estimate")
        print(json.dumps({"real_onnx_smoke": {"images_examined": checked, "observed_class_ids": sorted(observed)}}))

    def test_auth_invalid_oversize_and_missing_file_errors(self):
        c = self.client
        self.assertEqual(c.post("/predict", files={"file": ("x.jpg", self.data)}).status_code, 401)
        self.assertEqual(c.post("/predict", headers=self.headers, files={"file": ("bad.jpg", b"invalid", "image/jpeg")}).status_code, 400)
        self.assertEqual(c.post("/predict", headers=self.headers, content=b"x"*(6*1024*1024)).status_code, 413)
        self.assertEqual(c.post("/predict", headers=self.headers).status_code, 422)

    def test_pixel_limit_applies_before_inference(self):
        buf = io.BytesIO()
        Image.new("RGB", (2001, 2000)).save(buf, format="PNG")
        response = self.client.post("/predict", headers=self.headers, files={"file": ("large.png", buf.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 400)

    def test_verifier_accepts_real_response_and_monotonic_timings(self):
        def proxy(request):
            r = self.client.request(request.method, request.url.path, headers=dict(request.headers), content=request.read())
            return httpx.Response(r.status_code, content=r.content, headers=dict(r.headers))
        with httpx.Client(transport=httpx.MockTransport(proxy)) as client:
            result = verify.verify_api("https://unit.example", self.sample, KEY, repeat=2, client=client)
        self.assertEqual(result["samples"], 2)
        self.assertFalse(result["cold_start_measured"])
        self.assertFalse(result["cloud_run_deployment_verified"])
        for row in result["rows"]:
            self.assertGreaterEqual(row["end_to_end_ms"], row["roundtrip_ms"])
            self.assertGreaterEqual(row["roundtrip_ms"] + 2, row["inference_ms"])

    def test_verifier_rejects_wrong_artifact_and_timing(self):
        body = self.client.post("/predict", headers=self.headers, files={"file": ("frame.png", self.data)}).json()
        with Image.open(io.BytesIO(self.data)) as image:
            shape = image.size
        for change in ({"model_sha256": "0"*64}, {"input_sha256": "0"*64}, {"queue_ms": float("nan")}, {"width": -1}):
            with self.subTest(fields=list(change)), self.assertRaises(verify.CheckError):
                verify.validate_prediction({**body, **change}, self.data, *shape)

    def test_expiry_after_startup_does_not_reload_but_new_startup_is_refused(self):
        env = fixture_env(now=int(time.time()) - 604900)
        with self.assertRaises(verify.CheckError):
            verify.preflight(env)
        with patch.dict(os.environ, env):
            response = self.client.post("/predict", headers=self.headers, files={"file": ("frame.png", self.data)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.downloads, ["GET"])

    def test_tampered_storage_bytes_are_never_published_as_cache(self):
        from server.cloud_bootstrap import ensure_model
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "cache.onnx"
            with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"corrupted-model"))) as client:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    ensure_model(destination, verify.V7_SHA256, fixture_env()["CLOUD_MODEL_DOWNLOAD_URL"], client=client)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + ".download").exists())

    def test_unavailable_download_prevents_startup_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "HTTP 403"):
                with isolated_api(tmp, fail_status=403):
                    self.fail("Startup must not succeed")
            self.assertFalse((Path(tmp)/"downloaded.onnx").exists())


def container_smoke():
    """Invoked only inside the built Linux image with readonly local fixtures."""
    import resource
    import uvicorn
    from server import cloud_app
    from server.cloud_bootstrap import ensure_model
    from server.inference_onnx import OnnxMaskDetectionModel
    assert sys.platform == "linux"
    assert os.geteuid() != 0
    assert not Path("/app/models").exists(), "Production image must not contain model artifacts"
    assert not Path("/app/.env").exists()
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == verify.V7_SHA256
    calls = []
    env = {**fixture_env(), "PORT": "8080", "CLOUD_MODEL_PATH": "/tmp/facemask/face_mask_v7.onnx"}
    destination = Path(env["CLOUD_MODEL_PATH"])
    assert not destination.exists(), "Test must exercise an empty cache"
    def storage(request):
        assert request.method == "GET" and request.url.host == PROJECT
        calls.append("GET")
        return httpx.Response(200, content=MODEL.read_bytes())
    server = None
    thread = None
    def start_server(app, host, port, workers):
        nonlocal server, thread
        assert host == "0.0.0.0" and port == 8080 and workers == 1
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, workers=workers, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
    with httpx.Client(transport=httpx.MockTransport(storage)) as transport:
        def load(path, sha, url):
            return ensure_model(path, sha, url, client=transport)
        instance = OnnxMaskDetectionModel(destination, expected_sha256=verify.V7_SHA256, max_pixels=4000000)
        with patch.dict(os.environ, env), patch.object(cloud_app, "KEY", KEY), patch.object(cloud_app, "SHA", verify.V7_SHA256), \
             patch.object(cloud_app, "MODEL_PATH", destination), patch.object(cloud_app, "model", instance), \
             patch.object(cloud_app, "ensure_model", side_effect=load), patch.object(uvicorn, "run", side_effect=start_server):
            try:
                verify.serve()
                deadline = time.monotonic() + 45
                while not server.started:
                    if not thread.is_alive() or time.monotonic() > deadline:
                        raise AssertionError("Linux server did not start")
                    time.sleep(.05)
                report = verify.verify_api("http://127.0.0.1:8080", sample_paths()[0], KEY, repeat=3, allow_local_http=True)
                with httpx.Client(base_url="http://127.0.0.1:8080", trust_env=False) as client:
                    assert client.post("/predict", files={"file": ("bad.jpg", b"x")}).status_code == 401
                    assert client.post("/predict", headers={"X-API-Key": KEY}, files={"file": ("bad.jpg", b"x")}).status_code == 400
                    assert client.post("/predict", headers={"X-API-Key": KEY}, content=b"x"*(6*1024*1024)).status_code == 413
                    large = io.BytesIO()
                    Image.new("RGB", (2000, 2000), color=(96, 96, 96)).save(large, format="PNG")
                    edge = client.post("/predict", headers={"X-API-Key": KEY}, files={"file": ("edge.png", large.getvalue(), "image/png")})
                    assert edge.status_code == 200, "Valid image at the 4M-pixel limit must run"
                    verify.validate_prediction(edge.json(), large.getvalue(), 2000, 2000)
                assert calls == ["GET"], "Model must download once per fresh container, not per request"
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == verify.V7_SHA256
                assert "torch" not in sys.modules and "ultralytics" not in sys.modules
                report.update(linux_container=True, nonroot=True, storage_transport="mocked_no_cloud_access",
                              downloaded_once=True, original_model_mount_readonly=True, four_million_pixel_image_passed=True,
                              peak_process_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2))
                print("LINUX_CONTAINER_REPORT=" + json.dumps(report), flush=True)
            finally:
                if server:
                    server.should_exit = True
                if thread:
                    thread.join(timeout=10)


if __name__ == "__main__":
    if sys.argv[1:] == ["--container-smoke"]:
        container_smoke()
    else:
        unittest.main()
