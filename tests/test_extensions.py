"""New behavior tests use isolated mock transports; real YOLO stays in test_project."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from server.camera_client import CameraError, capture
from server.inference import CLASSES, MaskDetectionModel
from server.latency_store import LatencyStore
from server.telegram_service import TelegramConfig, TelegramService
from scripts.capture_dataset import collect_one
from scripts.checkpoint_tools import publish_checkpoints

ROOT = Path(__file__).resolve().parents[1]


def event(detections=None):
    return {"request_id": "unit-event", "timestamp": "2026-10-09T00:00:00+00:00", "source": "esp32",
            "detections": detections if detections is not None else [
                {"class_id": 1, "class_name": "without_mask", "confidence": 0.8,
                 "bbox": {"x1": 1, "y1": 1, "x2": 10, "y2": 10}}],
            "inference_ms": 12.0, "roundtrip_ms": None, "total_ms": None}


class StoreTests(unittest.TestCase):
    def test_existing_measurements_and_null_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"history.db"
            store = LatencyStore(path)
            record = {"timestamp":"2026-10-09T00:00:00+00:00","request_id":"success","source":"esp32",
                      "detection_count":2,"inference_ms":10,"server_processing_ms":12,
                      "capture_ms":3,"image_size_bytes":1000}
            token = store.insert(record)
            reported = store.report("success",token,{"roundtrip_ms":20,"total_ms":23,"capture_ms":3})
            self.assertEqual(reported["end_to_end_ms"],23)
            store = LatencyStore(path)
            self.assertEqual(store.get("success")["total_ms"],23)
            store.record_failure("failure","esp32",503,"http_503")
            rows = store.history()
            self.assertEqual(len(rows),2)
            failure = next(r for r in rows if r["status"]=="failure")
            self.assertIsNone(failure["inference_ms"])
            self.assertIsNone(failure["total_ms"])
            self.assertNotIn("report_token",failure)
            self.assertEqual(store.summary()["failed_requests"],1)
            self.assertEqual(store.summary()["inference_ms"]["samples"],1)
            self.assertEqual(store.latest_device_timestamp(),record["timestamp"])

    def test_concurrent_reports_keep_first_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=LatencyStore(Path(tmp)/"db.sqlite")
            token=store.insert({"timestamp":"2026-10-09T00:00:00+00:00","request_id":"concurrent",
                "source":"esp32","detection_count":0,"inference_ms":1,"server_processing_ms":2,
                "capture_ms":None,"image_size_bytes":100})
            def report(value):
                return store.report("concurrent",token,{"roundtrip_ms":value,"total_ms":value})
            with ThreadPoolExecutor(max_workers=2) as executor:
                results=list(executor.map(report,[10,20]))
            self.assertEqual(results[0],results[1])
            self.assertEqual(store.get("concurrent")["roundtrip_ms"],results[0]["roundtrip_ms"])


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = next((ROOT/"dataset/images/test").glob("*.png")).read_bytes()

    def test_real_image_manifest_and_no_labels(self):
        def handle(request):
            self.assertEqual(request.url.path,"/capture")
            return httpx.Response(200,content=self.sample,headers={"Content-Type":"image/png"})
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(transport=httpx.MockTransport(handle)) as client:
            record = collect_one(client,"http://camera.test",tmp)
            saved = Path(tmp)/record["image"]
            self.assertEqual(saved.read_bytes(),self.sample)
            self.assertEqual(record["sha256"],hashlib.sha256(self.sample).hexdigest())
            with Image.open(io.BytesIO(self.sample)) as image:
                self.assertEqual((record["width"],record["height"]),image.size)
            self.assertEqual(json.loads((Path(tmp)/"manifest.jsonl").read_text()),record)
            self.assertFalse(record["included_in_training"])
            self.assertEqual(record["label_status"],"unlabeled")
            self.assertFalse((Path(tmp)/"labels").exists())

    def test_capture_timeout_corruption_and_size(self):
        def timeout(request):
            raise httpx.ReadTimeout("mock timeout",request=request)
        with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
            with self.assertRaisesRegex(CameraError,"timed out"):
                capture(client,"http://camera.test")
        with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b"corrupt"))) as client:
            with self.assertRaises(CameraError):
                capture(client,"http://camera.test")
        with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=self.sample))) as client:
            with self.assertRaises(CameraError):
                capture(client,"http://camera.test",max_bytes=4)


class ModelContractTests(unittest.TestCase):
    def test_multi_box_conversion_and_empty_result(self):
        detector=MaskDetectionModel(Path("unused-unit-fixture.pt"),device="cpu")
        mocked = MagicMock()
        mocked.model.parameters.side_effect=lambda:iter([torch.zeros(1)])
        boxes=[]
        for cls in (0,1,2):
            box=MagicMock()
            box.cls=torch.tensor([cls])
            box.conf=torch.tensor([0.8])
            box.xyxy=torch.tensor([[1.,2.,20.,30.]])
            boxes.append(box)
        output=MagicMock()
        output.boxes=boxes
        mocked.predict.return_value=[output]
        detector._model=mocked
        prediction=detector.predict(np.zeros((64,64,3),dtype=np.uint8))
        self.assertEqual([d["class_id"] for d in prediction["detections"]],[0,1,2])
        self.assertEqual(prediction["detections"][0]["bbox"],{"x1":1.0,"y1":2.0,"x2":20.0,"y2":30.0})
        output.boxes=[]
        self.assertEqual(detector.predict(np.zeros((64,64,3),dtype=np.uint8))["detections"],[])

    def test_missing_corrupt_and_wrong_classes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"unit.pt"
            detector=MaskDetectionModel(path)
            with self.assertRaises(FileNotFoundError): detector.load_model()
            path.write_bytes(b"invalid weight file")
            with self.assertRaises(Exception): detector.load_model()
            self.assertFalse(detector.ready)
            wrong=MagicMock()
            wrong.names={0:"person"}
            with patch("ultralytics.YOLO",return_value=wrong):
                with self.assertRaisesRegex(ValueError,"classes mismatch"): detector.load_model()
            self.assertFalse(detector.ready)

    def test_publish_guards_preserve_good_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            best,last=root/"best.pt",root/"last.pt"
            best.write_bytes(b"unit-best"); last.write_bytes(b"unit-last")
            destination=root/"models"
            publish_checkpoints(best,last,{"epochs_completed":20},destination)
            with self.assertRaises(ValueError):
                publish_checkpoints(best,last,{"epochs_completed":1},destination,replace=True)
            best.write_bytes(b"new-unit-best")
            with self.assertRaises(FileExistsError):
                publish_checkpoints(best,last,{"epochs_completed":30},destination)
            self.assertEqual((destination/"face_mask_best.pt").read_bytes(),b"unit-best")
            publish_checkpoints(best,last,{"epochs_completed":30},destination,replace=True)
            archived=list((destination/"archive").glob("*/face_mask_best.pt"))
            self.assertEqual(len(archived),1)
            self.assertEqual(archived[0].read_bytes(),b"unit-best")


class UploadTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_body_timeout_408(self):
        from server import main
        called=False
        async def app(scope,receive,send):
            nonlocal called
            called=True
        async def receive():
            await asyncio.sleep(0.1)
            return {"type":"http.request","body":b"x"}
        messages=[]
        async def send(message): messages.append(message)
        settings=replace(main.settings,upload_timeout_seconds=0.01)
        with patch.object(main,"settings",settings):
            await main.UploadLimitMiddleware(app)({"type":"http","method":"POST","path":"/predict","headers":[]},receive,send)
        self.assertFalse(called)
        self.assertEqual(messages[0]["status"],408)


class TelegramTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls=[]
        self.tmp=tempfile.TemporaryDirectory()
        self.store=LatencyStore(Path(self.tmp.name)/"db.sqlite")
        self.result=event()
        self.image=MaskDetectionModel.annotate(np.zeros((32,32,3),dtype=np.uint8),self.result["detections"])
        def handle(request):
            self.calls.append(request)
            return httpx.Response(200,json={"ok":True,"result":{"id":1}})
        self.client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.config=TelegramConfig(enabled=True,token="UNIT_SECRET_DO_NOT_LOG",chat_id="123",
                                   cooldown=30,metrics_wait=0,commands=False)
        self.service=TelegramService(self.config,self.store,lambda:(self.result,self.image),self.client)

    async def asyncTearDown(self):
        await self.service.stop()
        await self.client.aclose()
        self.tmp.cleanup()

    async def test_disabled_and_missing_configuration(self):
        for config in (replace(self.config,enabled=False),replace(self.config,token="")):
            service=TelegramService(config,self.store,lambda:(None,None),self.client)
            await service.start()
            self.assertFalse(service.submit(self.result,self.image))
            self.assertIn(service.status()["state"],{"disabled","not_configured"})
            await service.stop()
        self.assertEqual(self.calls,[])

    async def test_annotated_sendphoto_cooldown_and_mode(self):
        await self.service.start()
        self.assertTrue(self.service.submit(self.result,self.image))
        self.assertFalse(self.service.submit(self.result,self.image))
        await asyncio.sleep(0.02)
        await self.service._queue.join()
        photos=[request for request in self.calls if request.url.path.endswith("/sendPhoto")]
        self.assertEqual(len(photos),1)
        body=await photos[0].aread()
        self.assertIn(self.image,body)
        self.assertIn(b'filename="annotated.jpg"',body)
        self.assertIn("Không đeo".encode(),body)
        self.assertIn("chưa đo".encode(),body)
        self.assertFalse(self.service.submit(event([]),self.image))
        self.assertEqual(self.service.status()["sent_count"],1)
        self.assertEqual(self.service.status()["skipped_count"],1)
        self.assertNotIn(self.config.token,json.dumps(self.service.status()))
        self.assertNotIn(self.config.token,repr(self.config))

    async def test_telegram_timeout_never_raises_from_submit(self):
        def timeout(request):
            if request.url.path.endswith("/getMe"):
                return httpx.Response(200,json={"ok":True,"result":{}})
            raise httpx.ReadTimeout("URL contains UNIT_SECRET_DO_NOT_LOG",request=request)
        await self.client.aclose()
        self.client=httpx.AsyncClient(transport=httpx.MockTransport(timeout))
        self.service=TelegramService(self.config,self.store,lambda:(None,None),self.client)
        with self.assertLogs("face_mask.telegram",level="WARNING") as logs:
            await self.service.start()
            self.assertTrue(self.service.submit(self.result,self.image))
            await asyncio.sleep(0.02)
            await self.service._queue.join()
        self.assertEqual(self.service.status()["state"],"error")
        self.assertNotIn(self.config.token,"".join(logs.output))

    async def test_manual_shots_bypass_violation_filter_and_cooldown_but_stay_bounded(self):
        await self.service.start()
        empty=event([]);empty['request_id']='manual-empty'
        proper=event([{'class_id':0,'confidence':.9}]);proper['request_id']='manual-proper'
        self.assertTrue(self.service.submit(empty,self.image,manual=True))
        self.assertTrue(self.service.submit(proper,self.image,manual=True))
        self.assertFalse(self.service.submit(empty,self.image,manual=True))
        await asyncio.sleep(.02)
        await self.service._queue.join()
        photos=[request for request in self.calls if request.url.path.endswith('/sendPhoto')]
        self.assertEqual(len(photos),2)
        for request in photos:self.assertIn(self.image,await request.aread())
        self.assertEqual(self.service.status()['last_sent_request_id'],'manual-proper')
        self.assertFalse(self.service.submit(self.result,self.image))

    async def test_client_report_enrichment(self):
        token=self.store.insert({"timestamp":self.result["timestamp"],"request_id":self.result["request_id"],
            "source":"esp32","detection_count":1,"inference_ms":12,"server_processing_ms":15,
            "capture_ms":3,"image_size_bytes":len(self.image)})
        self.store.report(self.result["request_id"],token,{"roundtrip_ms":20,"total_ms":23,"capture_ms":3})
        enriched=await self.service.enriched(self.result)
        self.assertEqual(enriched["end_to_end_ms"],23)
        self.assertIn("20.0 ms",self.service.caption(enriched))

    async def test_commands_authorized_chat_and_camera(self):
        sample=next((ROOT/"dataset/images/test").glob("*.png")).read_bytes()
        def handle(request):
            self.calls.append(request)
            if request.url.host=="camera.test":
                if request.url.path=="/capture":
                    return httpx.Response(200,content=sample)
                return httpx.Response(200,json={"flash":True})
            return httpx.Response(200,json={"ok":True,"result":{}})
        await self.client.aclose()
        self.client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
        seconds=[0]
        self.service=TelegramService(replace(self.config,camera_url="http://camera.test"),
                                     self.store,lambda:(self.result,self.image),self.client,clock=lambda:seconds[0])
        self.assertFalse(await self.service.handle_command({"chat":{"id":999},"text":"/flash"}))
        self.assertEqual(self.calls,[])
        for command in ("/start","/photo","/flash on","/result"):
            seconds[0]+=3
            self.assertTrue(await self.service.handle_command({"chat":{"id":123},"text":command}))
        camera_calls=[r for r in self.calls if r.url.host=="camera.test"]
        self.assertEqual([r.url.path for r in camera_calls],["/capture","/flash"])
        self.assertEqual(camera_calls[1].url.params["on"],"1")
        photos=[r for r in self.calls if r.url.path.endswith("/sendPhoto")]
        self.assertEqual(len(photos),2)
        self.assertIn(sample,await photos[0].aread())
        self.assertIn(self.image,await photos[1].aread())

class AdditionalApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server import main
        cls.main=main
        cls.client=TestClient(main.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None)


    def test_missing_checkpoint_api_503_and_null_history(self):
        sample=next((ROOT/"dataset/images/test").glob("*.png"))
        with patch.object(self.main.model,"_model",None):
            response=self.client.post("/predict",files={"file":(sample.name,sample.read_bytes(),"image/png")})
        self.assertEqual(response.status_code,503)
        row=next(r for r in self.client.get("/metrics/history").json()["items"]
                 if r["request_id"]==response.headers["X-Request-ID"])
        self.assertEqual(row["status"],"failure")
        self.assertIsNone(row["inference_ms"])
        self.assertIsNone(row["roundtrip_ms"])

    def test_telegram_enqueue_failure_keeps_real_prediction(self):
        if not self.main.model.ready:
            self.skipTest("TEST_MODEL_PATH required for real prediction")
        sample=next((ROOT/"dataset/images/test").glob("*.png"))
        with tempfile.TemporaryDirectory() as tmp:
            store=LatencyStore(Path(tmp)/"db.sqlite")
            telegram=MagicMock()
            telegram.submit.side_effect=RuntimeError("unit enqueue failure")
            with patch.object(self.main,"store",store),patch.object(self.main,"telegram",telegram), \
                 patch.object(self.main,"latest_result",None),patch.object(self.main,"latest_image",None):
                response=self.client.post("/predict",files={"file":(sample.name,sample.read_bytes(),"image/png")})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()["model_name"],"YOLOv8n Face Mask Detection")
                self.assertEqual(store.summary()["total_requests"],1)
                self.assertEqual(store.summary()["failed_requests"],0)

    def test_telegram_status_has_no_secrets(self):
        response=self.client.get("/telegram/status")
        self.assertEqual(response.status_code,200)
        value=response.json()
        self.assertEqual(value["state"],"disabled")
        self.assertNotIn("token",value)
        self.assertNotIn("chat_id",value)

    def test_api_key_required_only_when_configured(self):
        with patch.object(self.main,"settings",replace(self.main.settings,api_key="unit-api-key")):
            self.assertEqual(self.client.get("/health").status_code,200)
            self.assertEqual(self.client.get("/model/info").status_code,401)
            self.assertEqual(self.client.get("/telegram/status",headers={"X-API-Key":"wrong"}).status_code,401)
            self.assertEqual(self.client.get("/model/info",headers={"X-API-Key":"unit-api-key"}).status_code,200)

    def test_inference_failure_has_no_fake_measurements(self):
        if not self.main.model.ready:
            self.skipTest("TEST_MODEL_PATH required for inference error handler")
        sample=next((ROOT/"dataset/images/test").glob("*.png"))
        with patch.object(self.main.model,"predict",side_effect=TimeoutError("unit inference timeout")):
            response=self.client.post("/predict",files={"file":(sample.name,sample.read_bytes(),"image/png")})
        self.assertEqual(response.status_code,500)
        rows=self.client.get("/metrics/history").json()["items"]
        matching=[r for r in rows if r["request_id"]==response.headers.get("X-Request-ID")]
        # 500 body always carries the same request_id; older response path may have no header.
        if not matching:
            request_id=response.json()["detail"].split("request_id=")[1]
            matching=[r for r in rows if r["request_id"]==request_id]
        self.assertEqual(len(matching),1)
        self.assertEqual(matching[0]["status"],"failure")
        self.assertIsNone(matching[0]["inference_ms"])
        self.assertIsNone(matching[0]["end_to_end_ms"])
