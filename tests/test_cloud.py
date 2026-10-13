"""Meaningful cloud contract checks; real ONNX runs locally, never sends Telegram."""
import asyncio
import base64
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from PIL import Image
from fastapi.testclient import TestClient
from server.cloud_bootstrap import ensure_model
from server.cloud_client import CloudConfig,CloudInferenceClient,CloudError
from server.inference_onnx import OnnxMaskDetectionModel,agnostic_nms
from server.latency_store import LatencyStore
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/cloud_v7/face_mask_v7.onnx'
SHA='f1fc04ca98bace9f1a1a5537d83fc9c52574514c0641959084c94a9f0750cbf4'
KEY='unit-cloud-test-secret-at-least-24'

class BootstrapTests(unittest.TestCase):
    def test_local_bundled_model_works_without_supabase(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.onnx';p.write_bytes(b'fixture-model');sha=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(ensure_model(p,sha,''),p)
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):ensure_model(p,'0'*64,'')
            self.assertEqual(p.read_bytes(),b'fixture-model')

    def test_download_is_verified_before_replace_and_secrets_do_not_leak(self):
        data=b'test-artifact';sha=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.onnx'
            with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=data))) as client:
                with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):ensure_model(p,'0'*64,'https://example.supabase.co/model?token=PRIVATE',client)
                self.assertFalse(p.exists());self.assertFalse(p.with_name('model.onnx.download').exists())
                ensure_model(p,sha,'https://example.supabase.co/model?token=PRIVATE',client)
                self.assertEqual(p.read_bytes(),data)
            p.unlink()
            def fail(r):raise httpx.ReadTimeout('https://secret.supabase.co/?token=PRIVATE',request=r)
            with httpx.Client(transport=httpx.MockTransport(fail)) as client:
                with self.assertRaises(RuntimeError) as error:ensure_model(p,sha,'https://example.supabase.co/?token=PRIVATE',client)
                self.assertNotIn('PRIVATE',str(error.exception))
            for url in ['http://example.supabase.co/model','https://other.example/model','https://a@b.supabase.co/model']:
                with self.assertRaises(ValueError):ensure_model(p,sha,url)

    def test_missing_pin_or_tampered_onnx_does_not_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.onnx';p.write_bytes(b'not-a-model')
            with self.assertRaisesRegex(ValueError,'pin'):ensure_model(p,'','')
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):OnnxMaskDetectionModel(p,expected_sha256='0'*64).load_model()

    def test_nms_suppresses_duplicate_face_across_classes(self):
        boxes=np.array([[10,10,50,50],[11,10,51,50],[70,10,100,40]],dtype=np.float32)
        self.assertEqual(agnostic_nms(boxes,np.array([.9,.8,.7]),.45),[0,2])

class CloudContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL.exists():raise unittest.SkipTest('Exported ONNX required')
        import server.cloud_app as cloud_app
        cls.api=cloud_app
        cls.key_patch=patch.object(cloud_app,'KEY',KEY);cls.key_patch.start()
        cls.path_patch=patch.object(cloud_app,'MODEL_PATH',MODEL);cls.path_patch.start()
        cls.sha_patch=patch.object(cloud_app,'SHA',SHA);cls.sha_patch.start()
        cls.model_patch=patch.object(cloud_app,'model',OnnxMaskDetectionModel(MODEL,expected_sha256=SHA,max_pixels=4000000));cls.model_patch.start()
        cls.api_client=TestClient(cloud_app.app);cls.api_client.__enter__()
        sample=next(p for p in (ROOT/'data/processed/facemask_camera_v7/images/val').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        cls.data=sample.read_bytes()
        with Image.open(io.BytesIO(cls.data)) as img:cls.width,cls.height=img.size
    @classmethod
    def tearDownClass(cls):
        cls.api_client.__exit__(None,None,None)
        for item in [cls.model_patch,cls.sha_patch,cls.path_patch,cls.key_patch]:item.stop()

    def proxy(self,mutate=None):
        def handle(request):
            response=self.api_client.request(request.method,request.url.path,headers=dict(request.headers),content=request.read())
            if mutate and request.url.path=='/predict' and response.status_code==200:
                body=response.json();mutate(body);return httpx.Response(200,json=body)
            return httpx.Response(response.status_code,headers=dict(response.headers),content=response.content)
        return httpx.Client(transport=httpx.MockTransport(handle))

    def test_real_cloud_api_auth_limits_and_valid_onnx_frame(self):
        c=self.api_client
        self.assertTrue(c.get('/health').json()['model_loaded'])
        self.assertEqual(c.get('/model/info').status_code,401)
        self.assertEqual(c.post('/predict',content=b'x'*(6*1024*1024)).status_code,401)
        headers={'X-API-Key':KEY}
        self.assertEqual(c.post('/predict',headers=headers,content=b'x'*(6*1024*1024)).status_code,413)
        self.assertEqual(c.post('/predict',headers=headers,files={'file':('bad.jpg',b'invalid','image/jpeg')}).status_code,400)
        response=c.post('/predict',headers=headers,files={'file':('face.jpg',self.data,'image/jpeg')})
        self.assertEqual(response.status_code,200)
        result=response.json();self.assertEqual(result['model_sha256'],SHA)
        self.assertEqual(result['input_sha256'],hashlib.sha256(self.data).hexdigest())
        self.assertEqual(result['detection_count'],len(result['detections']))
        with Image.open(io.BytesIO(base64.b64decode(result['annotated_image_base64']))) as image:
            self.assertEqual(image.size,(self.width,self.height))

    def test_cloud_busy_rejects_second_body_without_buffering_it(self):
        async def scenario():
            entered=asyncio.Event();release=asyncio.Event()
            async def app(scope,receive,send):
                entered.set();await release.wait()
                await send({'type':'http.response.start','status':200,'headers':[]})
                await send({'type':'http.response.body','body':b'ok'})
            middleware=self.api.BodyLimit(app)
            scope={'type':'http','method':'POST','headers':[(b'x-api-key',KEY.encode())]}
            async def body():return {'type':'http.request','body':b'x','more_body':False}
            messages=[]
            async def send(message):messages.append(message)
            task=asyncio.create_task(middleware(scope,body,send));await entered.wait()
            async def forbidden():raise AssertionError('Busy request body must not be read')
            await middleware(scope,forbidden,send)
            self.assertEqual(messages[0]['status'],429)
            release.set();await task
        asyncio.run(scenario())

    def test_real_gateway_rejects_mismatched_artifact_frame_box_and_timing(self):
        mutations=[lambda b:b.update(model_sha256='0'*64),lambda b:b.update(input_sha256='0'*64),
                   lambda b:b.update(width=-1),lambda b:b.update(runtime='pytorch'),
                   lambda b:b.update(detections=[{'class_id':1,'class_name':'without_mask','confidence':.5,
                                                'bbox':{'x1':-1,'y1':0,'x2':10,'y2':10}}],detection_count=1),
                   lambda b:b.update(queue_ms=float('inf')),
                   lambda b:b.update(annotated_image_base64=base64.b64encode(b'invalid').decode())]
        for mutate in mutations:
            with self.subTest(mutate=mutate),self.proxy(mutate) as transport:
                gateway=CloudInferenceClient(CloudConfig('https://unit.onrender.com',KEY,SHA),transport)
                with self.assertRaises(CloudError):gateway.predict(self.data,self.width,self.height)
                self.assertTrue(gateway.status()['last_error'])

    def test_real_gateway_success_checks_identity_and_full_response(self):
        with self.proxy() as transport:
            gateway=CloudInferenceClient(CloudConfig('https://unit.onrender.com',KEY,SHA),transport)
            result,annotation=gateway.predict(self.data,self.width,self.height)
            self.assertTrue(gateway.status()['verified'])
            self.assertGreaterEqual(result['cloud_roundtrip_ms'],result['server_processing_ms'])
            self.assertEqual(annotation,base64.b64decode(result['annotated_image_base64']))

    def test_gateway_main_routes_never_run_local_model_for_cloud(self):
        from tests import test_project
        import server.main as main
        with tempfile.TemporaryDirectory() as tmp,self.proxy() as transport:
            gateway=CloudInferenceClient(CloudConfig('https://unit.onrender.com',KEY,SHA),transport)
            with TestClient(main.app) as client:
                with patch.object(main,'cloud',gateway),patch.object(main,'store',LatencyStore(Path(tmp)/'latency.db')), \
                     patch.object(main.model,'predict',side_effect=AssertionError('Local YOLO must not run')), \
                     patch.object(main.camera_client,'capture',return_value=(self.data,(self.width,self.height,'JPEG'))), \
                     patch.object(main.telegram,'submit',return_value=True) as submit:
                    r=client.post('/camera/predict',json={'camera_url':'http://192.168.137.188','inference_target':'cloud','send_telegram':True})
                    self.assertEqual(r.status_code,200,r.text);b=r.json()
                    self.assertEqual(b['inference_location'],'cloud');self.assertEqual(b['runtime'],'onnxruntime')
                    self.assertEqual(b['original_image_sha256'],b['input_sha256'] if 'input_sha256' in b else hashlib.sha256(self.data).hexdigest())
                    submit.assert_called_once();self.assertEqual(submit.call_args.kwargs,{'manual':True})
                    self.assertEqual(submit.call_args.args[1],base64.b64decode(b['annotated_image_base64']))
                    measurements={'request_id':b['request_id'],'report_token':b['report_token'],'roundtrip_ms':0,
                                  'total_ms':b['gateway_processing_ms']+100,'capture_ms':None}
                    reported=client.post('/metrics/report',json=measurements)
                    self.assertEqual(reported.status_code,200,reported.text)
                    row=reported.json()['measurement']
                    self.assertEqual(row['roundtrip_ms'],b['cloud_roundtrip_ms'])
                    self.assertEqual(row['inference_location'],'cloud')
                    self.assertEqual(client.get('/config').json()['cloud_configured'],True)
                    self.assertNotIn(KEY,client.get('/config').text)
                    offline=client.post('/predict',data={'inference_target':'cloud','include_image':'true'},files={'file':('frame.jpg',self.data,'image/jpeg')})
                    self.assertEqual(offline.status_code,200,offline.text)

    def test_gateway_failure_has_no_fallback_or_latest_replacement(self):
        from tests import test_project
        import server.main as main
        bad=CloudInferenceClient(CloudConfig('https://unit.onrender.com',KEY,SHA),httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(503))))
        with tempfile.TemporaryDirectory() as tmp,TestClient(main.app) as client:
            with patch.object(main,'cloud',bad),patch.object(main,'store',LatencyStore(Path(tmp)/'latency.db')),patch.object(main.model,'predict') as local:
                response=client.post('/predict',data={'inference_target':'cloud'},files={'file':('frame.jpg',self.data,'image/jpeg')})
                self.assertEqual(response.status_code,502);local.assert_not_called()
                self.assertIsNone(client.get('/results/latest').json()['result'])
                self.assertEqual(main.store.history()[0]['inference_location'],'cloud')
                self.assertFalse(bad.status()['verified']);self.assertTrue(bad.status()['last_error'])
            with patch.object(main,'cloud',CloudInferenceClient(CloudConfig())):
                response=client.post('/predict',data={'inference_target':'cloud'},files={'file':('frame.jpg',self.data,'image/jpeg')})
                self.assertEqual(response.status_code,503)
        bad.client.close()

class LatencyMigrationTests(unittest.TestCase):
    def test_old_rows_preserved_and_cloud_rtt_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.db'
            # First create the pre-cloud schema, including a measured legacy record.
            with sqlite3.connect(path) as db:
                db.execute('''CREATE TABLE latency_history(id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT NOT NULL,request_id TEXT UNIQUE NOT NULL,source TEXT NOT NULL,
                  detection_count INTEGER NOT NULL,inference_ms REAL NOT NULL,server_processing_ms REAL NOT NULL,
                  roundtrip_ms REAL,total_ms REAL,capture_ms REAL,upload_ms REAL,download_ms REAL,
                  network_overhead_estimated_ms REAL,image_size_bytes INTEGER NOT NULL,report_token TEXT NOT NULL,
                  reported INTEGER NOT NULL DEFAULT 0)''')
                db.execute("INSERT INTO latency_history(timestamp,request_id,source,detection_count,inference_ms,server_processing_ms,roundtrip_ms,total_ms,image_size_bytes,report_token,reported) VALUES ('2026-10-12','old','dashboard',1,20,30,50,60,100,'old-token',1)")
            db.close()
            store=LatencyStore(path);self.assertEqual(store.get('old')['roundtrip_ms'],50)
            self.assertEqual(store.get('old')['inference_location'],'local')
            token=store.insert({'timestamp':'2026-10-12','request_id':'cloud','source':'dashboard','detection_count':1,
                                'inference_ms':100,'server_processing_ms':120,'image_size_bytes':200,
                                'inference_location':'cloud','cloud_roundtrip_ms':400,'cloud_model_sha256':SHA})
            row=store.report('cloud',token,{'roundtrip_ms':9999,'total_ms':500})
            self.assertEqual(row['roundtrip_ms'],400);self.assertEqual(row['network_overhead_estimated_ms'],280)
            self.assertEqual(store.report('cloud',token,{'roundtrip_ms':1,'total_ms':2})['total_ms'],500)

if __name__=='__main__':unittest.main()
