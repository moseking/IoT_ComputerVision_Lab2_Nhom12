import base64,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import httpx
from fastapi.testclient import TestClient
from server.camera_client import CameraError
from server.camera_control import lan_camera_url,flash


class CameraControlTests(unittest.TestCase):
    def test_lan_camera_url_rejects_non_camera_targets(self):
        self.assertEqual(lan_camera_url(' http://192.168.137.28/ '),'http://192.168.137.28')
        for url in ['http://127.0.0.1','http://169.254.169.254','http://8.8.8.8','http://0.0.0.0',
                    'http://192.0.2.1','http://camera.example','http://x@y@192.168.137.28',
                    'http://192.168.137.28/capture','http://192.168.137.28?on=1']:
            with self.subTest(url=url),self.assertRaises(CameraError):lan_camera_url(url)

    def test_flash_sets_explicit_state_and_confirms_response(self):
        seen=[]
        def handle(request):
            seen.append(str(request.url));return httpx.Response(200,json={'flash':request.url.params['on']=='1'})
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            self.assertEqual(flash(client,'http://192.168.137.28',True),{'flash':True,'confirmed':True})
            self.assertEqual(flash(client,'http://192.168.137.28',True),{'flash':True,'confirmed':True})
            self.assertEqual(flash(client,'http://192.168.137.28',False),{'flash':False,'confirmed':True})
        self.assertEqual(seen,['http://192.168.137.28/flash?on=1']*2+['http://192.168.137.28/flash?on=0'])

    def test_flash_does_not_claim_confirmation_for_bad_response(self):
        responses=[httpx.Response(404),httpx.Response(302,headers={'location':'http://127.0.0.1'}),
            httpx.Response(200,json={'flash':False}),httpx.Response(200,json={'flash':1}),
            httpx.Response(200,json={}),httpx.Response(200,content=b'invalid'),httpx.Response(200,content=b'x'*5000)]
        for response in responses:
            with self.subTest(response=response.status_code),httpx.Client(transport=httpx.MockTransport(lambda r:response)) as client:
                with self.assertRaises(CameraError):flash(client,'http://192.168.137.28',True)

    def test_flash_timeout_leaves_state_unconfirmed(self):
        def handle(request):raise httpx.ReadTimeout('test timeout',request=request)
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaisesRegex(CameraError,'timed out'):flash(client,'http://192.168.137.28',True)


class CameraApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import test_project
        import server.main as main
        cls.main=main
        cls.client=TestClient(main.app);cls.client.__enter__()
        cls.tmp=tempfile.TemporaryDirectory(prefix='camera-control-api-tests-')
        cls.previous_store=main.store
        main.store=main.LatencyStore(Path(cls.tmp.name)/'latency.db')
    @classmethod
    def tearDownClass(cls):
        cls.main.store=cls.previous_store
        cls.client.__exit__(None,None,None)
        cls.tmp.cleanup()

    def test_flash_and_live_require_existing_api_auth(self):
        with patch.object(self.main,'settings',replace(self.main.settings,api_key='unit-camera-key')):
            for path in ['/camera/flash','/camera/predict']:
                response=self.client.post(path,json={'camera_url':'http://192.168.137.28','enabled':True})
                self.assertEqual(response.status_code,401)

    def test_flash_invalid_target_and_non_boolean_have_no_side_effect(self):
        with patch.object(self.main.camera_control,'flash') as command:
            self.assertEqual(self.client.post('/camera/flash',json={'camera_url':'http://127.0.0.1','enabled':True}).status_code,422)
            self.assertEqual(self.client.post('/camera/flash',json={'camera_url':'http://192.168.137.28','enabled':'true'}).status_code,422)
            command.assert_not_called()

    def test_flash_error_is_not_a_successful_toggle(self):
        with patch.object(self.main.camera_control,'flash',side_effect=CameraError('Camera timeout')):
            response=self.client.post('/camera/flash',json={'camera_url':'http://192.168.137.28','enabled':True})
        self.assertEqual(response.status_code,502)
        self.assertIn('timeout',response.json()['detail'])

    def test_camera_pull_real_model_preserves_source_and_unknown_device_capture(self):
        if not self.main.model.ready:self.skipTest('TEST_MODEL_PATH required')
        sample=next((Path(__file__).resolve().parents[1]/'dataset/images/test').glob('*.png')).read_bytes()
        with patch.object(self.main.camera_client,'capture',return_value=(sample,(640,480,'PNG'))):
            response=self.client.post('/camera/predict',json={'camera_url':'http://192.168.137.28'})
        self.assertEqual(response.status_code,200)
        body=response.json();self.assertEqual(body['source'],'camera_pull')
        self.assertIsNone(body['capture_ms']);self.assertIn('annotated_image_base64',body)
        self.assertEqual(base64.b64decode(body['original_image_base64']),sample)
        self.assertEqual(body['original_image_sha256'],hashlib.sha256(sample).hexdigest())
        self.assertEqual(body['original_image_content_type'],'image/png')
        self.assertTrue(body['original_image_filename'].endswith('.png'))
        self.assertGreaterEqual(body['camera_fetch_ms'],0)
        self.assertEqual(int(response.headers['content-length']),len(response.content))
        self.assertEqual(self.client.get('/results/latest').json()['result']['request_id'],body['request_id'])

    def test_manual_camera_telegram_sends_the_same_prediction_once(self):
        if not self.main.model.ready:self.skipTest('TEST_MODEL_PATH required')
        sample=next((Path(__file__).resolve().parents[1]/'dataset/images/test').glob('*.png')).read_bytes()
        with patch.object(self.main.camera_client,'capture',return_value=(sample,(640,480,'PNG'))), \
             patch.object(self.main.telegram,'submit',return_value=True) as submit:
            response=self.client.post('/camera/predict',json={'camera_url':'http://192.168.137.28','send_telegram':True})
        self.assertEqual(response.status_code,200);body=response.json()
        submit.assert_called_once()
        args,kwargs=submit.call_args
        self.assertEqual(kwargs,{'manual':True})
        self.assertEqual(args[0]['request_id'],body['request_id'])
        self.assertEqual(args[1],base64.b64decode(body['annotated_image_base64']))
        self.assertEqual(body['telegram_delivery']['state'],'queued')

    def test_manual_telegram_flag_is_strict_and_missing_flag_stays_automatic(self):
        with patch.object(self.main.camera_client,'capture') as capture:
            response=self.client.post('/camera/predict',json={'camera_url':'http://192.168.137.28','send_telegram':'true'})
        self.assertEqual(response.status_code,422);capture.assert_not_called()

    def test_camera_capture_error_has_no_fallback_prediction(self):
        if not self.main.model.ready:self.skipTest('TEST_MODEL_PATH required')
        with patch.object(self.main.camera_client,'capture',side_effect=CameraError('Camera unreachable')),patch.object(self.main.model,'predict') as predict:
            response=self.client.post('/camera/predict',json={'camera_url':'http://192.168.137.28'})
        self.assertEqual(response.status_code,502);predict.assert_not_called()
