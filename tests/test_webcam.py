"""Browser-camera upload contract and manual Telegram policy, with isolated fake inference."""
import io, unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from tests.test_project import app, model
from server import main

class WebcamTests(unittest.TestCase):
    def test_webcam_local_upload_records_source_capture_and_manual_telegram_even_without_faces(self):
        image=io.BytesIO();Image.new('RGB',(80,60),'white').save(image,format='JPEG')
        prediction={'detections':[],'inference_ms':1.0,'queue_ms':0.0,'inference_definition':'test'}
        with TestClient(app) as client, patch.object(model,'_model',object()), patch.object(model,'predict',return_value=prediction), patch.object(main.telegram,'submit',return_value=True) as submit:
            response=client.post('/predict',data={'source':'webcam','capture_ms':'2','include_image':'true','send_telegram':'true'},files={'file':('webcam.jpg',image.getvalue(),'image/jpeg')})
            self.assertEqual(response.status_code,200)
            result=response.json();self.assertEqual(result['source'],'webcam');self.assertEqual(result['capture_ms'],2)
            self.assertEqual(result['inference_location'],'local');self.assertEqual(result['detection_count'],0)
            self.assertTrue(result['annotated_image_base64']);self.assertTrue(submit.call_args.kwargs['manual'])
            saved=main.store.get(result['request_id']);self.assertEqual(saved['source'],'webcam')
            report=client.post('/metrics/report',json={'request_id':result['request_id'],'report_token':result['report_token'],'capture_ms':2,'roundtrip_ms':5,'total_ms':8})
            self.assertEqual(report.status_code,200)
            self.assertIn('Webcam laptop',main.telegram.caption(result))

    def test_webcam_live_keeps_automatic_telegram_policy(self):
        image=io.BytesIO();Image.new('RGB',(80,60),'white').save(image,format='JPEG')
        with TestClient(app) as client, patch.object(model,'_model',object()), patch.object(model,'predict',return_value={'detections':[],'inference_ms':1,'queue_ms':0,'inference_definition':'test'}), patch.object(main.telegram,'submit',return_value=False) as submit:
            response=client.post('/predict',data={'source':'webcam','send_telegram':'false'},files={'file':('webcam.jpg',image.getvalue(),'image/jpeg')})
            self.assertEqual(response.status_code,200);self.assertNotIn('manual',submit.call_args.kwargs)

    def test_webcam_source_is_still_checked_and_bad_image_is_rejected(self):
        with TestClient(app) as client, patch.object(model,'_model',object()):
            response=client.post('/predict',data={'source':'webcam'},files={'file':('webcam.jpg',b'not an image','image/jpeg')})
            self.assertEqual(response.status_code,400)
            response=client.post('/predict',data={'source':'unknown-camera'},files={'file':('webcam.jpg',b'x','image/jpeg')})
            self.assertEqual(response.status_code,422)



class CloudDashboardConfigTests(unittest.TestCase):
    def test_gcp_preferences_are_public_but_cloud_key_is_not(self):
        from dataclasses import replace
        from server.cloud_client import CloudConfig, CloudInferenceClient
        key='unit-gcp-key-never-exposed-to-browser'
        gateway=CloudInferenceClient(CloudConfig('https://unit-gcp.run.app',key,'f'*64))
        prefs=replace(main.settings,default_inference_target='cloud',default_camera_source='webcam')
        with patch.object(main,'settings',prefs),patch.object(main,'cloud',gateway),TestClient(app) as client:
            response=client.get('/config');config=response.json()
            self.assertTrue(config['cloud_configured'])
            self.assertEqual(config['cloud_provider'],'Google Cloud Run')
            self.assertEqual(config['default_inference_target'],'cloud')
            self.assertEqual(config['default_camera_source'],'webcam')
            self.assertNotIn(key,response.text)
            self.assertNotIn(key,client.get('/static/app.js').text)

    def test_render_profile_keeps_its_actual_provider_name(self):
        from server.cloud_client import CloudConfig, CloudInferenceClient
        gateway=CloudInferenceClient(CloudConfig('https://unit.onrender.com','unit-render-key-long-enough','f'*64))
        with patch.object(main,'cloud',gateway),TestClient(app) as client:
            self.assertEqual(client.get('/config').json()['cloud_provider'],'Render')

if __name__=='__main__':unittest.main()
