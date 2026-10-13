"""Temporary capture fixtures: delta import and human-review boundary."""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from scripts.prepare_new_review import initialize
from scripts.custom_dataset import prepare
from scripts.review_annotations import create_app


class EmptyDetector:
    def setInputSize(self, size): pass
    def detect(self, image): return None, None


class NewReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'raw/single_person';self.class_folder=self.source/'wore_incorrect'
        self.class_folder.mkdir(parents=True)
        self.output=self.root/'review';self.known=self.root/'known';self.known.mkdir()
        self.detector=self.root/'detector.onnx';self.detector.write_bytes(b'fixture detector')
        self.mock=patch('scripts.custom_dataset.cv2.FaceDetectorYN.create',return_value=EmptyDetector())
        self.mock.start()
        (self.known/'manifest.json').write_text(json.dumps({'images':[]}),encoding='utf-8')
    def tearDown(self):self.mock.stop();self.tmp.cleanup()
    def initialize(self):
        return initialize(self.source,self.output,[self.known],self.detector)
    def test_only_new_images_become_drafts_without_changing_old_reviews(self):
        old=self.class_folder/'old.png';Image.new('RGB',(32,24),'red').save(old)
        previous={'images':[{'source_file':str(old),'source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
                            'label_status':'verified','human_reviewed':True}]}
        (self.known/'manifest.json').write_text(json.dumps(previous),encoding='utf-8')
        before=(self.known/'manifest.json').read_bytes()
        (self.class_folder/'duplicate.png').write_bytes(old.read_bytes())
        fresh=self.class_folder/'left_ear.png';Image.new('RGB',(32,24),'blue').save(fresh)
        result=self.initialize();self.assertEqual(result['imported'],1)
        manifest=json.loads((self.output/'manifest.json').read_text())
        row=manifest['images'][0]
        self.assertEqual(row['source_file'],str(fresh))
        self.assertEqual(row['intended_class_id'],2)
        self.assertEqual(row['label_status'],'draft');self.assertFalse(row['human_reviewed'])
        self.assertFalse(row['included_in_training'])
        self.assertEqual(before,(self.known/'manifest.json').read_bytes())
    def test_empty_queue_can_import_later_and_requires_page_token(self):
        self.assertEqual(self.initialize()['total'],0)
        with TestClient(create_app(self.output)) as client:
            html=client.get('/').text
            self.assertIn('0 ảnh mới',html)
            token=re.search(r'const token="([a-f0-9]+)"',html).group(1)
            fresh=self.class_folder/'new.png';Image.new('RGB',(40,30),'green').save(fresh)
            self.assertEqual(client.post('/api/import-new').status_code,403)
            headers={'X-Review-Token':token}
            result=client.post('/api/import-new',headers=headers)
            self.assertEqual(result.status_code,200);self.assertEqual(result.json()['imported'],1)
            self.assertEqual(client.post('/api/import-new',headers=headers).json()['imported'],0)
            rows=client.get('/api/manifest').json()['images']
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['label_status'],'draft')
            self.assertEqual(client.get('/image/'+rows[0]['id']).status_code,200)
            (self.output/'training_use.json').write_text('{}')
            self.assertEqual(client.post('/api/import-new',headers=headers).status_code,409)
    def test_selected_source_cannot_escape_capture_folder(self):
        outside=self.root/'outside.png';Image.new('RGB',(20,20)).save(outside)
        with self.assertRaisesRegex(ValueError,'inside configured source'):
            prepare(self.source,self.output,self.detector,source_paths=[outside])


if __name__=='__main__':unittest.main()
