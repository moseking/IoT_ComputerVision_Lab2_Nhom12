import json, unittest
from contextlib import redirect_stdout
from io import StringIO
from tests import test_combined_builder as fixtures
from scripts.build_baseline_dataset import build

class CameraTrainingOptInTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CombinedBuilderTests('test_both_sources_and_repeated_classes_preserve_holdouts')
        self.fixture.setUp()
        self.camera=self.fixture.review('esp32',21,[{'class_id':2,'bbox':[1,1,20,20]}])
    def tearDown(self):self.fixture.tearDown()
    def test_explicit_opt_in_keeps_human_review_and_holdouts_unchanged(self):
        original=(self.camera/'manifest.json').read_bytes()
        with redirect_stdout(StringIO()):
            result=build(self.fixture.public,[self.camera],self.fixture.root/'output',allow_camera_train=True)
        self.assertEqual((self.camera/'manifest.json').read_bytes(),original)
        self.assertTrue(result['camera_training_opt_in'])
        self.assertEqual(result['custom_accepted_by_source'],{'esp32':1})
        row=next(r for r in result['splits']['train'] if r['source_kind']=='esp32')
        self.assertTrue(row['human_reviewed']);self.assertEqual(row['class_counts'],{2:1})
        for split in ['val','test']:
            self.assertFalse(any(r['source_kind']=='esp32' for r in result['splits'][split]))
            self.assertEqual((self.fixture.public/'labels'/split/(split+'.txt')).read_bytes(),
                             (self.fixture.root/'output/labels'/split/('public_'+split+'.txt')).read_bytes())
    def test_opt_in_cannot_accept_unverified_camera_labels(self):
        path=self.camera/'manifest.json';manifest=json.loads(path.read_text())
        manifest['images'][0]['human_reviewed']=False;path.write_text(json.dumps(manifest))
        with redirect_stdout(StringIO()):
            result=build(self.fixture.public,[self.camera],self.fixture.root/'output',allow_camera_train=True)
        self.assertEqual(result['custom_accepted'],0)
        self.assertIn('provenance',result['excluded_custom'][0]['reason'].lower())

    def test_training_usage_registry_blocks_independent_camera_evaluation(self):
        from scripts.evaluate_camera import reviewed_rows
        import hashlib
        original=(self.camera/'manifest.json').read_bytes()
        (self.camera/'training_use.json').write_text(json.dumps({'role':'training_development',
            'manifest_sha256':hashlib.sha256(original).hexdigest()}))
        with self.assertRaisesRegex(ValueError,'training/development'):
            reviewed_rows(self.camera)
        self.assertEqual((self.camera/'manifest.json').read_bytes(),original)
