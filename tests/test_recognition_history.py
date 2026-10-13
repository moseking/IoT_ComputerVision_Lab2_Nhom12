"""Regression coverage for persisted mixed labels, legacy data and exact API counts."""
import base64
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from server.latency_store import LatencyStore,recognition_counts

ROOT=Path(__file__).resolve().parents[1]

def record(request_id,counts=None,faces=4):
    value={'timestamp':'2026-10-12T00:00:00+00:00','request_id':request_id,'source':'camera_pull',
           'detection_count':faces,'inference_ms':10,'server_processing_ms':20,'image_size_bytes':100}
    if counts is not None:value.update(counts)
    return value

class RecognitionStoreTests(unittest.TestCase):
    def test_mixed_and_empty_requests_survive_restart_without_double_counting_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.db';store=LatencyStore(path)
            counts=recognition_counts([{'class_id':0},{'class_id':0},{'class_id':1},{'class_id':2}])
            token=store.insert(record('mixed',counts))
            store.insert(record('empty',recognition_counts([]),0))
            store.insert(record('legacy'))
            store.record_failure('failed','camera_pull',502,'camera_timeout')
            for _ in range(2):store.report('mixed',token,{'roundtrip_ms':30,'total_ms':40})
            store=LatencyStore(path);row=store.get('mixed')
            self.assertEqual(row['class_counts'],{'with_mask':2,'without_mask':1,'mask_weared_incorrect':1})
            self.assertEqual(row['total_ms'],40);self.assertNotIn('report_token',row)
            summary=store.summary()['recognition']
            self.assertEqual(summary['labeled_requests'],2);self.assertEqual(summary['unlabeled_requests'],1)
            self.assertEqual(summary['total_faces'],4)
            self.assertEqual(summary['classes']['with_mask'],{'detections':2,'requests':1})
            self.assertEqual(summary['classes']['without_mask'],{'detections':1,'requests':1})
            self.assertIsNone(store.get('legacy')['class_counts'])
            failure=next(row for row in store.history() if row['status']=='failure')
            self.assertIsNone(failure['class_counts']);self.assertIsNone(failure['with_mask_count'])

    def test_existing_database_gets_nullable_counts_without_relabeling_old_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.db';store=LatencyStore(path)
            store.insert(record('old'))
            with sqlite3.connect(path) as db:
                for column in ('with_mask_count','without_mask_count','incorrect_mask_count'):
                    db.execute('ALTER TABLE latency_history DROP COLUMN '+column)
            db.close()
            migrated=LatencyStore(path);self.assertEqual(migrated.get('old')['detection_count'],4)
            self.assertIsNone(migrated.get('old')['class_counts'])
            self.assertEqual(migrated.summary()['recognition']['labeled_requests'],0)
            self.assertEqual(migrated.summary()['recognition']['unlabeled_requests'],1)
            self.assertEqual(migrated.summary()['recognition']['classes']['mask_weared_incorrect']['detections'],0)

    def test_inconsistent_or_partial_counts_are_rejected_before_insert(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=LatencyStore(Path(tmp)/'db.sqlite')
            for counts in [{'with_mask_count':1}, {'with_mask_count':2,'without_mask_count':1,'incorrect_mask_count':0},
                           {'with_mask_count':4,'without_mask_count':-1,'incorrect_mask_count':1},
                           {'with_mask_count':True,'without_mask_count':2,'incorrect_mask_count':1}]:
                with self.subTest(counts=counts),self.assertRaises(ValueError):store.insert(record('bad',counts))
            self.assertEqual(store.summary()['total_requests'],0)
            with self.assertRaises(ValueError):recognition_counts([{'class_id':3}])

class RecognitionApiTests(unittest.TestCase):
    def test_saved_labels_match_real_inference_and_metrics_retry_does_not_change_them(self):
        from tests import test_project
        import server.main as main
        with tempfile.TemporaryDirectory() as tmp,TestClient(main.app) as client:
            if not main.model.ready:self.skipTest('TEST_MODEL_PATH required')
            with patch.object(main,'store',LatencyStore(Path(tmp)/'db.sqlite')):
                sample=next((ROOT/'dataset/images/test').glob('*.png')).read_bytes()
                response=client.post('/predict',files={'file':('face.png',sample,'image/png')})
                self.assertEqual(response.status_code,200,response.text);result=response.json()
                expected=recognition_counts(result['detections'])
                for key,value in expected.items():self.assertEqual(result[key],value)
                stored=main.store.get(result['request_id']);self.assertEqual(stored['class_counts'],result['class_counts'])
                self.assertEqual(sum(stored['class_counts'].values()),result['detection_count'])
                metrics={'request_id':result['request_id'],'report_token':result['report_token'],'roundtrip_ms':10000,'total_ms':10000}
                for _ in range(2):self.assertEqual(client.post('/metrics/report',json=metrics).status_code,200)
                self.assertEqual(client.get('/metrics/summary').json()['recognition']['total_faces'],result['detection_count'])
                item=client.get('/metrics/history').json()['items'][0]
                self.assertEqual(item['class_counts'],result['class_counts']);self.assertNotIn('report_token',item)

if __name__=='__main__':unittest.main()
