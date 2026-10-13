"""Temporary fixtures only: conversion/review gates and class-aware error counting."""
import json, re, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from scripts.custom_dataset import standardize, validate_boxes, face_proposals, FOLDER_IDS
from scripts.review_annotations import create_app
from scripts.evaluate import match_errors, overlap

class CustomConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_alias_fixed_mapping(self):
        self.assertEqual(FOLDER_IDS["wore_incorrect"],2)
        self.assertEqual(FOLDER_IDS["without_mask"],1)
        self.assertEqual(FOLDER_IDS["with_mask"],0)
    def test_jpeg_exif_orientation_and_source_unchanged(self):
        path=self.root/"phone.jpg"; exif=Image.Exif();exif[274]=6
        Image.new("RGB",(30,20)).save(path,exif=exif); before=path.read_bytes()
        image,meta=standardize(path)
        self.assertEqual(image.size,(20,30));self.assertEqual(meta["exif_orientation"],6)
        self.assertEqual(path.read_bytes(),before)
    def test_actual_heic_codec_conversion(self):
        import pillow_heif
        path=self.root/"phone.heic"
        pillow_heif.from_pillow(Image.new("RGB",(40,32),"red")).save(path)
        image,meta=standardize(path)
        self.assertEqual(image.size,(40,32));self.assertEqual(meta["processed_size"],[40,32])
    def test_corrupt_heic_rejected(self):
        path=self.root/"bad.heic";path.write_bytes(b"not a HEIF file")
        with self.assertRaises((ValueError,OSError)):standardize(path)
    def test_missing_heic_dependency_explained(self):
        path=self.root/"bad.heic";path.write_bytes(b"test")
        with patch.dict("sys.modules",{"pillow_heif":None}):
            with self.assertRaisesRegex(ValueError,"pillow-heif"):standardize(path)
    def test_downsample_keeps_valid_dimensions(self):
        path=self.root/"big.png";Image.new("RGB",(200,100)).save(path)
        image,meta=standardize(path,80);self.assertEqual(image.size,(80,40))
    def test_box_validation_bounds_mapping_and_single_face(self):
        valid=[{"class_id":2,"bbox":[1,2,10,12]}]
        self.assertEqual(validate_boxes(valid,20,30),valid)
        for boxes in ([],valid*2,[{"class_id":3,"bbox":[1,2,10,12]}],[{"class_id":True,"bbox":[1,2,10,12]}],
                [{"class_id":0,"bbox":[-1,0,10,12]}],[{"class_id":0,"bbox":[1,2,float("nan"),12]}],
                [{"class_id":0,"bbox":[10,2,1,12]}]):
            with self.assertRaises(ValueError):validate_boxes(boxes,20,30)
    def test_multi_person_requires_each_box_class(self):
        boxes=[{"class_id":0,"bbox":[0,0,10,10]},{"class_id":1,"bbox":[11,0,20,10]}]
        self.assertEqual(len(validate_boxes(boxes,20,20,False)),2)
    def test_detector_outputs_proposals_without_verification(self):
        class Detector:
            def setInputSize(self,size):self.size=size
            def detect(self,image):return None,np.array([[-2,2,20,30]+[0]*10+[.8]],dtype=np.float32)
        detector=Detector(); proposals=face_proposals(Image.new("RGB",(40,40)),detector)
        self.assertTrue(proposals[0]["proposal_clamped"])
        self.assertNotIn("label_status",proposals[0])

class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        (self.folder/"images").mkdir();Image.new("RGB",(30,20)).save(self.folder/"images/a.jpg")
        self.manifest={"single_person":True,"mapping":{0:"with_mask",1:"without_mask",2:"mask_weared_incorrect"},
            "statistics":{"verified_images":0},"images":[{"id":"a","processed_file":"images/a.jpg","width":30,"height":20,
            "label_status":"draft","boxes":[{"class_id":0,"bbox":[1,2,10,12]}]}]}
        (self.folder/"manifest.json").write_text(json.dumps(self.manifest),encoding="utf-8")
        self.client=TestClient(create_app(self.folder))
        self.token=re.search(r'const token="([a-f0-9]+)"',self.client.get("/").text).group(1)
        self.headers={"X-Review-Token":self.token}
    def tearDown(self):self.client.close();self.tmp.cleanup()
    def request(self,**extra):
        data={"action":"accept","reviewer":"unit-test fixture reviewer","boxes":[{"class_id":2,"bbox":[1,2,10,12]}]}
        data.update(extra);return self.client.post("/api/review/a",json=data,headers=self.headers)
    def test_no_token_no_write(self):
        before=(self.folder/"manifest.json").read_bytes()
        self.assertEqual(self.client.post("/api/review/a",json={"action":"accept"}).status_code,403)
        self.assertEqual((self.folder/"manifest.json").read_bytes(),before)
    def test_no_reviewer_blocks_acceptance(self):
        self.assertEqual(self.request(reviewer="").status_code,400)
    def test_explicit_fixture_review_records_provenance(self):
        self.assertEqual(self.request().status_code,200)
        row=json.loads((self.folder/"manifest.json").read_text())["images"][0]
        self.assertTrue(row["human_reviewed"]);self.assertEqual(row["label_status"],"verified")
        self.assertEqual(row["boxes"][0]["class_id"],2);self.assertFalse(row["included_in_training"])
    def test_bad_box_cannot_be_verified(self):
        self.assertEqual(self.request(boxes=[{"class_id":0,"bbox":[0,0,31,20]}]).status_code,400)
    def test_draft_not_human_verified(self):
        self.assertEqual(self.request(action="draft").status_code,200)
        row=json.loads((self.folder/"manifest.json").read_text())["images"][0]
        self.assertFalse(row["human_reviewed"]);self.assertEqual(row["label_status"],"draft")
    def test_extra_detector_faces_cannot_be_silently_dropped(self):
        manifest=json.loads((self.folder/"manifest.json").read_text())
        manifest["images"][0]["face_count"]=2
        (self.folder/"manifest.json").write_text(json.dumps(manifest))
        with TestClient(create_app(self.folder)) as client:
            self.assertEqual(client.post("/api/review/a",json={"action":"accept","reviewer":"fixture",
                "boxes":[{"class_id":0,"bbox":[1,2,10,12]}]},headers=self.headers).status_code,400)
    def test_multiple_faces_can_have_individual_verified_classes(self):
        boxes=[{"class_id":0,"bbox":[1,2,10,12]},{"class_id":1,"bbox":[12,2,23,15]}]
        self.assertEqual(self.request(boxes=boxes).status_code,200)
        row=json.loads((self.folder/"manifest.json").read_text())["images"][0]
        self.assertEqual([b["class_id"] for b in row["boxes"]],[0,1])
    def test_multiple_faces_can_share_the_same_class(self):
        boxes=[{"class_id":1,"bbox":[1,2,10,12]},{"class_id":1,"bbox":[12,2,23,15]}]
        self.assertEqual(self.request(boxes=boxes).status_code,200)
        row=json.loads((self.folder/"manifest.json").read_text())["images"][0]
        self.assertEqual(row["boxes"],boxes)

    def test_valid_image_route_and_missing_id(self):
        self.assertEqual(self.client.get("/image/a").status_code,200)
        self.assertEqual(self.client.get("/image/missing").status_code,404)

class CameraBoundaryTests(unittest.TestCase):
    def test_camera_review_cannot_accidentally_become_training(self):
        from scripts.build_baseline_dataset import build
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public=root/"public";review=root/"review"
            public.mkdir();review.mkdir()
            (public/"split_manifest.json").write_text('{"splits":{}}')
            (review/"manifest.json").write_text('{"source_kind":"esp32","images":[]}')
            with patch("scripts.build_baseline_dataset.audit_prepared",return_value={"error_count":0}):
                with self.assertRaisesRegex(ValueError,"Camera evaluation"):
                    build(public,review,root/"output")
            self.assertFalse((root/"output").exists())

class EvaluationMatchingTests(unittest.TestCase):
    def test_same_box_wrong_class_is_fp_and_fn(self):
        errors=match_errors([{"class_id":0,"bbox":[0,0,10,10]}],[{"class_id":1,"bbox":[0,0,10,10]}])
        self.assertEqual(errors["true_positives"],0)
        self.assertEqual(len(errors["false_positives"]),1);self.assertEqual(len(errors["false_negatives"]),1)
    def test_one_ground_truth_matches_only_once(self):
        g={"class_id":0,"bbox":[0,0,10,10]}
        errors=match_errors([g],[g,g])
        self.assertEqual(errors["true_positives"],1);self.assertEqual(len(errors["false_positives"]),1)
    def test_disjoint_zero_overlap(self):
        self.assertEqual(overlap([0,0,1,1],[2,2,3,3]),0)

if __name__=="__main__":unittest.main()
