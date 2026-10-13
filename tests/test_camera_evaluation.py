import copy
import hashlib
import json
import unittest
import tempfile
from pathlib import Path
from PIL import Image
from scripts.common import CLASSES
from scripts.evaluate_camera import context_rows, reviewed_rows


class CameraEvaluationTests(unittest.TestCase):
 def test_exclusive_face_nms_keeps_separate_people_and_one_class_per_face(self):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    # The first two candidates refer to the same face with opposing labels;
    # the third is another person with the same label as the second candidate.
    predictions=torch.tensor([[[30.,31.,80.],[30.,31.,30.],[20.,20.,20.],[20.,20.,20.],
                                [.9,0.,0.],[0.,0.,0.],[0.,.8,.7]]])
    result=non_max_suppression(predictions,conf_thres=.35,iou_thres=.45,agnostic=True,nc=3)[0]
    self.assertEqual(len(result),2)
    self.assertEqual(result[:,5].tolist(),[0.,2.])
    self.assertTrue(torch.allclose(result[:,0],torch.tensor([20.,70.])))

 def test_context_expansion_preserves_user_annotation_and_class(self):
    rows=[{'index':4,'width':100,'height':100,'boxes':[{'class_id':2,'bbox':[20,20,50,60]}]}]
    original=copy.deepcopy(rows)
    ledger={'manifest_sha256':'abc','actor':'assistant','changes':[{'index':4,'box_index':0,'bbox':[10,20,70,90],'reason':'hanging mask'}]}
    result=context_rows(rows,ledger,'abc')
    assert rows==original
    assert result[0]['boxes'][0]['class_id']==2
    assert result[0]['boxes'][0]['bbox_original_human']==[20,20,50,60]
    assert result[0]['boxes'][0]['geometry_adjustment']['human_verified'] is False


 def test_context_rejects_wrong_class_shrink_outside_or_stale(self):
    for label,new,hash_value in [(0,[10,10,80,90],'abc'),(2,[25,20,70,90],'abc'),(2,[10,10,110,90],'abc'),(2,[10,10,80,90],'stale')]:
        with self.subTest(label=label,new=new,hash_value=hash_value):
            rows=[{'index':4,'width':100,'height':100,'boxes':[{'class_id':label,'bbox':[20,20,50,60]}]}]
            ledger={'manifest_sha256':'abc','actor':'assistant','changes':[{'index':4,'box_index':0,'bbox':new,'reason':'mask'}]}
            with self.assertRaises(ValueError):context_rows(rows,ledger,hash_value)


 def test_camera_gate_rejects_unverified_and_changed_pixels(self):
    tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);tmp_path=Path(tmp.name)
    image=tmp_path/'image.jpg'; Image.new('RGB',(100,100),'white').save(image)
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    row={'index':0,'processed_file':'image.jpg','source_file':str(image),'processed_sha256':digest,'source_sha256':digest,
         'width':100,'height':100,'boxes':[{'class_id':2,'bbox':[20,20,50,60]}],'label_status':'draft','human_reviewed':False,
         'reviewer':'Mose','reviewed_at':'2026-10-11','included_in_training':False}
    manifest={'source_kind':'esp32','mapping':CLASSES,'images':[row]}
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(manifest),encoding='utf-8')
    with self.assertRaisesRegex(ValueError,'verification'):reviewed_rows(tmp_path)
    row.update(label_status='verified',human_reviewed=True)
    path.write_text(json.dumps(manifest),encoding='utf-8'); reviewed_rows(tmp_path)
    Image.new('RGB',(100,100),'black').save(image)
    with self.assertRaisesRegex(ValueError,'changed'):reviewed_rows(tmp_path)
