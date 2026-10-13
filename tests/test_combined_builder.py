"""Combined custom source exports keep holdouts frozen and duplicate labels safe."""
import hashlib,json,random,tempfile,unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from PIL import Image
import yaml
from scripts.build_baseline_dataset import build
from scripts.dataset_tools import dhash

CLASSES={0:"with_mask",1:"without_mask",2:"mask_weared_incorrect"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def image_at(path,seed):
    path.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed);image=Image.frombytes("RGB",(40,40),rng.randbytes(40*40*3));image.save(path)
    pixel=hashlib.sha256(b"40x40:"+image.tobytes()).hexdigest()
    return pixel,dhash(image)

class CombinedBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.public=self.root/"public"
        self.public.mkdir();splits={}
        for seed,split in enumerate(["train","val","test"],1):
            name=split+".png";p=self.public/"images"/split/name;pixel,hash=image_at(p,seed)
            labels=self.public/"labels"/split;labels.mkdir(parents=True)
            (labels/(split+".txt")).write_text("0 .2 .2 .1 .1\n1 .5 .5 .1 .1\n2 .8 .8 .1 .1\n")
            splits[split]=[{"image":name,"source_sha256":sha(p),"pixel_hash":pixel,"dhash":hash,"objects":3,"class_counts":{0:1,1:1,2:1}}]
        (self.public/"split_manifest.json").write_text(json.dumps({"splits":splits}))
        (self.public/"data.yaml").write_text(yaml.safe_dump({"path":str(self.public),"train":"images/train","val":"images/val","test":"images/test","names":CLASSES,"nc":3}))
        self.phone=self.review("phone",10,[{"class_id":0,"bbox":[1,1,12,12]}])
        self.multi=self.review("multi",11,[{"class_id":1,"bbox":[1,1,12,12]},{"class_id":1,"bbox":[15,1,30,12]}])
    def tearDown(self):self.tmp.cleanup()
    def review(self,kind,seed,boxes):
        folder=self.root/kind;folder.mkdir();p=folder/"images"/(kind+".png");pixel,hash=image_at(p,seed)
        row={"id":kind,"index":0,"source_relative":p.name,"processed_file":"images/"+p.name,"processed_sha256":sha(p),
             "source_file":str(p),"source_sha256":sha(p),"pixel_hash":pixel,"dhash":hash,"width":40,"height":40,
             "label_status":"verified","conversion_status":"success","human_reviewed":True,"reviewer":"unit fixture","reviewed_at":"2026-10-11T00:00:00Z","boxes":boxes,"face_count":len(boxes)}
        (folder/"manifest.json").write_text(json.dumps({"source_kind":kind,"images":[row]}));return folder
    def export(self):
        with redirect_stdout(StringIO()):return build(self.public,[self.phone,self.multi],self.root/"output")
    def duplicate_multi_from_phone(self,boxes=None):
        phone=json.loads((self.phone/"manifest.json").read_text())["images"][0]
        multi=json.loads((self.multi/"manifest.json").read_text());r=multi["images"][0]
        (self.multi/r["processed_file"]).write_bytes((self.phone/phone["processed_file"]).read_bytes())
        for k in ["processed_sha256","source_sha256","pixel_hash","dhash"]:r[k]=phone[k]
        r["boxes"]=boxes or phone["boxes"];r["face_count"]=len(r["boxes"])
        (self.multi/"manifest.json").write_text(json.dumps(multi))
    def test_both_sources_and_repeated_classes_preserve_holdouts(self):
        manifest=self.export()
        self.assertEqual(manifest["custom_accepted_by_source"],{"phone":1,"multi":1})
        self.assertEqual(len(manifest["splits"]["train"]),3)
        for split in ["val","test"]:
            for category,extension in [("images","png"),("labels","txt")]:
                self.assertEqual((self.public/category/split/(split+"."+extension)).read_bytes(),
                    (self.root/"output"/category/split/("public_"+split+"."+extension)).read_bytes())
        self.assertEqual([line.split()[0] for line in (self.root/"output/labels/train/multi.txt").read_text().splitlines()],["1","1"])
        self.assertEqual(len(manifest["custom_review_sources"]),2)
    def test_identical_cross_source_image_keeps_one(self):
        self.duplicate_multi_from_phone();manifest=self.export()
        self.assertEqual(manifest["custom_accepted"],1);self.assertEqual(manifest["custom_excluded_by_source"],{"multi":1})
    def test_conflicting_same_image_classes_exclude_all_copies(self):
        self.duplicate_multi_from_phone([{"class_id":1,"bbox":[1,1,12,12]}]);manifest=self.export()
        self.assertEqual(manifest["custom_accepted"],0)
        self.assertEqual(manifest["custom_excluded_by_source"],{"phone":1,"multi":1})
        self.assertTrue(manifest["duplicate_annotation_conflicts"])
    def cleanup(self):
        path=self.multi/"manifest.json";m=json.loads(path.read_text());r=m["images"][0]
        r["boxes"].append({"class_id":1,"bbox":[32,32,38,38]});r["face_count"]=3
        path.write_text(json.dumps(m));snapshot=path.read_bytes()
        cleanup={"target":"real_people_only","review_sources":{str(self.multi.resolve()):{
            "manifest_sha256":sha(path),"rows":[{"id":r["id"],"processed_sha256":r["processed_sha256"],
            "original_boxes":r["boxes"],"removed_box_indices":[2],"reason":"printed_face_not_real_person",
            "visual_evidence":"fixture intentionally supplies a third printed-face annotation"}]}}}
        ledger=self.root/"cleanup.json";ledger.write_text(json.dumps(cleanup));return ledger,snapshot
    def test_non_person_cleanup_keeps_original_review_and_real_boxes(self):
        ledger,snapshot=self.cleanup()
        with redirect_stdout(StringIO()):manifest=build(self.public,[self.phone,self.multi],self.root/"output",box_exclusions=ledger)
        self.assertEqual((self.multi/"manifest.json").read_bytes(),snapshot)
        self.assertEqual(manifest["non_person_boxes_removed_from_export"],1)
        row=next(r for r in manifest["splits"]["train"] if r["source_kind"]=="multi")
        self.assertEqual(row["objects"],2);self.assertEqual(row["class_counts"],{1:2})
        self.assertEqual(row["non_person_cleanup"]["original_proposal_count"],3)
    def test_stale_cleanup_refuses_changed_review(self):
        ledger,_=self.cleanup();path=self.multi/"manifest.json";m=json.loads(path.read_text())
        m["images"][0]["boxes"][0]["class_id"]=0;path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"snapshot is stale"):
            build(self.public,[self.multi],self.root/"output",box_exclusions=ledger)
    def test_cleanup_cannot_hide_missing_real_face(self):
        ledger,_=self.cleanup();path=self.multi/"manifest.json";m=json.loads(path.read_text())
        m["images"][0]["face_count"]=4;path.write_text(json.dumps(m))
        cleanup=json.loads(ledger.read_text());cleanup["review_sources"][str(self.multi.resolve())]["manifest_sha256"]=sha(path)
        ledger.write_text(json.dumps(cleanup))
        with self.assertRaisesRegex(ValueError,"missing real-person"):
            build(self.public,[self.multi],self.root/"output",box_exclusions=ledger)
    def image_qc(self):
        path=self.multi/"manifest.json";m=json.loads(path.read_text());r=m["images"][0]
        ledger=self.root/"image_qc.json"
        ledger.write_text(json.dumps({"review_sources":{str(self.multi.resolve()):{
            "manifest_sha256":sha(path),"rows":[{"id":r["id"],"processed_sha256":r["processed_sha256"],
            "original_boxes":r["boxes"],"reason":"ambiguous_mask_transition","visual_evidence":"test ambiguous transition"}]}}}))
        return ledger,path.read_bytes()
    def test_image_qc_excludes_whole_frame_without_changing_review(self):
        ledger,snapshot=self.image_qc()
        with redirect_stdout(StringIO()):manifest=build(self.public,[self.phone,self.multi],self.root/"output",image_exclusions=ledger)
        self.assertEqual(manifest["custom_accepted_by_source"],{"phone":1})
        self.assertEqual((self.multi/"manifest.json").read_bytes(),snapshot)
        self.assertTrue(manifest["excluded_custom"][0]["reason"].startswith("ambiguous_mask_transition"))
        self.assertFalse((self.root/"output/images/train/multi.png").exists())
        for split in ["val","test"]:
            self.assertEqual((self.public/"labels"/split/(split+".txt")).read_bytes(),
                (self.root/"output/labels"/split/("public_"+split+".txt")).read_bytes())
    def test_stale_image_qc_refuses_changed_review(self):
        ledger,_=self.image_qc();path=self.multi/"manifest.json";m=json.loads(path.read_text())
        m["images"][0]["boxes"][0]["class_id"]=0;path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"Image QC snapshot is stale"):
            build(self.public,[self.multi],self.root/"output",image_exclusions=ledger)
    def test_image_qc_refuses_mismatched_audited_box(self):
        ledger,_=self.image_qc();data=json.loads(ledger.read_text())
        data["review_sources"][str(self.multi.resolve())]["rows"][0]["original_boxes"][0]["class_id"]=0
        ledger.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"image/boxes do not match"):
            build(self.public,[self.multi],self.root/"output",image_exclusions=ledger)
    def test_test_fixture_source_cannot_be_exported(self):
        path=self.multi/"manifest.json";m=json.loads(path.read_text());m["test_fixture"]=True;path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"Test fixtures"):self.export()

if __name__=="__main__":unittest.main()
