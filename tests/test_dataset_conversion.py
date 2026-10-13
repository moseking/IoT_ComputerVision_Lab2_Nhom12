"""VOC conversion tests use temporary generated fixtures, never the user's dataset."""
import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image
from scripts.common import CLASSES
from scripts.dataset_tools import audit, yolo_lines
from scripts.prepare_dataset import prepare


def snapshot(folder):
    return {p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


class VocConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="voc-conversion-tests-")
        self.root=Path(self.temp.name)
        self.raw=self.root/"raw"
        (self.raw/"images").mkdir(parents=True)
        (self.raw/"annotations").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def image(self,name,width=100,height=80,copy=None):
        path=self.raw/"images"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if copy:
            path.write_bytes(copy.read_bytes())
        else:
            rng=random.Random(name)
            image=Image.new("RGB",(width,height))
            image.putdata([(rng.randrange(256),rng.randrange(256),rng.randrange(256)) for _ in range(width*height)])
            image.save(path)
        return path

    def xml(self,image,objects=None,xml_name=None,width=100,height=80,filename=None):
        root=ET.Element("annotation")
        ET.SubElement(root,"filename").text=filename or image
        size=ET.SubElement(root,"size")
        ET.SubElement(size,"width").text=str(width)
        ET.SubElement(size,"height").text=str(height)
        for cls,coords in (objects if objects is not None else [("with_mask",(20,10,60,50))]):
            obj=ET.SubElement(root,"object")
            ET.SubElement(obj,"name").text=cls
            box=ET.SubElement(obj,"bndbox")
            for key,value in zip(("xmin","ymin","xmax","ymax"),coords):
                ET.SubElement(box,key).text=str(value)
        name=xml_name or Path(image).stem+".xml"
        path=self.raw/"annotations"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        ET.ElementTree(root).write(path,encoding="utf-8")
        return path

    def three(self):
        for index,cls in CLASSES.items():
            name=f"image{index}.png"
            self.image(name)
            self.xml(name,[(cls,(20,10,60,50))])

    def run_prepare(self,**options):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.raw,self.root/"output",near_duplicate_distance=-1,**options)

    def test_valid_xml_exact_conversion(self):
        self.image("face.png")
        self.xml("face.png")
        records,report=audit(self.raw)
        self.assertEqual(report["error_count"],0)
        values=yolo_lines(records[0]).split()
        self.assertEqual(int(values[0]),0)
        self.assertEqual(list(map(float,values[1:])),[.4,.375,.4,.5])
        self.assertEqual(report["raw_objects"],1)

    def test_multiple_objects_fixed_class_mapping(self):
        self.image("faces.png")
        self.xml("faces.png",[(CLASSES[2],(1,2,10,20)),(CLASSES[0],(20,10,60,50)),(CLASSES[1],(60,10,90,70))])
        records,report=audit(self.raw)
        lines=yolo_lines(records[0]).splitlines()
        self.assertEqual([int(line.split()[0]) for line in lines],[2,0,1])
        self.assertEqual(report["raw_class_distribution"],{name:1 for name in CLASSES.values()})

    def test_unknown_class_excludes_whole_image(self):
        self.image("face.png")
        self.xml("face.png",[("with_mask",(1,2,10,20)),("other",(20,10,60,50))])
        records,report=audit(self.raw)
        self.assertEqual(records,[])
        self.assertEqual(report["raw_objects"],2)
        self.assertEqual(report["unknown_class_distribution"],{"other":1})
        self.assertEqual(report["issue_counts"]["unknown_class"],1)

    def test_malformed_xml(self):
        self.image("face.png")
        (self.raw/"annotations/face.xml").write_text("<annotation>",encoding="utf-8")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["invalid_xml"],1)

    def test_missing_image_counts_xml_objects_without_inference(self):
        self.xml("missing.png")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["raw_objects"],1)
        self.assertEqual(report["issue_counts"]["missing_image"],1)

    def test_image_missing_xml(self):
        self.image("orphan.png")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["missing_annotation"],1)

    def test_wrong_coordinates_and_nonfinite(self):
        self.image("face.png")
        for coords in ((20,10,10,50),(20,10,20,50),(float("nan"),10,60,50),(20,10,float("inf"),50)):
            with self.subTest(coords=coords):
                self.xml("face.png",[("with_mask",coords)])
                records,report=audit(self.raw)
                self.assertFalse(records)
                self.assertEqual(report["issue_counts"]["invalid_bbox"],1)

    def test_outside_bbox_not_silently_clipped(self):
        self.image("face.png")
        self.xml("face.png",[("with_mask",(20,10,101,50))])
        before=snapshot(self.raw)
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["invalid_bbox"],1)
        self.assertNotIn("clipped_bbox",report["issue_counts"])
        self.assertEqual(snapshot(self.raw),before)

    def test_explicit_legacy_repair_reports_every_clip(self):
        self.image("face.png")
        self.xml("face.png",[("with_mask",(20,10,101,50))])
        before=snapshot(self.raw)
        records,report=audit(self.raw,repair_boxes=True)
        self.assertEqual(records[0]["boxes"][0]["bbox"][2],100)
        self.assertEqual(report["issue_counts"]["clipped_bbox"],1)
        self.assertEqual(snapshot(self.raw),before)

    def test_xml_dimension_mismatch_rejected(self):
        self.image("face.png")
        self.xml("face.png",width=999)
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["dimension_mismatch"],1)

    def test_xml_basename_can_differ_pair_by_filename(self):
        self.image("actual.png")
        self.xml("actual.png",xml_name="different.xml")
        records,report=audit(self.raw)
        self.assertEqual(Path(records[0]["image"]).name,"actual.png")
        self.assertEqual(report["issue_counts"]["xml_image_name_mismatch"],1)
        self.assertEqual(report["error_count"],0)

    def test_folder_name_does_not_infer_class(self):
        self.image("with_mask/face.png")
        self.xml("with_mask/face.png",[("without_mask",(20,10,60,50))])
        records,_=audit(self.raw)
        self.assertEqual(records[0]["boxes"][0]["class_id"],1)

    def test_unsafe_filename_is_not_guessed_by_basename(self):
        self.image("face.png")
        self.xml("face.png",filename="../../face.png")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["invalid_filename"],1)

    def test_dtd_entity_rejected(self):
        self.image("face.png")
        (self.raw/"annotations/face.xml").write_text(
            '<!DOCTYPE annotation [<!ENTITY attack SYSTEM "file:///private">]><annotation><filename>face.png</filename></annotation>',
            encoding="utf-8")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["invalid_xml"],1)

    def test_duplicate_basenames_ambiguous(self):
        self.image("a/face.png")
        self.image("b/face.png")
        self.xml("a/face.png",xml_name="a.xml")
        self.xml("b/face.png",xml_name="b.xml")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["duplicate_basename"],2)

    def test_multiple_xml_same_image_excludes_all_annotations(self):
        self.image("face.png")
        self.xml("face.png",xml_name="a.xml")
        self.xml("face.png",xml_name="b.xml")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["duplicate_annotation"],2)
        self.assertEqual(report["invalid_xml_files"],2)

    def test_corrupt_image_rejected(self):
        (self.raw/"images/face.png").write_bytes(b"not an image")
        self.xml("face.png")
        records,report=audit(self.raw)
        self.assertFalse(records)
        self.assertEqual(report["issue_counts"]["invalid_image"],1)

    def test_dry_run_writes_no_files(self):
        self.three()
        before=snapshot(self.root)
        report,summary=self.run_prepare(dry_run=True,
            audit_output=self.root/"reports/raw.json",conversion_output=self.root/"reports/conversion.json")
        self.assertFalse(summary["blocking"])
        self.assertEqual(snapshot(self.root),before)
        self.assertFalse((self.root/"output").exists())
        self.assertFalse((self.root/"reports").exists())

    def test_rerun_is_deterministic_no_append_or_raw_overwrite(self):
        self.three()
        before=snapshot(self.raw)
        self.run_prepare(seed=42)
        output=self.root/"output"
        files=snapshot(output)
        manifest=json.loads((output/"split_manifest.json").read_text())
        with self.assertRaises(FileExistsError):
            self.run_prepare(seed=42)
        self.assertEqual(snapshot(output),files)
        self.run_prepare(seed=42,overwrite=True)
        after=snapshot(output)
        for name,digest in files.items():
            if name not in {"split_manifest.json","README.md"}:
                self.assertEqual(after[name],digest,name)
        self.assertEqual(json.loads((output/"split_manifest.json").read_text())["splits"],manifest["splits"])
        self.assertEqual(snapshot(self.raw),before)
        config=yaml.safe_load((output/"data.yaml").read_text())
        self.assertEqual(config["nc"],3)
        self.assertEqual(config["names"],CLASSES)
        self.assertEqual(len(list((output/"labels").rglob("*.txt"))),3)
        self.assertTrue(all(len(p.read_text().splitlines())==1 for p in (output/"labels").rglob("*.txt")))

    def test_raw_output_and_report_paths_protected(self):
        self.three()
        before=snapshot(self.raw)
        for output in (self.raw,self.raw/"generated",self.root):
            with self.subTest(output=output),self.assertRaises(ValueError):
                prepare(self.raw,output,dry_run=True)
        with self.assertRaises(ValueError):
            self.run_prepare(audit_output=self.raw/"audit.json")
        self.assertEqual(snapshot(self.raw),before)

    def test_skip_whole_invalid_image_not_fake_background(self):
        self.three()
        self.image("bad.png")
        self.xml("bad.png",[("with_mask",(1,2,10,20)),("without_mask",(20,10,101,50))])
        _,summary=self.run_prepare(skip_invalid=True)
        self.assertEqual(summary["accepted_images"],3)
        self.assertEqual(summary["skipped_images"],1)
        self.assertEqual(summary["valid_background_images"],0)
        self.assertFalse(list((self.root/"output/labels").rglob("bad.txt")))

    def test_valid_zero_object_xml_is_explicit_background(self):
        self.three()
        self.image("background.png")
        self.xml("background.png",[])
        _,summary=self.run_prepare()
        self.assertEqual(summary["valid_background_images"],1)
        label=next((self.root/"output/labels").rglob("background.txt"))
        self.assertEqual(label.read_text(),"")

    def test_exact_duplicates_stay_in_one_split(self):
        self.three()
        first=self.raw/"images/image0.png"
        self.image("duplicate.png",copy=first)
        self.xml("duplicate.png")
        self.run_prepare()
        manifest=json.loads((self.root/"output/split_manifest.json").read_text())
        assigned={item["image"]:split for split,items in manifest["splits"].items() for item in items}
        self.assertEqual(assigned["image0.png"],assigned["duplicate.png"])

    def test_conflicting_duplicate_policy_explicit_and_logged(self):
        self.three()
        first=self.image("original.png")
        self.xml("original.png",[("with_mask",(20,10,60,50))])
        self.image("conflict.png",copy=first)
        self.xml("conflict.png",[("without_mask",(20,10,60,50))])
        before=snapshot(self.raw)
        with self.assertRaises(ValueError):
            self.run_prepare()
        _,summary=self.run_prepare(duplicate_policy="exclude")
        self.assertEqual(summary["accepted_images"],3)
        self.assertEqual(summary["excluded_conflicting_images"],2)
        self.assertTrue(all("conflicting_duplicate_labels" in item["reasons"] for item in summary["discarded_images"]))
        self.assertEqual(snapshot(self.raw),before)

    def test_external_person_session_group_stays_together(self):
        self.three()
        self.image("extra.png")
        self.xml("extra.png")
        groups=self.root/"groups.json"
        groups.write_text(json.dumps({"image0.png":"same-person","extra.png":"same-person"}),encoding="utf-8")
        self.run_prepare(groups_path=groups)
        manifest=json.loads((self.root/"output/split_manifest.json").read_text())
        assigned={item["image"]:split for split,items in manifest["splits"].items() for item in items}
        self.assertEqual(assigned["image0.png"],assigned["extra.png"])
