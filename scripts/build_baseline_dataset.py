"""Versioned baseline: preserve public splits, add ONLY explicitly reviewed custom faces to train."""
import argparse, csv, hashlib, json, shutil, tempfile
from collections import Counter
from pathlib import Path
import yaml
try:
    from .common import ROOT, CLASSES, timestamp, write_json
    from .dataset_tools import yolo_lines
    from .custom_dataset import validate_boxes
    from .audit_prepared_dataset import audit_prepared
except ImportError:
    from common import ROOT, CLASSES, timestamp, write_json
    from dataset_tools import yolo_lines
    from custom_dataset import validate_boxes
    from audit_prepared_dataset import audit_prepared

def apply_box_exclusions(review, manifest_bytes, custom, cleanup):
    """Remove only exact visually audited non-person annotations; retain human provenance."""
    if not cleanup: return
    source=cleanup.get("review_sources",{}).get(str(review))
    if not source: return
    if hashlib.sha256(manifest_bytes).hexdigest()!=source["manifest_sha256"]:
        raise ValueError("Non-person cleanup snapshot is stale; re-audit changed review manifest")
    by_id={r["id"]:r for r in custom["images"]}
    for entry in source["rows"]:
        row=by_id[entry["id"]]
        if row.get("label_status")!="verified" or row.get("human_reviewed") is not True:
            raise ValueError("Non-person cleanup cannot authorize an unreviewed image")
        if row["processed_sha256"]!=entry["processed_sha256"] or row["boxes"]!=entry["original_boxes"]:
            raise ValueError("Non-person cleanup image/boxes do not match audited snapshot")
        removed=entry["removed_box_indices"]
        if not removed or len(set(removed))!=len(removed) or any(type(i) is not int or not 0<=i<len(row["boxes"]) for i in removed):
            raise ValueError("Invalid non-person box exclusions")
        if entry.get("reason")!="printed_face_not_real_person" or not entry.get("visual_evidence"):
            raise ValueError("Non-person exclusions require reason and visual evidence")
        # An exclusion cannot hide missing real faces already detected by the review gate.
        if row.get("face_count",0)>len(row["boxes"]):
            raise ValueError("Cleanup cannot resolve missing real-person labels")
        original=row["boxes"]
        row["boxes"]=[b for i,b in enumerate(original) if i not in removed]
        if not row["boxes"]: raise ValueError("Cleanup must retain reviewed real-person boxes")
        row["non_person_cleanup"]={"original_boxes":original,"removed_box_indices":removed,
            "removed_boxes":[original[i] for i in removed],"reason":entry["reason"],
            "visual_evidence":entry["visual_evidence"],"original_proposal_count":row.get("face_count",0),
            "audit_actor":"assistant visual inspection; remaining boxes were human-reviewed",
            "user_scope":"only real people count; printed poster faces excluded"}

def apply_image_exclusions(review, manifest_bytes, custom, ledger):
    """Hold entire ambiguous images out without changing human annotations."""
    source=ledger.get("review_sources",{}).get(str(review)) if ledger else None
    if not source: return
    if hashlib.sha256(manifest_bytes).hexdigest()!=source["manifest_sha256"]:
        raise ValueError("Image QC snapshot is stale; re-audit changed review manifest")
    by_id={r["id"]:r for r in custom["images"]}
    for entry in source["rows"]:
        row=by_id[entry["id"]]
        if row["processed_sha256"]!=entry["processed_sha256"] or row["boxes"]!=entry["original_boxes"]:
            raise ValueError("Image QC image/boxes do not match audited snapshot")
        if entry.get("reason") not in {"ambiguous_mask_transition","prediction_overlay_baked_into_image"} or not entry.get("visual_evidence"):
            raise ValueError("Image QC requires reason and visual evidence")
        row["image_qc_exclusion"]=entry["reason"]+": "+entry["visual_evidence"]


def apply_context_expansions(review, manifest_bytes, custom, ledger):
    """User-authorized context export from verified base labels; original reviews stay intact."""
    source=ledger.get("review_sources",{}).get(str(review)) if ledger else None
    if not source:return
    if ledger.get("authorization")!="user_requested_mask_context":raise ValueError("Context expansion requires explicit user scope")
    if hashlib.sha256(manifest_bytes).hexdigest()!=source["manifest_sha256"]:raise ValueError("Context snapshot is stale")
    by_id={r["id"]:r for r in custom["images"]}
    for entry in source["rows"]:
        row=by_id[entry["id"]]
        if row.get("label_status")!="verified" or row.get("human_reviewed") is not True:raise ValueError("Context cannot authorize unreviewed base labels")
        if row["processed_sha256"]!=entry["processed_sha256"] or row["boxes"]!=entry["original_boxes"]:raise ValueError("Context base image/boxes mismatch")
        original=validate_boxes(row["boxes"],row["width"],row["height"],False)
        expanded=validate_boxes(entry["expanded_boxes"],row["width"],row["height"],False)
        if len(original)!=len(expanded):raise ValueError("Context cannot add/remove faces")
        for before,after in zip(original,expanded):
            if before["class_id"]!=after["class_id"]:raise ValueError("Context cannot change class")
            if before["class_id"]!=2 and before!=after:raise ValueError("Only incorrect-mask context can expand")
            a,b=before["bbox"],after["bbox"]
            if not(b[0]<=a[0] and b[1]<=a[1] and b[2]>=a[2] and b[3]>=a[3]):raise ValueError("Context must contain original face box")
        row["context_expansion"]={"original_boxes":original,"expanded_boxes":expanded,
            "geometry_actor":"assistant visual inspection","geometry_human_reverified":False,
            "human_review_scope":"original class and geometry before expansion",
            "authorization":"user requested wider incorrect-mask boxes for training"}
        row["boxes"]=expanded

def reviewed_count(row):
    return len(row["boxes"])+len(row.get("non_person_cleanup",{}).get("removed_boxes",[]))

def build(public, review, output, reports=None, box_exclusions=None, image_exclusions=None, allow_camera_train=False, bbox_context=None):
    public,output=Path(public).resolve(),Path(output).resolve()
    reviews=[Path(p).resolve() for p in (review if isinstance(review,(list,tuple)) else [review])]
    if not reviews or len(set(reviews))!=len(reviews): raise ValueError("Review sources must be unique and non-empty")
    reports=Path(reports).resolve() if reports else output
    if output.exists(): raise FileExistsError("Baseline version already exists; use a new version")
    for source in [public,*reviews,ROOT/"data/raw"]:
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Baseline output must be disjoint from input sources")
    check=audit_prepared(public/"data.yaml")
    if check["error_count"]: raise ValueError("Public prepared audit failed")
    public_manifest=json.loads((public/"split_manifest.json").read_text(encoding="utf-8"))
    staged_rows=[]; exclusions=[]; review_sources=[]
    cleanup_bytes=Path(box_exclusions).read_bytes() if box_exclusions else None
    cleanup=json.loads(cleanup_bytes.decode("utf-8")) if cleanup_bytes else None
    if cleanup and cleanup.get("target")!="real_people_only": raise ValueError("Invalid cleanup target")
    image_qc_bytes=Path(image_exclusions).read_bytes() if image_exclusions else None
    image_qc=json.loads(image_qc_bytes.decode("utf-8")) if image_qc_bytes else None
    context_bytes=Path(bbox_context).read_bytes() if bbox_context else None
    context=json.loads(context_bytes.decode("utf-8")) if context_bytes else None
    public_holdout=[r for split,rows in public_manifest["splits"].items() if split!="train" for r in rows]
    duplicate_seen={r["pixel_hash"]:"public:"+r["image"] for items in public_manifest["splits"].values() for r in items}
    source_hash_seen={}
    output_names={"public_"+r["image"] for items in public_manifest["splits"].values() for r in items}
    snapshots=[]; annotation_groups={}; conflicting_rows=set(); annotation_conflicts=[]
    for review in reviews:
        manifest_path=review/"manifest.json"
        if not manifest_path.is_file(): raise FileNotFoundError("Review manifest missing: "+str(review))
        manifest_bytes=manifest_path.read_bytes()
        custom=json.loads(manifest_bytes.decode("utf-8"))
        source_kind=custom.get("source_kind", "phone" if custom.get("single_person",True) else "capture")
        if source_kind=="esp32" and not allow_camera_train: raise ValueError("Camera evaluation must stay separate from baseline training; explicit camera training opt-in required")
        if custom.get("test_fixture") or source_kind=="test_fixture": raise ValueError("Test fixtures cannot become training data")
        apply_image_exclusions(review,manifest_bytes,custom,image_qc)
        apply_box_exclusions(review,manifest_bytes,custom,cleanup)
        apply_context_expansions(review,manifest_bytes,custom,context)
        snapshots.append((review,manifest_bytes,custom,source_kind))
        for row in custom["images"]:
            if row.get("label_status")!="verified" or not row.get("human_reviewed") or row.get("conversion_status")!="success": continue
            if row.get("image_qc_exclusion"): continue
            try: clean=validate_boxes(row["boxes"],row["width"],row["height"],False)
            except (ValueError,TypeError,KeyError): continue
            if row.get("face_count",0)>reviewed_count(row): continue
            ordered=sorted(clean,key=lambda b:((b["bbox"][0]+b["bbox"][2])/2,(b["bbox"][1]+b["bbox"][3])/2))
            signature=tuple(b["class_id"] for b in ordered)
            entry={"review_source":str(review),"source_kind":source_kind,"id":row["id"],"index":row.get("index"),"signature":signature}
            for key in ("pixel_hash","source_sha256"):
                if row.get(key): annotation_groups.setdefault(key+":"+row[key],[]).append(entry)
    for key,items in annotation_groups.items():
        if len({item["signature"] for item in items})>1:
            conflicting_rows.update((item["review_source"],item["id"]) for item in items)
            annotation_conflicts.append({"hash_kind":key.split(":",1)[0],"hash":key.split(":",1)[1],"rows":items,
                "reason":"Same exact image/source has conflicting classes by left-to-right face order; exclude all until reviewed"})
    for review,manifest_bytes,custom,source_kind in snapshots:
        custom_group=f"{source_kind}_capture_group_01"
        review_sources.append({"path":str(review),"source_kind":source_kind,
            "manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
            "status_counts":dict(Counter(r.get("label_status","unknown") for r in custom["images"]))})
        for row in custom["images"]:
            reason=None
            if row.get("label_status")!="verified": reason="label_not_verified:"+str(row.get("label_status"))
            elif row.get("conversion_status")!="success": reason="conversion_failed"
            elif row.get("image_qc_exclusion"): reason=row["image_qc_exclusion"]
            elif (str(review),row["id"]) in conflicting_rows: reason="Exact duplicate image has conflicting reviewed face classes; exclude all copies"
            else:
                try:
                    if row.get("human_reviewed") is not True or not row.get("reviewer") or not row.get("reviewed_at"): raise ValueError("Review provenance missing")
                    source=(review/row["processed_file"]).resolve()
                    if not source.is_relative_to(review): raise ValueError("Processed image escapes review folder")
                    if hashlib.sha256(source.read_bytes()).hexdigest()!=row["processed_sha256"]: raise ValueError("Processed image changed since review")
                    if hashlib.sha256(Path(row["source_file"]).read_bytes()).hexdigest()!=row["source_sha256"]: raise ValueError("Original source changed since audit")
                    row["boxes"]=validate_boxes(row["boxes"],row["width"],row["height"],False)
                    if any(row["pixel_hash"]==other["pixel_hash"] or (row["dhash"]^other["dhash"]).bit_count()<=4 for other in public_holdout):
                        raise ValueError("Custom image matches public holdout")
                    if row.get("face_count",0)>reviewed_count(row):
                        raise ValueError("More face proposals than reviewed boxes; recheck all real faces before train")
                    if row["pixel_hash"] in duplicate_seen or row["source_sha256"] in source_hash_seen:
                        raise ValueError("Duplicate custom/public pixels or original source; keep first reviewed sample")
                    if source.name in output_names: raise ValueError("Output filename collision across review sources")
                    duplicate_seen[row["pixel_hash"]]=str(review)+":"+row["id"]
                    source_hash_seen[row["source_sha256"]]=str(review)+":"+row["id"]
                    output_names.add(source.name)
                except (ValueError,OSError,KeyError) as exc: reason=str(exc)
            if reason: exclusions.append({"id":row["id"],"source_kind":source_kind,"review_source":str(review),"source_relative":row["source_relative"],"reason":reason})
            else: staged_rows.append({"row":row,"review":review,"source_kind":source_kind,"group_id":custom_group})
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]; manifest={"created":timestamp(),"seed":42,"classes":CLASSES,"public_source":str(public),
        "custom_review_source":str(reviews[0]),"custom_review_sources":review_sources,
        "camera_training_opt_in":bool(allow_camera_train),
        "camera_evaluation_policy":"Camera sources included with opt-in are training/development data, never independent evaluation",
        "custom_policy":"all reviewed custom capture sources assigned to train only; source groups do not prove separate identities",
        "split_policy":"public train/val/test frozen; custom holdout matches excluded; no independent custom test claimed",
        "identity_disjoint":"unverified for public/custom; phone and multi may contain the same people; all custom remains train only",
        "box_exclusions_source":str(Path(box_exclusions).resolve()) if box_exclusions else None,
        "box_exclusions_sha256":hashlib.sha256(cleanup_bytes).hexdigest() if cleanup_bytes else None,
        "image_exclusions_source":str(Path(image_exclusions).resolve()) if image_exclusions else None,
        "image_exclusions_sha256":hashlib.sha256(image_qc_bytes).hexdigest() if image_qc_bytes else None,
        "bbox_context_source":str(Path(bbox_context).resolve()) if bbox_context else None,
        "bbox_context_sha256":hashlib.sha256(context_bytes).hexdigest() if context_bytes else None,
        "splits":{},"excluded_custom":exclusions,"duplicate_annotation_conflicts":annotation_conflicts,"prepared_statistics":{}}
    with tempfile.TemporaryDirectory(prefix="baseline-",dir=output.parent) as tmp:
        stage=Path(tmp)
        for split,items in public_manifest["splits"].items():
            (stage/"images"/split).mkdir(parents=True); (stage/"labels"/split).mkdir(parents=True)
            manifest["splits"][split]=[]
            for entry in items:
                source=public/"images"/split/entry["image"]
                if hashlib.sha256(source.read_bytes()).hexdigest()!=entry["source_sha256"]: raise ValueError("Public image hash mismatch")
                label=public/"labels"/split/(source.stem+".txt")
                name="public_"+source.name
                shutil.copy2(source,stage/"images"/split/name)
                shutil.copy2(label,stage/"labels"/split/(Path(name).stem+".txt"))
                original={**entry,"image":name,"source_kind":"public","label_status":"source_annotation_audited"}
                manifest["splits"][split].append(original)
                rows.append({"image":name,"split":split,"source_kind":"public","source_file":str(source),"source_sha256":entry["source_sha256"],
                    "label_sha256":hashlib.sha256(label.read_bytes()).hexdigest(),"group_id":"public_hash_groups","label_status":"source_annotation_audited"})
        for item in staged_rows:
            row=item["row"]; source_kind=item["source_kind"]; custom_group=item["group_id"]
            source=item["review"]/row["processed_file"]; name=source.name
            shutil.copy2(source,stage/"images/train"/name)
            label=stage/"labels/train"/(source.stem+".txt")
            label.write_text(yolo_lines(row),encoding="utf-8")
            manifest["splits"]["train"].append({"image":name,"source_kind":source_kind,"source_image":row["source_file"],
                "source_sha256":row["source_sha256"],"processed_sha256":row["processed_sha256"],"objects":len(row["boxes"]),
                "class_counts":dict(Counter(b["class_id"] for b in row["boxes"])),"label_status":"verified",
                "reviewer":row["reviewer"],"reviewed_at":row["reviewed_at"],"group_id":custom_group,"human_reviewed":not bool(row.get("context_expansion")),"base_human_reviewed":True,"context_expansion":row.get("context_expansion"),"review_source":str(item["review"]),"non_person_cleanup":row.get("non_person_cleanup")})
            rows.append({"image":name,"split":"train","source_kind":source_kind,"source_file":row["source_file"],"source_sha256":row["source_sha256"],
                "label_sha256":hashlib.sha256(label.read_bytes()).hexdigest(),"group_id":custom_group,"label_status":"verified"})
        config={"path":output.as_posix(),"train":"images/train","val":"images/val","test":"images/test","nc":3,"names":CLASSES}
        (stage/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
        with (stage/"manifest.csv").open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        manifest["non_person_boxes_removed_from_export"]=sum(len(item["row"].get("non_person_cleanup",{}).get("removed_boxes",[])) for item in staged_rows)
        manifest["custom_accepted"]=len(staged_rows)
        manifest["custom_excluded"]=len(exclusions)
        manifest["custom_accepted_by_source"]=dict(Counter(item["source_kind"] for item in staged_rows))
        manifest["custom_excluded_by_source"]=dict(Counter(item["source_kind"] for item in exclusions))
        (stage/"README.md").write_text(f"# {output.name}\n\nPublic test/val are preserved; reviewed custom images belong to train only.\n"
            f"Reviewed custom accepted: {len(staged_rows)}; excluded/pending: {len(exclusions)}.\n" +
            ("Reviewed ESP32 captures are train/development data; new captures required for independent camera evaluation.\n" if allow_camera_train else "No ESP32 evaluation images are included. Test is never used for tuning.\n") +
            "Custom sources remain train only; phone/multi identities and capture sessions may overlap. No independent custom test is claimed.\n",encoding="utf-8")
        # Audit staging with a temporary YAML path, before publishing output.
        temporary_config={**config,"path":stage.as_posix()}
        (stage/"validation.yaml").write_text(yaml.safe_dump(temporary_config),encoding="utf-8")
        report=audit_prepared(stage/"validation.yaml")
        (stage/"validation.yaml").unlink()
        if report["error_count"]: raise ValueError(f"Baseline staging audit failed: {report['issues'][:5]}")
        manifest["prepared_statistics"]={"images":report["total_images"],"bounding_boxes":report["total_boxes"],
            "class_distribution":{cls:sum(s["class_distribution"][cls] for s in report["splits"].values()) for cls in CLASSES.values()}}
        write_json(stage/"split_manifest.json",manifest)
        shutil.move(str(stage),output)
    final=audit_prepared(output/"data.yaml")
    write_json(reports/"baseline_dataset_validation.json",final)
    write_json(reports/"baseline_dataset_manifest.json",{k:v for k,v in manifest.items() if k!="splits"})
    print(json.dumps({"custom_accepted":len(staged_rows),"custom_excluded":len(exclusions),"splits":final["splits"],
        "error_count":final["error_count"],"warning_count":final["warning_count"]},indent=2))
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public",type=Path,default=ROOT/"data/processed/facemask_public_yolo")
    parser.add_argument("--review",type=Path,action="append",help="Repeat for multiple reviewed sources; defaults to single_person_review_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"data/processed/facemask_baseline_v1")
    parser.add_argument("--reports",type=Path,help="New report directory; defaults to dataset output")
    parser.add_argument("--box-exclusions",type=Path,help="Exact visually audited non-person box exclusions; never modifies source review")
    parser.add_argument("--image-exclusions",type=Path,help="Snapshot-pinned whole-image QC exclusions; preserve original annotations")
    parser.add_argument("--allow-camera-train",action="store_true",help="Explicitly use human-verified camera sources for train; they cease to be independent evaluation")
    parser.add_argument("--bbox-context",type=Path,help="Snapshot-pinned user-authorized incorrect-mask context export; base review preserved")
    args=parser.parse_args(); build(args.public,args.review or [ROOT/"data/processed/single_person_review_v1"],args.output,args.reports,args.box_exclusions,args.image_exclusions,args.allow_camera_train,args.bbox_context)

if __name__=="__main__": main()
