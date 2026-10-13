"""Convert VOC to YOLO with deterministic grouped train/val/test splits."""
import argparse
import hashlib
import math
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .common import ROOT, RAW, CLASSES, write_json, timestamp
    from .dataset_tools import audit, report_markdown, yolo_lines, EXTENSIONS
except ImportError:
    from common import ROOT, RAW, CLASSES, write_json, timestamp
    from dataset_tools import audit, report_markdown, yolo_lines, EXTENSIONS


def make_groups(records, near_duplicate_distance, external_groups):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    pixels, sessions = {}, {}
    for i, record in enumerate(records):
        key = record["pixel_hash"]
        if key in pixels:
            union(i, pixels[key])
        pixels[key] = i
        session = external_groups.get(Path(record["image"]).name)
        if session is not None:
            if str(session) in sessions:
                union(i, sessions[str(session)])
            sessions[str(session)] = i
        if near_duplicate_distance >= 0:
            for j in range(i):
                if (record["dhash"] ^ records[j]["dhash"]).bit_count() <= near_duplicate_distance:
                    union(i, j)
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    return list(groups.values())


def split_groups(groups, ratios, seed):
    if len(groups) < 3:
        raise ValueError("Need at least 3 independent groups for train/val/test")
    rng = random.Random(seed)
    groups = groups.copy()
    rng.shuffle(groups)
    total_classes = Counter(b["class_id"] for g in groups for r in g for b in r["boxes"])
    total_images = sum(len(g) for g in groups)
    groups.sort(key=lambda g: sum(1 / max(1, total_classes[b["class_id"]]) for r in g for b in r["boxes"]), reverse=True)
    splits = {key: [] for key in ratios}
    counts = {key: Counter() for key in ratios}
    sizes = Counter()
    # Greedy balancing of image count and rare-class counts at group level.
    for index, group in enumerate(groups):
        gc = Counter(b["class_id"] for r in group for b in r["boxes"])
        empty = [s for s in ratios if not splits[s]]
        candidates = empty if len(groups) - index == len(empty) else list(ratios)
        def cost(split):
            score = 0.0
            for name, ratio in ratios.items():
                n = sizes[name] + (len(group) if split == name else 0)
                score += (n - total_images * ratio) ** 2 / max(1, total_images * ratio)
                for c in CLASSES:
                    value = counts[name][c] + (gc[c] if split == name else 0)
                    target = total_classes[c] * ratio
                    score += (value - target) ** 2 / max(1, target)
            return score
        selected = min(candidates, key=cost)
        splits[selected].extend(group)
        sizes[selected] += len(group)
        counts[selected].update(gc)
    return splits



def validate_output(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if raw == output or raw.is_relative_to(output) or output.is_relative_to(raw):
        raise ValueError("Output must be disjoint from the raw dataset (neither inside nor above it)")
    for name in ("images", "labels", "data.yaml", "split_manifest.json", "README.md"):
        path = output/name
        if path.is_symlink() or (path.exists() and not path.resolve().is_relative_to(output)):
            raise ValueError(f"Generated target escapes output directory: {path}")
    return raw,output


def plan_conversion(raw, output, *, train_ratio=0.7, val_ratio=0.15, seed=42,
                    near_duplicate_distance=4, groups_path=None, coordinates="zero-based",
                    repair_boxes=False, skip_invalid=False, duplicate_policy="error"):
    raw,output=validate_output(raw,output)
    ratios={"train":train_ratio,"val":val_ratio,"test":1-train_ratio-val_ratio}
    if any(not math.isfinite(v) or v<=0 for v in ratios.values()):
        raise ValueError("All split ratios must be finite and >0")
    if duplicate_policy not in {"error","exclude"}:
        raise ValueError("Unsupported duplicate policy")
    if not -1 <= near_duplicate_distance <= 64:
        raise ValueError("dHash distance must be -1..64")
    records,report=audit(raw,repair_boxes,coordinates)
    valid_count=len(records)
    conflicts={image for group in report["conflicting_duplicate_image_groups"] for image in group}
    if duplicate_policy=="exclude":
        records=[r for r in records if r["image"] not in conflicts]
    unresolved=[i for i in report["issues"] if i["severity"]=="error" and not
                (duplicate_policy=="exclude" and i["kind"]=="conflicting_duplicate_labels")]
    blocking=bool((unresolved and not skip_invalid) or (conflicts and duplicate_policy=="error"))
    if not records:
        blocking=True
    selected={r["image"] for r in records}
    discarded=[]
    by_image=defaultdict(list)
    for issue in report["issues"]:
        if issue.get("image"):
            by_image[issue["image"]].append(issue["kind"])
    for path in sorted((raw/"images").rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS and str(path) not in selected:
            reasons=sorted(set(by_image[str(path)]))
            if str(path) in conflicts:
                reasons.append("conflicting_duplicate_labels")
            discarded.append({"image":path.relative_to(raw/"images").as_posix(),
                              "source":str(path),"reasons":reasons or ["invalid_or_unpaired_annotation"]})
    external={}
    if groups_path:
        external=json.loads(Path(groups_path).read_text(encoding="utf-8"))
        if not isinstance(external,dict) or not all(isinstance(k,str) and isinstance(v,(str,int)) for k,v in external.items()):
            raise ValueError("groups.json must map image filenames to person/session IDs")
    groups=make_groups(records,near_duplicate_distance,external)
    if len(groups)<3:
        blocking=True
        splits={name:[] for name in ratios}
    else:
        splits=split_groups(groups,ratios,seed)
    counts=Counter(b["class_id"] for r in records for b in r["boxes"])
    split_stats={}
    for name,items in splits.items():
        split_counts=Counter(b["class_id"] for r in items for b in r["boxes"])
        split_stats[name]={"images":len(items),"bounding_boxes":sum(split_counts.values()),
                          "class_distribution":{label:split_counts[cls] for cls,label in CLASSES.items()}}
    summary={"timestamp":timestamp(),"raw_path":str(raw),"output":str(output),"coordinates":coordinates,
             "repair_boxes":repair_boxes,"invalid_policy":"skip_entire_image" if skip_invalid else "error",
             "duplicate_policy":duplicate_policy,"total_source_images":report["total_images"],
             "total_source_xml":report["total_annotations"],"source_objects":report["raw_objects"],
             "source_class_distribution":report["raw_class_distribution"],
             "accepted_images":len(records),"label_files":len(records),
             "skipped_images":report["total_images"]-len(records),"invalid_image_count":report["total_images"]-valid_count,
             "invalid_xml_files":report["invalid_xml_files"],
             "excluded_conflicting_images":valid_count-len(records) if duplicate_policy=="exclude" else 0,
             "bounding_boxes":sum(counts.values()),"class_distribution":{label:counts[cls] for cls,label in CLASSES.items()},
             "valid_background_images":sum(not r["boxes"] for r in records),
             "discarded_images":discarded,"splits":split_stats,"group_count":len(groups),"seed":seed,
             "ratios":ratios,"near_duplicate_distance":near_duplicate_distance,"blocking":blocking,
             "source_error_count":report["error_count"],"source_warning_count":report["warning_count"],
             "identity_disjoint":"unverified: no person identities provided" if not groups_path else "only supplied group metadata and hash heuristics",
             "test_policy":"test never used for checkpoint selection or hyperparameter tuning"}
    return records,report,splits,summary


def conversion_markdown(summary):
    lines=["# Chuyển đổi public VOC sang YOLO", "", f"Nguồn: `{summary['raw_path']}`",
           f"Đầu ra: `{summary['output']}`", "",
           f"- Chấp nhận: **{summary['accepted_images']} ảnh / {summary['label_files']} TXT / {summary['bounding_boxes']} box**.",
           f"- Loại khỏi output: **{summary['skipped_images']} ảnh**, gồm {summary['invalid_image_count']} ảnh annotation/ghép cặp lỗi",
           f"  và {summary['excluded_conflicting_images']} ảnh hợp lệ nhưng pixel trùng nhãn mâu thuẫn.",
           f"- invalid_policy={summary['invalid_policy']}; duplicate_policy={summary['duplicate_policy']}; repair_boxes={summary['repair_boxes']}.",
           f"- Seed {summary['seed']}; dHash distance <= {summary['near_duplicate_distance']}; group_count={summary['group_count']}.",
           "", "| Split | Ảnh | Box | with_mask | without_mask | incorrect |", "|---|---:|---:|---:|---:|---:|"]
    for name,data in summary["splits"].items():
        c=data["class_distribution"]
        lines.append(f"| {name} | {data['images']} | {data['bounding_boxes']} | {c['with_mask']} | {c['without_mask']} | {c['mask_weared_incorrect']} |")
    lines += ["", f"Ảnh nền hợp lệ (XML không có object): {summary['valid_background_images']}.",
              "Không coi ảnh có object lỗi là ảnh nền. Không sửa raw, không tự train.", "",
              "## Ảnh loại và lý do", "", "| Ảnh | Lý do |", "|---|---|"]
    lines.extend(f"| {item['image']} | {', '.join(item['reasons'])} |" for item in summary["discarded_images"])
    lines += ["", "## Giới hạn",
              "Ảnh exact/near duplicate được group cùng split; dHash chỉ là heuristic, có thể bỏ sót crop/ảnh gần giống.",
              "Không có ID người nên không chứng minh identity-disjoint. Cung cấp --groups cho ảnh cùng người/phiên.",
              "Test giữ riêng; không dùng chọn checkpoint hay threshold.",
              "Số lượng lớp ít/không có background có thể làm model bỏ sót hoặc false positive.",
              "Dataset cũ tại dataset/ không bị ghi đè; bộ strict này không dùng sửa box của bản cũ.", ""]
    return "\n".join(lines)


def prepare(raw, output, *, dry_run=False, overwrite=False, audit_output=None, conversion_output=None, **options):
    records,report,splits,summary=plan_conversion(raw,output,**options)
    raw,output=validate_output(raw,output)
    # Validate report destinations before writing anything.
    for path in (audit_output,conversion_output):
        if path is not None:
            resolved=Path(path).resolve()
            if resolved==raw or resolved.is_relative_to(raw) or resolved.is_relative_to(output):
                raise ValueError("Report destinations must be outside raw and generated output")
    if dry_run:
        print(json.dumps({**summary,"dry_run":True,"output_exists":output.exists()},ensure_ascii=False,indent=2))
        return report,summary
    if audit_output:
        audit_output=Path(audit_output)
        write_json(audit_output,report)
        audit_output.with_suffix(".md").write_text(report_markdown(report),encoding="utf-8")
    if conversion_output:
        conversion_output=Path(conversion_output)
        write_json(conversion_output,summary)
        conversion_output.with_suffix(".md").write_text(conversion_markdown(summary),encoding="utf-8")
    if summary["blocking"]:
        raise ValueError("Audit blocks conversion. Review reports; explicitly choose skip/exclude or fix source annotations in a separate reviewed copy.")
    generated=("images","labels","data.yaml","split_manifest.json","README.md")
    if not overwrite and any((output/name).exists() for name in generated):
        raise FileExistsError("Output already exists; default does not overwrite. Use --overwrite explicitly for generated files.")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-",dir=output.parent) as temporary:
        staged=Path(temporary)
        manifest={"generator":"face_mask_voc_prepare","timestamp":timestamp(),"seed":summary["seed"],
                  "raw":str(raw),"coordinates":summary["coordinates"],"ratios":summary["ratios"],
                  "near_duplicate_distance":summary["near_duplicate_distance"],"group_count":summary["group_count"],
                  "statistics":report,"prepared_statistics":{
                    "images":summary["accepted_images"],"bounding_boxes":summary["bounding_boxes"],
                    "class_distribution":summary["class_distribution"],
                    "single_face_images":sum(len(r["boxes"])==1 for r in records),
                    "multi_face_images":sum(len(r["boxes"])>1 for r in records),
                    "background_images":summary["valid_background_images"],
                    "excluded_conflicting_images":summary["excluded_conflicting_images"]},
                  "splits":{},"discarded_images":summary["discarded_images"],
                  "duplicate_policy":summary["duplicate_policy"],"invalid_policy":summary["invalid_policy"],
                  "leakage_scope":summary["identity_disjoint"]}
        observed={}
        for split,items in splits.items():
            image_dir,label_dir=staged/"images"/split,staged/"labels"/split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            manifest["splits"][split]=[]
            for record in items:
                source=Path(record["image"])
                annotation=Path(record["annotation"])
                if hashlib.sha256(source.read_bytes()).hexdigest()!=record["sha256"] or hashlib.sha256(annotation.read_bytes()).hexdigest()!=record["annotation_sha256"]:
                    raise ValueError(f"Source changed after audit: {source.name}")
                destination=image_dir/source.name
                shutil.copy2(source,destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=record["sha256"]:
                    raise OSError("Image copy checksum mismatch")
                (label_dir/(source.stem+".txt")).write_text(yolo_lines(record),encoding="utf-8")
                previous=observed.setdefault(record["pixel_hash"],split)
                if previous!=split:
                    raise ValueError("Exact duplicate crossed split")
                manifest["splits"][split].append({"image":source.name,"source_image":str(source),
                    "source_annotation":str(annotation),"source_sha256":record["sha256"],
                    "annotation_sha256":record["annotation_sha256"],"pixel_hash":record["pixel_hash"],
                    "dhash":record["dhash"],"objects":len(record["boxes"]),
                    "class_counts":dict(Counter(b["class_id"] for b in record["boxes"]))})
        import yaml
        (staged/"data.yaml").write_text(yaml.safe_dump({"path":output.as_posix(),"train":"images/train",
            "val":"images/val","test":"images/test","nc":3,"names":CLASSES},sort_keys=False),encoding="utf-8")
        write_json(staged/"split_manifest.json",manifest)
        (staged/"README.md").write_text(conversion_markdown(summary)+"\nNguồn và class lấy từ XML gốc; không suy đoán class từ folder.\n"
            "Mỗi ảnh có đúng một TXT với một dòng mỗi bbox; chỉ XML zero-object hợp lệ mới tạo TXT rỗng.\n"
            "Mặc định zero-based continuous; nếu nguồn VOC one-based inclusive, truyền --coordinates one-based-inclusive.\n",encoding="utf-8")
        output.mkdir(parents=True,exist_ok=True)
        for name in generated:
            destination=output/name
            # Resolved absolute target must stay in the validated generated directory.
            if destination.is_symlink() or not destination.resolve().is_relative_to(output):
                raise ValueError("Unsafe generated replacement target")
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            shutil.move(str(staged/name),destination)
    print(json.dumps({k:summary[k] for k in ("accepted_images","skipped_images","bounding_boxes","class_distribution","splits")},ensure_ascii=False,indent=2))
    return report,summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--output",type=Path,default=ROOT/"dataset")
    parser.add_argument("--train-ratio",type=float,default=0.7)
    parser.add_argument("--val-ratio",type=float,default=0.15)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--near-duplicate-distance",type=int,default=4)
    parser.add_argument("--groups",type=Path)
    parser.add_argument("--repair-boxes",action="store_true",help="Legacy explicit repair mode; every clip is recorded, strict converter never enables this")
    parser.add_argument("--duplicate-policy",choices=["error","exclude"],default="error")
    parser.add_argument("--skip-invalid",action="store_true",help="Discard entire invalid images, never drop individual boxes")
    parser.add_argument("--coordinates",choices=["zero-based","one-based-inclusive"],default="zero-based")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    _,summary=prepare(args.raw,args.output,train_ratio=args.train_ratio,val_ratio=args.val_ratio,
        seed=args.seed,near_duplicate_distance=args.near_duplicate_distance,groups_path=args.groups,
        repair_boxes=args.repair_boxes,coordinates=args.coordinates,duplicate_policy=args.duplicate_policy,
        skip_invalid=args.skip_invalid,overwrite=args.overwrite,dry_run=args.dry_run,
        audit_output=ROOT/"results/preparation_audit.json")
    return 1 if summary["blocking"] else 0


if __name__=="__main__":
    raise SystemExit(main())
