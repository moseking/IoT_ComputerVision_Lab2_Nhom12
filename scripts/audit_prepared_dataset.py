"""Re-audit actual YOLO images/labels before training; never repairs or re-splits."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image
try:
    from .common import ROOT, CLASSES, timestamp, write_json
    from .dataset_tools import dhash, EXTENSIONS
except ImportError:
    from common import ROOT, CLASSES, timestamp, write_json
    from dataset_tools import dhash, EXTENSIONS


def audit_prepared(data_path, near_distance=4):
    data_path = Path(data_path).resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    issues, hashes, samples, split_stats = [], {}, [], {}
    names = config.get("names", {})
    mapping = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if mapping != CLASSES:
        issues.append({"kind": "class_mapping", "message": repr(mapping)})
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    for split in ("train", "val", "test"):
        image_dir = root / config.get(split, "")
        label_dir = image_dir.parent.parent / "labels" / image_dir.name
        if not image_dir.is_dir() or not label_dir.is_dir():
            issues.append({"kind": "missing_split_directory", "split": split})
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        stems = Counter(p.stem for p in images)
        counts, faces = Counter(), []
        for stem, number in stems.items():
            if number != 1:
                issues.append({"kind": "basename_collision", "split": split, "image": stem})
        for stem in labels.keys() - stems.keys():
            issues.append({"kind": "orphan_label", "split": split, "label": stem})
        for path in images:
            try:
                with Image.open(path) as image:
                    image.load()
                    rgb = image.convert("RGB")
                    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
                    perceptual = dhash(rgb)
                if digest in hashes:
                    previous = hashes[digest]
                    issues.append({"kind": "exact_duplicate", "split": split, "image": path.name,
                                   "other": previous, "cross_split": previous["split"] != split})
                else:
                    hashes[digest] = {"split": split, "image": path.name}
                samples.append((split, path.name, perceptual))
            except Exception as exc:
                issues.append({"kind": "corrupt_image", "split": split, "image": path.name, "message": str(exc)})
            label = labels.get(path.stem)
            if label is None:
                issues.append({"kind": "missing_label", "split": split, "image": path.name})
                continue
            valid = 0
            for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError("Expected class_id cx cy width height")
                    cls = int(parts[0])
                    if cls not in CLASSES:
                        raise ValueError("Unknown class_id")
                    x, y, w, h = map(float, parts[1:])
                    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError("Non-finite/empty box")
                    # Tolerance permits 8-decimal normalization at exact image borders.
                    if min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7:
                        raise ValueError("Box extends outside image")
                    counts[cls] += 1
                    valid += 1
                except (ValueError, OverflowError) as exc:
                    issues.append({"kind": "invalid_label", "split": split, "label": label.name,
                                   "line": number, "message": str(exc)})
            faces.append(valid)
        split_stats[split] = {"images": len(images), "bounding_boxes": sum(counts.values()),
                             "class_distribution": {name: counts[k] for k, name in CLASSES.items()},
                             "zero_object_images": faces.count(0), "multi_object_images": sum(n > 1 for n in faces)}
        for cls, name in CLASSES.items():
            if not counts[cls]:
                issues.append({"kind": "missing_class", "split": split, "class": name})
    if near_distance >= 0:
        for index, (split, name, digest) in enumerate(samples):
            for other_split, other_name, other_digest in samples[:index]:
                if split != other_split and (digest ^ other_digest).bit_count() <= near_distance:
                    issues.append({"kind": "near_duplicate_leakage", "split": split, "image": name,
                                   "other_split": other_split, "other_image": other_name})
    # Exact duplicates inside one split are reported as warnings; crossing splits blocks training.
    errors = [item for item in issues if item["kind"] != "exact_duplicate" or item["cross_split"]]
    return {"timestamp": timestamp(), "data_yaml": str(data_path), "classes": mapping,
            "splits": split_stats, "total_images": sum(s["images"] for s in split_stats.values()),
            "total_boxes": sum(s["bounding_boxes"] for s in split_stats.values()),
            "error_count": len(errors), "warning_count": len(issues)-len(errors), "issues": issues,
            "near_duplicate_distance": near_distance,
            "test_policy": "test excluded from checkpoint selection/tuning; train uses val only",
            "identity_disjoint": "unverified: dataset has no person identity IDs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/data.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prepared_dataset_report.json")
    args = parser.parse_args()
    report = audit_prepared(args.data)
    write_json(args.output, report)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("splits", "total_images", "total_boxes", "error_count", "warning_count")}, indent=2))
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
