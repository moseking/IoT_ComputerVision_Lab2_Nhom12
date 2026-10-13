"""Append only unseen local captures as drafts; never verify labels or train."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

try:
    from .common import ROOT, timestamp, write_json
    from .custom_dataset import FORMATS, prepare
except ImportError:
    from common import ROOT, timestamp, write_json
    from custom_dataset import FORMATS, prepare


def import_new(folder, manifest):
    folder = Path(folder).resolve()
    if (folder / "training_use.json").exists():
        raise ValueError("This review was used for training; create a new review queue")
    config = manifest["incremental_import"]
    source = Path(config["source"]).resolve()
    known_paths, known_hashes = set(), set()
    for review in config["known_reviews"]:
        previous = json.loads((Path(review) / "manifest.json").read_text(encoding="utf-8"))
        for row in previous["images"]:
            if row.get("source_file"):
                known_paths.add(str(Path(row["source_file"]).resolve()).casefold())
            if row.get("source_sha256"):
                known_hashes.add(row["source_sha256"])
    for row in manifest["images"]:
        known_paths.add(str(Path(row["source_file"]).resolve()).casefold())
        known_hashes.add(row["source_sha256"])
    selected, seen = [], set()
    duplicates = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FORMATS:
            continue
        if not path.resolve().is_relative_to(source):
            raise ValueError("Capture file escapes configured source")
        if str(path.resolve()).casefold() in known_paths:
            continue  # A previously reviewed filename is never reinterpreted.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in known_hashes or digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        selected.append(path)
    if selected:
        batch_name = "batch_" + uuid.uuid4().hex
        stage = folder / "imports" / batch_name
        batch = prepare(source, stage, Path(config["detector"]), source_kind="capture",
                        source_paths=selected)
        for row in batch["images"]:
            row["index"] = len(manifest["images"])
            row["human_reviewed"] = False
            row["group_id"] = batch_name
            if row.get("processed_file"):
                row["processed_file"] = (Path("imports") / batch_name / row["processed_file"]).as_posix()
            manifest["images"].append(row)
        manifest.setdefault("import_batches", []).append({
            "created": timestamp(), "folder": "imports/" + batch_name,
            "source_images": len(selected), "manifest_sha256": hashlib.sha256(
                (stage / "manifest.json").read_bytes()).hexdigest()})
    manifest["statistics"].update(
        source_images=len(manifest["images"]),
        conversion_success=sum(r.get("conversion_status") == "success" for r in manifest["images"]),
        conversion_failed=sum(r.get("conversion_status") == "failed" for r in manifest["images"]),
        verified_images=sum(r.get("label_status") == "verified" for r in manifest["images"]))
    config["last_scan"] = timestamp()
    pending = folder / ".import_pending.json"
    write_json(pending, manifest)
    pending.replace(folder / "manifest.json")
    return {"imported": len(selected), "duplicate_files_skipped": duplicates,
            "total": len(manifest["images"])}


def initialize(source, output, reviews, detector):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Review queue already exists; use its import button")
    for review in reviews:
        if output == Path(review).resolve():
            raise ValueError("New review must be separate from previous annotations")
    manifest = prepare(source, output, detector, source_kind="capture", source_paths=[])
    manifest["incremental_import"] = {"source": str(source),
        "known_reviews": [str(Path(p).resolve()) for p in reviews], "detector": str(Path(detector).resolve()),
        "policy": "Only unseen files become drafts. Device identity is not independently verified. No automatic training."}
    return import_new(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--known-review", type=Path, action="append", default=[])
    parser.add_argument("--detector", type=Path, default=ROOT / ".runtime/face_detector/face_detection_yunet_2023mar.onnx")
    args = parser.parse_args()
    print(json.dumps(initialize(args.source, args.output, args.known_review, args.detector)))


if __name__ == "__main__":
    main()
