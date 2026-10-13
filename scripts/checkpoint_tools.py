"""Publish completed runs without silently overwriting an existing model."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def publish_checkpoints(best, last, metadata, destination, replace=False):
    best, last, destination = Path(best), Path(last), Path(destination)
    if metadata.get("epochs_completed", 0) <= 1:
        raise ValueError("One-epoch smoke checkpoints cannot be published for demo")
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("Both completed best and last checkpoints are required")
    destination.mkdir(parents=True, exist_ok=True)
    names = ("face_mask_best.pt", "face_mask_last.pt", "model_metadata.json")
    existing = [destination/name for name in names if (destination/name).exists()]
    if existing and not replace:
        raise FileExistsError("Published checkpoint exists; use --replace-published explicitly to archive and replace")
    if existing:
        archive = destination / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, archive/path.name)
    sources = [(best, names[0]), (last, names[1])]
    for source, name in sources:
        temporary = destination/(name+".pending")
        shutil.copy2(source, temporary)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(temporary.read_bytes()).digest():
            raise OSError("Checkpoint copy verification failed")
        os.replace(temporary, destination/name)
    temporary = destination/(names[2]+".pending")
    temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temporary,destination/names[2])
