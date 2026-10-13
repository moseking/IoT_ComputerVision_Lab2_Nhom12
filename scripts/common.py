"""Shared paths, classes and JSON serialization."""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".runtime/ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime/ultralytics"))
os.environ["YOLO_AUTOINSTALL"] = "false"
CLASSES = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
CLASS_IDS = {name: index for index, name in CLASSES.items()}
# Preserve the original Kaggle folder supplied by the user.
RAW = ROOT / "data/raw/facemask_detection"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def device_name(requested="auto"):
    import torch
    return ("0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested


def metrics_dict(metrics):
    box = metrics.box
    measured = {int(c): i for i, c in enumerate(box.ap_class_index)}
    per_class = {}
    for class_id, name in CLASSES.items():
        index = measured.get(class_id)
        if index is None:
            per_class[name] = {"class_id": class_id, "precision": None, "recall": None,
                               "mAP50": None, "mAP50-95": None, "status": "no_ground_truth"}
        else:
            p, r, ap50, ap = box.class_result(index)
            per_class[name] = {"class_id": class_id, "precision": finite(p), "recall": finite(r),
                               "mAP50": finite(ap50), "mAP50-95": finite(ap), "status": "measured"}
    return {"precision": finite(box.mp), "recall": finite(box.mr), "mAP50": finite(box.map50),
            "mAP50-95": finite(box.map), "per_class": per_class,
            "framework_metrics": {str(k): finite(v) for k, v in metrics.results_dict.items()}}
