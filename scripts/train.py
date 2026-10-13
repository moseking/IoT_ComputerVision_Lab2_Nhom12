"""Train YOLOv8n from scratch by default; publish only completed training runs."""
import argparse
import hashlib
import json
from pathlib import Path
from common import ROOT, CLASSES, device_name, metrics_dict, timestamp, write_json
from audit_prepared_dataset import audit_prepared
from checkpoint_tools import publish_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/facemask_baseline_v1/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0, help="0 is safe on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, help="Initial LR; use --optimizer AdamW to avoid auto overriding LR")
    parser.add_argument("--optimizer", choices=["auto","SGD","AdamW"], default="auto")
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--run-name", default="face_mask")
    parser.add_argument("--init-checkpoint", type=Path, help="Fine-tune a completed face-mask checkpoint in a new run")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from last.pt")
    parser.add_argument("--pretrained", action="store_true", help="Optional fine tuning from COCO yolov8n.pt")
    parser.add_argument("--no-publish", action="store_true", help="For a smoke run: keep weights inside the run only")
    parser.add_argument("--replace-published", action="store_true", help="Archive existing published weights before replacement")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Prepared dataset missing. Run scripts/prepare_dataset.py first.")
    if args.lr0 is not None and not 0 < args.lr0 <= 1:
        parser.error("lr0 must be in (0,1]")
    if not 0 < args.lrf <= 1:
        parser.error("lrf must be in (0,1]")
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("epochs, imgsz and batch must be positive")
    if args.epochs <= 1 and not args.no_publish:
        parser.error("Smoke runs require --no-publish; one-epoch checkpoints cannot become demo models")
    published = ROOT / "models"
    if not args.no_publish and not args.replace_published and any((published/name).exists() for name in ("face_mask_best.pt", "face_mask_last.pt", "model_metadata.json")):
        parser.error("Published model already exists. Use --no-publish or --replace-published (archives old files)")
    audit_report = audit_prepared(args.data)
    audit_path = ROOT / "reports/training_data" / (timestamp().replace(":", "").replace("+", "_") + ".json")
    write_json(audit_path, audit_report)
    if audit_report["error_count"]:
        parser.error(f"Dataset integrity audit failed; see {audit_path}")
    import yaml
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = config["names"]
    mapping = dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
    if mapping != CLASSES:
        parser.error("Dataset classes do not match the required 0/1/2 mapping")
    from ultralytics import YOLO, __version__
    device = device_name(args.device)
    run_name = args.run_name + "_" + timestamp().replace(":", "").replace("+", "_")
    if args.init_checkpoint and (not args.init_checkpoint.is_file() or args.pretrained or args.resume):
        parser.error("--init-checkpoint needs a real checkpoint; cannot combine with --pretrained/--resume")
    if args.resume and (not args.resume.is_file() or args.pretrained):
        parser.error("--resume requires a real last.pt and cannot be combined with --pretrained")
    initialization = args.resume or args.init_checkpoint
    model = YOLO(str(initialization.resolve()) if initialization else ("yolov8n.pt" if args.pretrained else "yolov8n.yaml"))
    if args.resume and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
        parser.error("Checkpoint is completed/stripped; use --init-checkpoint for a NEW fine-tuning run")
    if initialization and {int(k):v for k,v in model.names.items()} != CLASSES:
        parser.error("Face-mask checkpoint classes must match required 0/1/2 mapping")
    initialization_check = {}
    if args.init_checkpoint:
        import torch
        expected_state = {k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
        def verify_initialization(trainer):
            actual = trainer.model.state_dict()
            mismatched = [k for k,v in expected_state.items() if k not in actual or
                          v.shape != actual[k].shape or not torch.equal(v, actual[k].detach().cpu())]
            initialization_check.update(source=str(args.init_checkpoint.resolve()),
                checked_tensors=len(expected_state),mismatched_tensors=mismatched,passed=not mismatched)
            write_json(Path(trainer.save_dir)/"initialization_verification.json",initialization_check)
            if mismatched:
                raise RuntimeError("Checkpoint initialization mismatch before training: " + str(mismatched[:5]))
            print(f"Verified checkpoint initialization: {len(expected_state)} tensors match before first optimizer step")
        model.add_callback("on_train_start", verify_initialization)
    print(f"Training on {device}, pretrained={args.pretrained}, initialization={initialization}")
    metrics = model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=device, workers=args.workers, seed=args.seed,
                          patience=args.patience, project=str(ROOT / "results/training"),
                          name=run_name, exist_ok=False, plots=True, save=True,
                          pretrained=bool(args.pretrained or initialization), deterministic=True, amp=False, resume=bool(args.resume),
                          optimizer=args.optimizer, lrf=args.lrf, **({"lr0":args.lr0} if args.lr0 is not None else {}))
    run = Path(model.trainer.save_dir)
    best, last = run / "weights/best.pt", run / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Training did not produce both best.pt and last.pt")
    manifest_path = args.data.parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metadata = {"model_name": "YOLOv8n Face Mask Detection", "version": run.name,
                "classes": CLASSES, "image_size": model.trainer.args.imgsz, "training_date": timestamp(),
                "training_mode": "resume" if args.resume else ("fine_tuning_checkpoint" if args.init_checkpoint else ("fine_tuning_COCO" if args.pretrained else "from_scratch")),
                "initialization": str(initialization) if initialization else ("official Ultralytics yolov8n.pt COCO" if args.pretrained else "yolov8n.yaml random initialization"),
                "initialization_verification": initialization_check or None,
                "checkpoint_selection": "best validation fitness; test never used for selection",
                "seed": model.trainer.args.seed, "workers": model.trainer.args.workers, "batch": model.trainer.args.batch, "patience": model.trainer.args.patience,
                "initialization_sha256": hashlib.sha256(initialization.read_bytes()).hexdigest() if initialization else None,
                "optimizer": type(model.trainer.optimizer).__name__, "learning_rate_initial": model.trainer.optimizer.param_groups[0].get("initial_lr", model.trainer.optimizer.defaults["lr"]),
                "optimizer_constructor_default_lr": model.trainer.optimizer.defaults["lr"],
                "lr0_requested": args.lr0, "lrf": model.trainer.args.lrf,
                "resume_from": str(args.resume) if args.resume else None,
                "ultralytics_version": __version__, "device": device,
                "epochs_requested": model.trainer.args.epochs, "epochs_completed": model.trainer.epoch + 1,
                "dataset_statistics": manifest.get("prepared_statistics", {}),
                "raw_dataset_statistics": manifest.get("statistics", {}),
                "split_images": {key: len(value) for key, value in manifest.get("splits", {}).items()},
                "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None,
                "metrics": metrics_dict(metrics), "metrics_split": "val",
                "weights_sha256": hashlib.sha256(best.read_bytes()).hexdigest(), "run_directory": str(run),
                "published": not args.no_publish,
                "smoke_run": model.trainer.epoch + 1 <= 1,
                "demo_eligible": False,
                "quality_status": "not_accepted_for_demo: requires held-out evaluation and live camera validation",
                "prepared_audit": audit_report}
    write_json(run / "metrics.json", metadata["metrics"])
    write_json(run / "model_metadata.json", metadata)
    if not args.no_publish:
        models = ROOT / "models"
        publish_checkpoints(best, last, metadata, models, args.replace_published)
        print("Published best/last weights and metadata to", models)
    print("Artifacts:", run)
    return 0


if __name__ == "__main__":
    # Required guard for PyTorch DataLoader on Windows.
    raise SystemExit(main())
