"""Strict public VOC -> YOLO entry point; reuses the project's audit and group split."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
try:
    from .common import ROOT, RAW
    from .prepare_dataset import prepare
except ImportError:
    from common import ROOT, RAW
    from prepare_dataset import prepare


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--output",type=Path,default=ROOT/"data/processed/facemask_public_yolo")
    parser.add_argument("--reports",type=Path,default=ROOT/"reports/dataset_audit")
    parser.add_argument("--invalid-policy",choices=["error","skip"],default="error",
                        help="skip excludes entire invalid images; never drops individual bad boxes")
    parser.add_argument("--duplicate-policy",choices=["error","exclude"],default="error",
                        help="exclude removes every member of conflicting pixel-duplicate groups from output only")
    parser.add_argument("--train-ratio",type=float,default=0.7)
    parser.add_argument("--val-ratio",type=float,default=0.15)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--near-duplicate-distance",type=int,default=4)
    parser.add_argument("--groups",type=Path)
    parser.add_argument("--coordinates",choices=["zero-based","one-based-inclusive"],default="zero-based")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(argv)
    try:
        _,summary=prepare(args.raw,args.output,train_ratio=args.train_ratio,val_ratio=args.val_ratio,
                          seed=args.seed,near_duplicate_distance=args.near_duplicate_distance,
                          groups_path=args.groups,coordinates=args.coordinates,repair_boxes=False,
                          skip_invalid=args.invalid_policy=="skip",duplicate_policy=args.duplicate_policy,
                          dry_run=args.dry_run,overwrite=args.overwrite,
                          audit_output=args.reports/"public_dataset_audit.json",
                          conversion_output=args.reports/"public_dataset_conversion.json")
    except (ValueError,OSError) as exc:
        print(f"Conversion stopped: {exc}",file=sys.stderr)
        return 1
    return 0 if not summary["blocking"] else 1


if __name__=="__main__":
    raise SystemExit(main())
