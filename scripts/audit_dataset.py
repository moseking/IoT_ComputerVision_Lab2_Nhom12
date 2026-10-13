"""Read-only VOC audit with JSON/TXT and optional Markdown reports."""
import argparse
from pathlib import Path
try:
    from .common import ROOT, RAW, write_json
    from .dataset_tools import audit, report_text, report_markdown
except ImportError:
    from common import ROOT, RAW, write_json
    from dataset_tools import audit, report_text, report_markdown


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=RAW)
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--report-name",default="dataset_report")
    parser.add_argument("--markdown",action="store_true")
    parser.add_argument("--coordinates",choices=["zero-based","one-based-inclusive"],default="zero-based")
    args=parser.parse_args(argv)
    if not args.report_name or Path(args.report_name).name != args.report_name:
        parser.error("--report-name must be a plain filename stem")
    # A report destination may not be inside or above the raw dataset.
    raw,output=args.raw.resolve(),args.output.resolve()
    if output==raw or output.is_relative_to(raw):
        parser.error("Report directory must be outside the raw dataset")
    try:
        _,report=audit(raw,coordinates=args.coordinates)
    except FileNotFoundError as exc:
        report={"raw_path":str(raw),"error_count":1,"warning_count":0,
                "issues":[{"kind":"missing_dataset","file":str(raw),"message":str(exc),"severity":"error"}]}
    write_json(output/(args.report_name+".json"),report)
    (output/(args.report_name+".txt")).write_text(report_text(report),encoding="utf-8")
    if args.markdown:
        (output/(args.report_name+".md")).write_text(report_markdown(report),encoding="utf-8")
    print(report_text(report))
    return 1 if report["error_count"] else 0


if __name__=="__main__":
    raise SystemExit(main())
