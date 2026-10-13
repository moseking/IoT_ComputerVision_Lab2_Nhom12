"""Evaluate validation or untouched test with measured metrics, per-image errors and artifacts."""
import argparse, csv, hashlib, json, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
try:
    from .common import CLASSES, ROOT, device_name, metrics_dict, timestamp, write_json
    from .audit_prepared_dataset import audit_prepared
except ImportError:
    from common import CLASSES, ROOT, device_name, metrics_dict, timestamp, write_json
    from audit_prepared_dataset import audit_prepared

def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.0

def match_errors(ground_truth,predictions,threshold=0.5):
    pairs=sorted([(overlap(g["bbox"],p["bbox"]),gi,pi) for gi,g in enumerate(ground_truth)
                  for pi,p in enumerate(predictions) if g["class_id"]==p["class_id"]],reverse=True)
    matched_g=set();matched_p=set()
    for score,gi,pi in pairs:
        if score>=threshold and gi not in matched_g and pi not in matched_p:
            matched_g.add(gi);matched_p.add(pi)
    return {"false_negatives":[g for i,g in enumerate(ground_truth) if i not in matched_g],
            "false_positives":[p for i,p in enumerate(predictions) if i not in matched_p],
            "true_positives":len(matched_g)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=ROOT/"models/face_mask_best.pt")
    parser.add_argument("--data",type=Path,default=ROOT/"data/processed/facemask_baseline_v1/data.yaml")
    parser.add_argument("--imgsz",type=int,default=512);parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--output",type=Path,default=ROOT/"results/evaluation/baseline_v1")
    parser.add_argument("--split",choices=["val","test"],default="test")
    parser.add_argument("--conf",type=float,default=0.35,help="Operational/qualitative confidence; mAP uses conf=.001")
    parser.add_argument("--agnostic-nms",action="store_true",help="Suppress duplicate face boxes across the mutually exclusive mask classes")
    args=parser.parse_args()
    if not args.model.is_file() or not args.data.is_file():parser.error("Model/data missing")
    if not 0<=args.conf<=1:parser.error("Confidence must be 0..1")
    output=args.output.resolve()
    if (output/"metrics.json").exists():parser.error("Evaluation output exists; choose a new output path")
    audited=audit_prepared(args.data)
    if audited["error_count"]:parser.error("Prepared dataset failed audit")
    import yaml,torch
    from ultralytics import YOLO
    config=yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_root=Path(config.get("path",args.data.parent))
    if not dataset_root.is_absolute():dataset_root=(args.data.parent/dataset_root).resolve()
    images=sorted((dataset_root/config[args.split]).glob("*"))
    images=[p for p in images if p.suffix.lower() in {".png",".jpg",".jpeg",".webp",".bmp"}]
    labels=dataset_root/"labels"/Path(config[args.split]).name
    model=YOLO(str(args.model.resolve()))
    if {int(k):v for k,v in model.names.items()}!=CLASSES:parser.error("Checkpoint mapping mismatch")
    device=device_name(args.device)
    output.mkdir(parents=True,exist_ok=True)
    measured=model.val(data=str(args.data.resolve()),split=args.split,imgsz=args.imgsz,batch=args.batch,
        device=device,workers=0,conf=.001,iou=.45,agnostic_nms=args.agnostic_nms,plots=True,project=str(output),name=args.split,exist_ok=False)
    report={"timestamp":timestamp(),"split":args.split,"model":str(args.model.resolve()),
        "model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"data_yaml":str(args.data.resolve()),
        "image_size":args.imgsz,"device":device,"metric_min_confidence":.001,"operational_confidence":args.conf,
        "nms_iou":.45,"nms_class_agnostic":args.agnostic_nms,"error_matching_iou":.5,"checkpoint_selection":"validation fitness only",
        "precision_recall_definition":"Ultralytics metrics at its evaluation operating point; separate counts at operational confidence",
        **metrics_dict(measured),"framework_speed_ms":measured.speed}
    gt_counts=Counter();pred_counts=Counter();fp_counts=Counter();fn_counts=Counter();items=[];latencies=[]
    examples=output/"error_examples";examples.mkdir(exist_ok=True)
    cuda=device!="cpu" and torch.cuda.is_available()
    for index,path in enumerate(images):
        with Image.open(path) as original:image=original.convert("RGB")
        gt=[]
        for line in (labels/(path.stem+".txt")).read_text(encoding="utf-8").splitlines():
            cls,xc,yc,w,h=map(float,line.split())
            gt.append({"class_id":int(cls),"bbox":[(xc-w/2)*image.width,(yc-h/2)*image.height,(xc+w/2)*image.width,(yc+h/2)*image.height]})
        if index==0:model.predict(image,conf=args.conf,iou=.45,agnostic_nms=args.agnostic_nms,imgsz=args.imgsz,device=device,verbose=False)
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter()
        result=model.predict(image,conf=args.conf,iou=.45,agnostic_nms=args.agnostic_nms,imgsz=args.imgsz,device=device,verbose=False)[0]
        if cuda:torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000;latencies.append(elapsed)
        predicted=[{"class_id":int(box.cls.item()),"confidence":float(box.conf.item()),"bbox":box.xyxy[0].cpu().tolist()} for box in result.boxes]
        errors=match_errors(gt,predicted)
        gt_counts.update(x["class_id"] for x in gt);pred_counts.update(x["class_id"] for x in predicted)
        fp_counts.update(x["class_id"] for x in errors["false_positives"]);fn_counts.update(x["class_id"] for x in errors["false_negatives"])
        item={"image":path.name,"ground_truth":gt,"predictions":predicted,"inference_wall_ms":elapsed,**errors}
        if (errors["false_positives"] or errors["false_negatives"]) and len(list(examples.glob("*.jpg")))<16:
            canvas=image.copy();draw=ImageDraw.Draw(canvas)
            for box in gt:
                draw.rectangle(box["bbox"],outline="#18b86a",width=2);draw.text((box["bbox"][0],max(0,box["bbox"][1]-12)),f"GT {CLASSES[box['class_id']]}",fill="#18b86a")
            for box in predicted:
                draw.rectangle(box["bbox"],outline="#ff4141",width=2);draw.text((box["bbox"][0],min(image.height-12,box["bbox"][3])),f"P {CLASSES[box['class_id']]} {box['confidence']:.2f}",fill="#ff4141")
            canvas.save(examples/(path.stem+".jpg"),quality=90);item["example"]=str(examples/(path.stem+".jpg"))
        items.append(item)
    report["counts_at_operational_confidence"]={name:{"ground_truth":gt_counts[cls],"predicted":pred_counts[cls],
        "false_positive":fp_counts[cls],"false_negative":fn_counts[cls]} for cls,name in CLASSES.items()}
    report["inference_wall_ms"]={"mean":float(np.mean(latencies)),"median":float(np.median(latencies)),
        "p95":float(np.percentile(latencies,95)),"samples":len(latencies),"warmup_excluded":True,
        "definition":"preprocess+forward+NMS with CUDA synchronization, decode excluded"}
    write_json(output/"predictions.json",{"split":args.split,"confidence":args.conf,"images":items})
    write_json(output/"metrics.json",report)
    with (output/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=["class_id","class_name","precision","recall","mAP50","mAP50-95","ground_truth","predicted","false_positive","false_negative"])
        writer.writeheader()
        for name,data in report["per_class"].items():
            writer.writerow({**{k:data[k] for k in ("class_id","precision","recall","mAP50","mAP50-95")},"class_name":name,**report["counts_at_operational_confidence"][name]})
    lines=[f"# Evaluation {args.split}","",f"Checkpoint: {args.model.resolve()}",f"Dataset: {args.data.resolve()}",
        f"Device: {device}; imgsz={args.imgsz}; operational conf={args.conf}; mAP min conf=0.001.","",
        "| Class | Precision | Recall | mAP50 | mAP50-95 | GT | Pred | FP | FN |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,data in report["per_class"].items():
        counts=report["counts_at_operational_confidence"][name]
        lines.append("| "+name+" | "+" | ".join(f"{data[k]:.4f}" if data[k] is not None else "unavailable" for k in ("precision","recall","mAP50","mAP50-95"))+
                     " | "+" | ".join(str(counts[k]) for k in ("ground_truth","predicted","false_positive","false_negative"))+" |")
    lines.extend(["",f"Overall: precision={report['precision']:.4f}, recall={report['recall']:.4f}, mAP50={report['mAP50']:.4f}, mAP50-95={report['mAP50-95']:.4f}.",
        f"Inference mean={report['inference_wall_ms']['mean']:.2f} ms, p95={report['inference_wall_ms']['p95']:.2f} ms.",
        "Green boxes=GT, red=prediction in error_examples. Confusion/PR plots are in the framework split directory.",
        "FP/FN counts use conf=0.35 (or explicit --conf) and class-aware greedy IoU matching; P/R above follow Ultralytics metric operating point.",
        "Rare incorrect class and domain shift need additional reviewed camera data. No camera or personal held-out test claimed."])
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (output/"results.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("split","precision","recall","mAP50","mAP50-95","per_class","inference_wall_ms")},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
