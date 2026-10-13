"""Local image audit/HEIC conversion and face proposals. Proposals NEVER mean verified labels."""
import argparse, hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
try:
    from .common import ROOT, CLASSES, timestamp, write_json
    from .dataset_tools import dhash
except ImportError:
    from common import ROOT, CLASSES, timestamp, write_json
    from dataset_tools import dhash

_IMAGE_OPEN = Image.open
FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
FOLDER_IDS = {"with_mask": 0, "without_mask": 1, "wore_incorrect": 2, "mask_weared_incorrect": 2}

def standardize(path, max_side=1600):
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise ValueError("HEIC requires pillow-heif: .venv\\Scripts\\python.exe -m pip install pillow-heif") from exc
        register_heif_opener()
    with _IMAGE_OPEN(path) as original:
        original.load()
        original_size = original.size
        orientation = int(original.getexif().get(274, 1))
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        oriented_size = rgb.size
        if max_side > 0:
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        rgb.info.clear()
        return rgb, {"original_size": list(original_size), "exif_orientation": orientation,
                     "oriented_size": list(oriented_size), "processed_size": list(rgb.size)}

def face_proposals(image, detector):
    width, height = image.size
    scale = min(1.0, 640/max(width,height))
    resized = image.resize((max(1,round(width*scale)), max(1,round(height*scale))))
    bgr = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    proposals = []
    if faces is not None:
        for face in faces:
            x, y, w, h = [float(v)/scale for v in face[:4]]
            # Proposal bounding boxes may overshoot; clamp proposals and flag this explicitly.
            original = [x,y,x+w,y+h]
            box = [max(0,x),max(0,y),min(width,x+w),min(height,y+h)]
            if box[2] > box[0] and box[3] > box[1]:
                proposals.append({"bbox":box,"detector_confidence":float(face[-1]),
                                  "proposal_clamped":box != original})
    return proposals

def pad_mask_context(bbox, width, height):
    """Draft context heuristic; humans still check the visible hanging mask."""
    x1,y1,x2,y2=map(float,bbox);w=x2-x1;h=y2-y1
    return [max(0,x1-.35*w),max(0,y1-.10*h),min(width,x2+.35*w),min(height,y2+.35*h)]

def validate_boxes(boxes, width, height, single_person=True):
    if not isinstance(boxes, list) or (single_person and len(boxes) != 1) or not boxes:
        raise ValueError("Expected exactly one face" if single_person else "At least one reviewed face is required")
    clean = []
    for box in boxes:
        cls = box.get("class_id")
        if isinstance(cls, bool) or not isinstance(cls, int) or cls not in CLASSES:
            raise ValueError("Class ID must be integer 0/1/2")
        values = box.get("bbox", [])
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("Expected [x1,y1,x2,y2]")
        x1,y1,x2,y2 = map(float, values)
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Reviewed bbox must have positive area and stay inside image")
        clean.append({"class_id":cls,"bbox":[x1,y1,x2,y2]})
    return clean

def contact_sheets(manifest, output, tile=(320,360), per_page=20):
    folder = Path(output)/"contact_sheets"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r for r in manifest["images"] if r.get("processed_file")]
    names = []
    for start in range(0,len(rows),per_page):
        page = Image.new("RGB",(tile[0]*4,tile[1]*5),(22,29,41))
        draw = ImageDraw.Draw(page)
        for index,row in enumerate(rows[start:start+per_page]):
            x,y = index%4*tile[0],index//4*tile[1]
            with _IMAGE_OPEN(Path(output)/row["processed_file"]) as im:
                im = im.convert("RGB")
                proposals = row.get("boxes",[])
                overlay = ImageDraw.Draw(im)
                for proposal in proposals:
                    overlay.rectangle(proposal["bbox"], outline="#29e677", width=5)
                # Face crop for reliable visual inspection; original context at top left.
                if proposals:
                    b=proposals[0]["bbox"]; pad=max(b[2]-b[0],b[3]-b[1])*.3
                    crop=im.crop((max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad))))
                else:
                    crop=im.copy()
                crop.thumbnail((310,280))
                page.paste(crop,(x+(320-crop.width)//2,y+40))
                thumb=im.copy(); thumb.thumbnail((82,70)); page.paste(thumb,(x+4,y+244))
                draw.text((x+5,y+4),f"{row['index']:03d} {CLASSES.get(row.get('intended_class_id'),'unknown')}",fill="white")
                draw.text((x+5,y+20),f"faces={row.get('face_count')} status={row['label_status']}",fill="#ffcb6b")
                draw.text((x+5,y+325),row["source_relative"][-42:],fill="#a9bdcf")
        name=f"contact_sheets/page_{start//per_page+1:02d}.jpg"
        page.save(Path(output)/name,quality=93)
        names.append(name)
    return names

def prepare(source, output, detector_path, single_person=True, max_side=1600, source_kind=None, source_paths=None):
    source_kind=source_kind or ("phone" if single_person else "capture")
    if source_kind not in {"phone","multi","esp32","capture"}: raise ValueError("Invalid source kind")
    source,output=Path(source).resolve(),Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source==output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Processed output must be separate from source")
    if (output/"manifest.json").exists():
        raise FileExistsError("Manifest already exists; choose a new version instead of overwriting reviews")
    detector=cv2.FaceDetectorYN.create(str(detector_path),"",(640,640),0.6,0.3,5000)
    output.mkdir(parents=True,exist_ok=True)
    (output/"images").mkdir(exist_ok=True)
    entries=[]; counts=Counter(); hashes=defaultdict(list); pixels=defaultdict(list)
    sources=sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in FORMATS) if source_paths is None else sorted(Path(p).resolve() for p in source_paths)
    if any(not p.is_file() or p.suffix.lower() not in FORMATS or not p.resolve().is_relative_to(source) for p in sources):
        raise ValueError("Selected images must exist inside configured source")
    for index,path in enumerate(sources):
        relative=path.relative_to(source).as_posix()
        counts[(path.parent.name,path.suffix.lower())]+=1
        row={"index":index,"id":hashlib.sha256(relative.encode()).hexdigest()[:16],"source_relative":relative,
             "source_file":str(path),"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
             "intended_class_id":FOLDER_IDS.get(path.parent.name),"label_status":"unreviewed",
             "source_kind":source_kind,"included_in_training":False,"group_id":f"{source_kind}_single_capture_01" if single_person else None}
        hashes[row["source_sha256"]].append(row["id"])
        try:
            image,size=standardize(path,max_side)
            row.update(size)
            file=f"images/{source_kind}_{index:04d}_{row['id']}.jpg"
            image.save(output/file,"JPEG",quality=93)
            with _IMAGE_OPEN(output/file) as jpeg:
                jpeg.load(); decoded=jpeg.convert("RGB")
                if decoded.size!=image.size or min(decoded.size)<=0: raise ValueError("JPEG dimensions invalid")
                pixel_hash=hashlib.sha256(f"{decoded.width}x{decoded.height}:".encode()+decoded.tobytes()).hexdigest()
                row["pixel_hash"]=pixel_hash; pixels[pixel_hash].append(row["id"])
                row["dhash"]=dhash(decoded)
                row["blur_laplacian"]=float(cv2.Laplacian(cv2.cvtColor(np.asarray(decoded),cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
                boxes=face_proposals(decoded,detector)
            for box in boxes:
                box["class_id"]=row["intended_class_id"]
                if box["class_id"]==2:
                    box["proposal_bbox_before_context"]=list(box["bbox"])
                    box["bbox"]=pad_mask_context(box["bbox"],image.width,image.height)
                    box["context_padding_applied"]=True
            row.update(processed_file=file,processed_sha256=hashlib.sha256((output/file).read_bytes()).hexdigest(),
                       width=image.width,height=image.height,boxes=boxes,face_count=len(boxes),
                       conversion_status="success",label_status="draft",review_reason=[] if len(boxes)==1 else ["no_face" if not boxes else "multiple_faces"])
            if row["intended_class_id"] is None: row["review_reason"].append("class_unknown")
            if row["blur_laplacian"]<30: row["review_reason"].append("low_sharpness")
        except Exception as exc:
            row.update(conversion_status="failed",conversion_error=str(exc),label_status="excluded",boxes=[],review_reason=["decode_or_conversion_failed"])
        entries.append(row)
        if (index+1)%20==0: print(f"Processed {index+1}/{len(sources)}",flush=True)
    near=[]
    valid=[r for r in entries if r.get("dhash") is not None]
    for i,row in enumerate(valid):
        for other in valid[:i]:
            distance=(row["dhash"]^other["dhash"]).bit_count()
            if distance<=4: near.append({"a":other["id"],"b":row["id"],"distance":distance})
    manifest={"created":timestamp(),"source":str(source),"output":str(output),"single_person":single_person,"source_kind":source_kind,
        "mapping":CLASSES,"folder_aliases":FOLDER_IDS,"max_side":max_side,"proposal_detector":str(detector_path),
        "detector_sha256":hashlib.sha256(Path(detector_path).read_bytes()).hexdigest(),
        "detector_source":"https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        "images":entries,"exact_source_groups":[v for v in hashes.values() if len(v)>1],
        "exact_processed_pixel_groups":[v for v in pixels.values() if len(v)>1],"near_duplicate_pairs":near,
        "statistics":{"source_images":len(entries),"by_folder_format":[{"folder":k[0],"format":k[1],"count":v} for k,v in sorted(counts.items())],
            "conversion_success":sum(r["conversion_status"]=="success" for r in entries),
            "conversion_failed":sum(r["conversion_status"]=="failed" for r in entries),
            "face_counts":dict(Counter(str(r.get("face_count","failed")) for r in entries)),
            "verified_images":0}}
    manifest["contact_sheets"]=contact_sheets(manifest,output)
    write_json(output/"manifest.json",manifest)
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"data/raw/custom_capture/single_person")
    parser.add_argument("--output",type=Path,default=ROOT/"data/processed/single_person_review_v1")
    parser.add_argument("--detector",type=Path,default=ROOT/".runtime/face_detector/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--multi-person",action="store_true")
    parser.add_argument("--max-side",type=int,default=1600)
    parser.add_argument("--reports",type=Path,help="Report directory; defaults to processed output")
    parser.add_argument("--source-kind",choices=["phone","multi","esp32","capture"])
    args=parser.parse_args()
    manifest=prepare(args.source,args.output,args.detector,not args.multi_person,args.max_side,args.source_kind)
    write_json((args.reports or args.output)/"custom_audit.json",manifest["statistics"])
    print(json.dumps(manifest["statistics"],indent=2))

if __name__=="__main__": raise SystemExit(main())
