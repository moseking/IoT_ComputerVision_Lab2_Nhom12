"""Shared read-only VOC audit, safe pairing and bbox conversion (no inferred labels)."""
import hashlib
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from PIL import Image, ImageOps
try:
    from .common import CLASSES, CLASS_IDS, timestamp
except ImportError:
    from common import CLASSES, CLASS_IDS, timestamp

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_XML_BYTES = 4 * 1024 * 1024
_IMAGE_OPEN = Image.open  # Preserve the original decoder before optional YOLO imports.


def dhash(image):
    gray = ImageOps.grayscale(image).resize((9, 8))
    pixels = list(gray.getdata())
    return sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


def yolo_lines(record):
    """Use validated image dimensions and keep exactly one line per original object."""
    lines = []
    for obj in record["boxes"]:
        x1, y1, x2, y2 = obj["bbox"]
        width, height = record["width"], record["height"]
        if obj["class_id"] not in CLASSES or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Unvalidated box cannot become a YOLO label")
        values = ((x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Invalid normalized bbox")
        lines.append(f"{obj['class_id']} " + " ".join(f"{v:.12g}" for v in values))
    return "\n".join(lines) + ("\n" if lines else "")


def audit(raw, repair_boxes=False, coordinates="zero-based"):
    if coordinates not in {"zero-based", "one-based-inclusive"}:
        raise ValueError("Unknown VOC coordinate convention")
    raw = Path(raw).resolve()
    image_root, xml_root = raw/"images", raw/"annotations"
    if not image_root.is_dir() or not xml_root.is_dir():
        raise FileNotFoundError(f"Dataset needs images/ and annotations/: {raw}")
    images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    annotations = sorted(p for p in xml_root.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")
    issues, records, annotations_inventory = [], [], []
    by_name, by_relative, by_stem = defaultdict(list), defaultdict(list), defaultdict(list)
    image_info, pixel_groups, file_groups, annotation_groups = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    signatures, referenced = defaultdict(list), set()
    raw_counts, unknown_counts = Counter(), Counter()
    raw_objects, parsed_xml = 0, 0

    def issue(kind, path, message, severity="error", image=None):
        item = {"kind":kind,"file":str(path),"message":message,"severity":severity}
        if image:
            item["image"] = str(image)
        issues.append(item)

    for path in images:
        by_name[path.name.casefold()].append(path)
        by_relative[path.relative_to(image_root).as_posix().casefold()].append(path)
        by_stem[path.stem.casefold()].append(path)
        try:
            if not path.resolve().is_relative_to(image_root.resolve()):
                raise ValueError("Image symlink resolves outside images directory")
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with _IMAGE_OPEN(path) as image:
                image.load()
                rgb = image.convert("RGB")
                width, height = rgb.size
                pixel_hash = hashlib.sha256(f"{width}x{height}:".encode()+rgb.tobytes()).hexdigest()
                info = {"width":width,"height":height,"pixel_hash":pixel_hash,"dhash":dhash(rgb),"sha256":file_hash}
            image_info[path] = info
            pixel_groups[pixel_hash].append(str(path))
            file_groups[file_hash].append(str(path))
        except Exception as exc:
            issue("invalid_image",path,str(exc))
    for paths in by_stem.values():
        if len(paths) > 1:
            for path in paths:
                issue("duplicate_basename",path,f"YOLO basename collision: {[str(p) for p in paths]}",image=path)

    for xml_path in annotations:
        item = {"annotation":str(xml_path),"image":None,"objects":None,"valid":False}
        annotations_inventory.append(item)
        errors_before = sum(i["severity"]=="error" for i in issues)
        try:
            if not xml_path.resolve().is_relative_to(xml_root.resolve()):
                raise ValueError("XML symlink resolves outside annotations directory")
            if xml_path.stat().st_size > MAX_XML_BYTES:
                raise ValueError("XML exceeds 4 MiB safety limit")
            content = xml_path.read_bytes()
            text = content.decode("utf-8-sig")
            if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
                raise ValueError("XML DTD/entity declarations are not accepted")
            root = ET.fromstring(text)
            if root.tag != "annotation":
                raise ValueError("Root element must be annotation")
        except Exception as exc:
            issue("invalid_xml",xml_path,str(exc))
            continue
        parsed_xml += 1
        objects = root.findall("object")
        item["objects"] = len(objects)
        raw_objects += len(objects)
        for obj in objects:
            name = (obj.findtext("name") or "").strip()
            if name in CLASS_IDS:
                raw_counts[name] += 1
            else:
                unknown_counts[name or "<missing>"] += 1
        filename = (root.findtext("filename") or "").strip().replace("\\","/")
        relative = PurePosixPath(filename)
        if not filename or relative.is_absolute() or any(p in {"..","."} or ":" in p for p in relative.parts):
            issue("invalid_filename",xml_path,f"Unsafe/missing filename: {filename!r}")
            continue
        # XML filename is authoritative. Never read arbitrary paths from its contents.
        matches = by_relative.get(relative.as_posix().casefold(),[]) if len(relative.parts)>1 else by_name.get(filename.casefold(),[])
        if len(matches) != 1:
            issue("missing_image" if not matches else "ambiguous_image",xml_path,f"filename={filename!r}, matched={len(matches)}")
            continue
        image_path = matches[0]
        item["image"] = str(image_path)
        referenced.add(image_path)
        if xml_path.stem.casefold() != image_path.stem.casefold():
            issue("xml_image_name_mismatch",xml_path,"Paired by XML filename; label uses the real image basename","warning",image_path)
        info = image_info.get(image_path)
        if info is None:
            issue("annotation_image_unreadable",xml_path,"Referenced image could not be decoded",image=image_path)
            continue
        width,height = info["width"],info["height"]
        try:
            xml_width,xml_height = int(root.findtext("size/width","0")),int(root.findtext("size/height","0"))
            if xml_width <= 0 or xml_height <= 0:
                raise ValueError(f"XML size <=0: {xml_width}x{xml_height}")
            if (xml_width,xml_height)!=(width,height):
                issue("dimension_mismatch",xml_path,f"XML {xml_width}x{xml_height}, real {width}x{height}",
                      "warning" if repair_boxes else "error",image_path)
        except ValueError as exc:
            issue("invalid_dimensions",xml_path,str(exc),image=image_path)
        boxes = []
        for index,obj in enumerate(objects):
            class_name = (obj.findtext("name") or "").strip()
            if class_name not in CLASS_IDS:
                issue("unknown_class",xml_path,f"object {index}: {class_name!r}",image=image_path)
                continue
            try:
                bbox = obj.find("bndbox")
                if bbox is None:
                    raise ValueError("Missing bndbox")
                x1,y1,x2,y2 = [float(bbox.findtext(k,"nan")) for k in ("xmin","ymin","xmax","ymax")]
                if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):
                    raise ValueError("Non-finite coordinate")
                if coordinates=="one-based-inclusive":
                    x1-=1
                    y1-=1
                if x2<=x1 or y2<=y1:
                    raise ValueError("Box has non-positive area")
                if x1<0 or y1<0 or x2>width or y2>height:
                    if not repair_boxes:
                        raise ValueError(f"Box {(x1,y1,x2,y2)} outside {width}x{height}")
                    original=(x1,y1,x2,y2)
                    x1,y1,x2,y2=max(0,x1),max(0,y1),min(width,x2),min(height,y2)
                    if x2<=x1 or y2<=y1:
                        raise ValueError("Explicit clipping would produce empty box")
                    issue("clipped_bbox",xml_path,f"object {index}: {original} -> {(x1,y1,x2,y2)}","warning",image_path)
                boxes.append({"class_id":CLASS_IDS[class_name],"bbox":[x1,y1,x2,y2]})
            except ValueError as exc:
                issue("invalid_bbox",xml_path,f"object {index}: {exc}",image=image_path)
        canonical=sorted((b["class_id"],*b["bbox"]) for b in boxes)
        signature=hashlib.sha256(repr((width,height,canonical)).encode()).hexdigest()
        annotation_groups[signature].append(str(xml_path))
        signatures[image_path].append(signature)
        valid = sum(i["severity"]=="error" for i in issues)==errors_before
        item["valid"]=valid
        if valid:
            records.append({"image":str(image_path),"annotation":str(xml_path),**info,"boxes":boxes,
                            "annotation_hash":signature,"annotation_sha256":hashlib.sha256(content).hexdigest()})

    # Multiple XMLs referring to one image are ambiguous: exclude all, not just the later file.
    xml_by_image = defaultdict(list)
    for item in annotations_inventory:
        if item["image"]:
            xml_by_image[item["image"]].append(item)
    ambiguous = {image for image,items in xml_by_image.items() if len(items)>1}
    for image in sorted(ambiguous):
        for item in xml_by_image[image]:
            item["valid"]=False
            issue("duplicate_annotation",item["annotation"],"Multiple XML files reference this image",image=image)
    colliding = {str(path) for paths in by_stem.values() if len(paths)>1 for path in paths}
    records=[r for r in records if r["image"] not in ambiguous|colliding]
    for item in annotations_inventory:
        if item["image"] in colliding:
            item["valid"]=False
    for path in images:
        if path not in referenced:
            issue("missing_annotation",path,"No parsed safe XML filename references this image",image=path)

    conflicting = []
    for group in pixel_groups.values():
        digests={digest for path in group for digest in signatures.get(Path(path),[])}
        if len(group)>1 and len(digests)>1:
            conflicting.append(group)
            issue("conflicting_duplicate_labels",group[0],"Identical pixel images have different parsed boxes/classes")
    counts=Counter(b["class_id"] for r in records for b in r["boxes"])
    face_counts=[len(r["boxes"]) for r in records]
    errors=sum(i["severity"]=="error" for i in issues)
    invalid_xml_files=sum(not item["valid"] for item in annotations_inventory)
    issue_counts=Counter(i["kind"] for i in issues)
    report={"timestamp":timestamp(),"raw_path":str(raw),"image_directory":str(image_root),
            "annotation_directory":str(xml_root),"coordinates":coordinates,"repair_boxes":repair_boxes,
            "supported_image_extensions":sorted(EXTENSIONS),"actual_image_formats":dict(Counter(p.suffix.lower() for p in images)),
            "total_images":len(images),"total_annotations":len(annotations),"parsed_xml_files":parsed_xml,
            "raw_objects":raw_objects,"raw_class_distribution":{name:raw_counts[name] for name in CLASSES.values()},
            "unknown_class_distribution":dict(unknown_counts),"all_three_classes_present":all(raw_counts[name]>0 for name in CLASSES.values()),
            "raw_statistics_scope":"Every object in syntactically parsed XML, including objects later rejected; no inferred labels",
            "valid_images":len(records),"invalid_xml_files":invalid_xml_files,"bounding_boxes":sum(face_counts),
            "class_distribution":{name:counts[i] for i,name in CLASSES.items()},
            "images_with_0_objects":face_counts.count(0),"images_with_1_object":face_counts.count(1),
            "images_with_multiple_objects":sum(n>1 for n in face_counts),
            "faces_per_image":{"min":min(face_counts,default=0),"max":max(face_counts,default=0),
                              "average":statistics.mean(face_counts) if face_counts else 0},
            "class_imbalance_ratio":max(counts.values())/min(counts.values()) if len(counts)==3 and min(counts.values()) else None,
            "duplicate_images":[g for g in pixel_groups.values() if len(g)>1],
            "duplicate_file_sha256":[g for g in file_groups.values() if len(g)>1],
            "duplicate_annotations":[g for g in annotation_groups.values() if len(g)>1],
            "conflicting_duplicate_image_groups":conflicting,"annotation_inventory":annotations_inventory,
            "error_count":errors,"warning_count":len(issues)-errors,"issue_counts":dict(issue_counts),"issues":issues,
            "statistics_scope":"Complete valid image/XML pairs; conflicting pixel duplicates flagged separately, not silently repaired"}
    return records,report


def report_text(report):
    excluded={"issues","duplicate_images","duplicate_annotations","duplicate_file_sha256",
              "conflicting_duplicate_image_groups","annotation_inventory"}
    lines=[f"{key}: {value}" for key,value in report.items() if key not in excluded]
    for key in ("duplicate_images","duplicate_file_sha256","duplicate_annotations"):
        lines.append("\n"+key+":")
        lines.extend("  "+" | ".join(group) for group in report.get(key,[]))
    lines.append("\nIssues:")
    lines.extend(f"[{i['severity']}] {i['kind']} {i['file']}: {i['message']}" for i in report["issues"])
    return "\n".join(lines)+"\n"


def report_markdown(report):
    lines=["# Audit dataset Pascal VOC", "", f"Nguồn: `{report['raw_path']}`",
           f"Thời điểm UTC: {report.get('timestamp','')}", "",
           f"- Ảnh: **{report.get('total_images',0)}**; XML: **{report.get('total_annotations',0)}**.",
           f"- Đối tượng trong XML đọc được: **{report.get('raw_objects',0)}** (bao gồm box bị loại sau kiểm tra).",
           f"- Cặp ảnh/XML hợp lệ trước xử lý duplicates: **{report.get('valid_images',0)}**.",
           f"- Lỗi: **{report['error_count']}**; cảnh báo: **{report.get('warning_count',0)}**.",
           f"- Quy ước tọa độ: `{report.get('coordinates','')}`; repair_boxes={report.get('repair_boxes',False)}.",
           "", "| ID | Class | Objects XML gốc | Boxes trong cặp hợp lệ |",
           "|---|---|---:|---:|"]
    for cls,name in CLASSES.items():
        lines.append(f"| {cls} | {name} | {report.get('raw_class_distribution',{}).get(name,0)} | {report.get('class_distribution',{}).get(name,0)} |")
    lines += ["", "Class ngoài danh sách: "+repr(report.get("unknown_class_distribution",{})), "",
              "Thống kê XML gốc không phải thống kê train. Ảnh có bất kỳ object lỗi bị loại toàn bộ;",
              "không chuyển phần còn lại thành nhãn rỗng/ảnh nền. Chi tiết từng cặp nằm trong JSON.", "",
              "## Lỗi/cảnh báo cụ thể", "", "| Mức | Loại | File | Chi tiết |", "|---|---|---|---|"]
    for item in report["issues"]:
        name=Path(item["file"]).name.replace("|","\\|")
        message=item["message"].replace("|","\\|").replace("\n"," ")
        lines.append(f"| {item['severity']} | {item['kind']} | {name} | {message} |")
    lines += ["", "## Ảnh trùng và giới hạn", "",
              f"Nhóm SHA-256 bytes trùng: {len(report.get('duplicate_file_sha256',[]))}.",
              f"Nhóm SHA-256 pixels trùng: {len(report.get('duplicate_images',[]))}.",
              f"Nhóm pixel trùng nhưng nhãn mâu thuẫn: {len(report.get('conflicting_duplicate_image_groups',[]))}.",
              "Không xóa ảnh/XML gốc. Group split theo pixel SHA/dHash, thêm --groups khi có metadata người/phiên.",
              "Không bảo đảm chống mọi dạng leakage, crop hoặc trùng danh tính. Test không dùng chọn model/siêu tham số.",
              "", "## Cách xử lý cho bản public strict", "",
              "Chạy converter với --invalid-policy skip --duplicate-policy exclude:",
              "loại toàn bộ ảnh annotation lỗi và cả nhóm pixel trùng nhãn mâu thuẫn; không clip.",
              "Lớp hiếm/không có ảnh nền có thể hạn chế train; xem conversion report cho số liệu cuối.", ""]
    return "\n".join(lines)
