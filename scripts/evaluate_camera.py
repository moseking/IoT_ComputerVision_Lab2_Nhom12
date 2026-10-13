"""Audit and evaluate reviewed ESP32 images without adding them to training.

Two annotation views are retained: the user's original face boxes (primary)
and optional user-authorized mask-context geometry (supplementary). AP is
measured at conf=.001; operational errors use conf=.35 and class-aware IoU=.5.
"""
import argparse
import copy
import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

try:
    from .common import CLASSES, device_name, metrics_dict, timestamp, write_json
    from .evaluate import match_errors
except ImportError:
    from common import CLASSES, device_name, metrics_dict, timestamp, write_json
    from evaluate import match_errors


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_rows(review):
    path = review / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    registry = review / 'training_use.json'
    if registry.is_file():
        usage = json.loads(registry.read_text(encoding='utf-8'))
        if usage.get('manifest_sha256') == sha(path) and usage.get('role') == 'training_development':
            raise ValueError('Camera snapshot is now training/development data; use fresh reviewed captures for independent evaluation')
    if manifest.get('source_kind') != 'esp32':
        raise ValueError('Only a reviewed ESP32 manifest is accepted')
    if {int(k): v for k, v in manifest['mapping'].items()} != CLASSES:
        raise ValueError('Class mapping mismatch')
    rows = manifest['images']
    if not rows:
        raise ValueError('Empty camera review')
    from PIL import Image
    for row in rows:
        if not (row.get('label_status') == 'verified' and row.get('human_reviewed')
                and row.get('reviewer') and row.get('reviewed_at')):
            raise ValueError(f"Image {row['index']} lacks completed human verification")
        if row.get('included_in_training'):
            raise ValueError('Camera evaluation must not include training images')
        image_path = (review / row['processed_file']).resolve()
        if not image_path.is_relative_to(review.resolve()):
            raise ValueError('Image path outside review directory')
        if sha(image_path) != row['processed_sha256'] or sha(Path(row['source_file'])) != row['source_sha256']:
            raise ValueError('Camera source/processed image changed after review')
        with Image.open(image_path) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError('Reviewed pixel dimensions mismatch')
        for box in row['boxes']:
            x1, y1, x2, y2 = box['bbox']
            if box['class_id'] not in CLASSES or not (0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height']):
                raise ValueError('Invalid reviewed box/class')
    return manifest, sha(path)


def context_rows(rows, ledger, manifest_hash):
    if ledger['manifest_sha256'] != manifest_hash:
        raise ValueError('Context ledger refers to a different review snapshot')
    result = copy.deepcopy(rows)
    lookup = {row['index']: row for row in result}
    seen = set()
    for change in ledger['changes']:
        key = (change['index'], change['box_index'])
        if key in seen:
            raise ValueError('Duplicate context adjustment')
        seen.add(key)
        row = lookup[change['index']]
        box = row['boxes'][change['box_index']]
        if box['class_id'] != 2:
            raise ValueError('Context expansion may only modify incorrect-mask geometry')
        old = box['bbox']; new = change['bbox']
        if not (0 <= new[0] <= old[0] < old[2] <= new[2] <= row['width']
                and 0 <= new[1] <= old[1] < old[3] <= new[3] <= row['height']):
            raise ValueError('Context box must contain the reviewed face and stay inside image')
        box['bbox_original_human'] = old
        box['bbox'] = new
        box['geometry_adjustment'] = {'actor': ledger['actor'], 'reason': change['reason'],
                                      'human_verified': False, 'user_authorized': True}
        row['human_reviewed_applies_to'] = 'Original face geometry and class only; adjusted geometry is not human re-verified'
    return result


def export_view(rows, review, destination):
    import yaml
    images = destination / 'images/evaluation'; labels = destination / 'labels/evaluation'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    (destination / 'images/unused_train').mkdir(parents=True)
    for row in rows:
        path = review / row['processed_file']
        shutil.copy2(path, images / path.name)
        lines = []
        for box in row['boxes']:
            x1,y1,x2,y2 = box['bbox']; w,h = row['width'],row['height']
            lines.append(f"{box['class_id']} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}")
        (labels / (path.stem + '.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Ultralytics requires train/val keys even for split=test. Train is empty;
    # val aliases this evaluation set only for schema compatibility, not selection.
    (destination / 'data.yaml').write_text(yaml.safe_dump({'path':str(destination.resolve()),
        'train':'images/unused_train','val':'images/evaluation','test':'images/evaluation','names':CLASSES},sort_keys=False),encoding='utf-8')
    write_json(destination/'manifest.json',{'purpose':'camera evaluation only; never train or checkpoint selection',
        'schema_val_alias_is_not_validation_partition':True,'images':rows})
    return destination/'data.yaml'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--context-ledger',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--imgsz',type=int,default=512)
    parser.add_argument('--conf',type=float,default=.35)
    parser.add_argument('--device',default='0')
    parser.add_argument('--agnostic-nms',action='store_true',help='Use deployed exclusive-class face NMS; camera then documents postprocessing debugging, not an untouched test')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory; evaluations are immutable')
    if not 0<=args.conf<=1:parser.error('Confidence must be 0..1')
    manifest,manifest_hash=reviewed_rows(args.review)
    rows=manifest['images']
    ledger=json.loads(args.context_ledger.read_text(encoding='utf-8')) if args.context_ledger else None
    context=context_rows(rows,ledger,manifest_hash) if ledger else None
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    model=YOLO(str(args.model.resolve()))
    if {int(k):v for k,v in model.names.items()}!=CLASSES:parser.error('Checkpoint mapping mismatch')
    device=device_name(args.device)
    args.output.mkdir(parents=True)
    write_json(args.output/'locked_inputs.json',{'locked_before_inference_at':timestamp(),
        'manifest_sha256':manifest_hash,'model_sha256':sha(args.model),
        'context_ledger_sha256':sha(args.context_ledger) if ledger else None,
            'geometry_changed_by_assistant_not_human':bool(ledger),'camera_images_used_in_training':0,
            'postprocessing_debugged_using_these_camera_images':args.agnostic_nms,
        'checkpoint_selection':'Previously frozen public validation decision; camera results not used to select or tune'})
    views={'human_face':rows}
    if context:views['mask_context']=context
    metrics={}
    for name,view in views.items():
        folder=args.output/name
        data=export_view(view,args.review,folder/'dataset')
        measured=model.val(data=str(data.resolve()),split='test',imgsz=args.imgsz,batch=8,device=device,
            workers=0,conf=.001,iou=.45,agnostic_nms=args.agnostic_nms,plots=True,project=str(folder),name='ap',exist_ok=False)
        metrics[name]={**metrics_dict(measured),'framework_speed_ms':measured.speed}
    items=[];latencies=[];counts={name:{key:Counter() for key in ['ground_truth','predicted','false_positive','false_negative']} for name in views}
    example_dir=args.output/'examples'; example_dir.mkdir()
    for index,row in enumerate(rows):
        with Image.open(args.review/row['processed_file']) as image: image=image.convert('RGB')
        if index==0:model.predict(image,conf=args.conf,iou=.45,agnostic_nms=args.agnostic_nms,imgsz=args.imgsz,device=device,verbose=False)
        if device!='cpu':torch.cuda.synchronize()
        start=time.perf_counter()
        prediction=model.predict(image,conf=args.conf,iou=.45,agnostic_nms=args.agnostic_nms,imgsz=args.imgsz,device=device,verbose=False)[0]
        if device!='cpu':torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000; latencies.append(elapsed)
        predicted=[{'class_id':int(b.cls.item()),'confidence':float(b.conf.item()),'bbox':b.xyxy[0].cpu().tolist()} for b in prediction.boxes]
        errors={}
        for name,view in views.items():
            gt=view[index]['boxes'];error=match_errors(gt,predicted)
            errors[name]=error
            for key,boxes in [('ground_truth',gt),('predicted',predicted),('false_positive',error['false_positives']),('false_negative',error['false_negatives'])]:counts[name][key].update(b['class_id'] for b in boxes)
        canvas=image.copy();draw=ImageDraw.Draw(canvas)
        for box in rows[index]['boxes']:
            draw.rectangle(box['bbox'],outline='#18ef80',width=2)
            draw.text((box['bbox'][0],max(0,box['bbox'][1]-12)),f"GT {box['class_id']}",fill='#18ef80',stroke_width=1,stroke_fill='black')
        if context:
            for box in context[index]['boxes']:
                if box.get('geometry_adjustment'):draw.rectangle(box['bbox'],outline='#ffd53b',width=2)
        for box in predicted:
            draw.rectangle(box['bbox'],outline='#ff4141',width=2)
            draw.text((box['bbox'][0],min(image.height-12,box['bbox'][3])),f"P {box['class_id']} {box['confidence']:.2f}",fill='#ff4141',stroke_width=1,stroke_fill='black')
        example=example_dir/f"{row['index']:02d}.jpg"; canvas.save(example,quality=93)
        items.append({'index':row['index'],'source':row['source_relative'],'ground_truth':row['boxes'],
            'context_ground_truth':context[index]['boxes'] if context else None,'predictions':predicted,
            'errors':errors,'inference_wall_ms':elapsed,'example':str(example)})
    for name in views:
        metrics[name]['counts_at_operational_confidence']={label:{key:counts[name][key][cls] for key in counts[name]} for cls,label in CLASSES.items()}
    report={'timestamp':timestamp(),'model':str(args.model.resolve()),'model_sha256':sha(args.model),
        'review_manifest_sha256':manifest_hash,'images':len(rows),'real_people_boxes':sum(len(r['boxes']) for r in rows),
        'device':device,'image_size':args.imgsz,'operational_confidence':args.conf,'nms_iou':.45,'error_matching_iou':.5,
        'metric_min_confidence':.001,'nms_class_agnostic':args.agnostic_nms,'metrics':metrics,'class_mapping':CLASSES,
        'camera_images_used_in_training':0,'camera_results_used_for_checkpoint_selection':False,
        'postprocessing_debugged_using_these_camera_images':args.agnostic_nms,
        'limitations':'Small convenience sample from the same room/people; correlated frames, no identity-disjoint claim. Context geometry is assistant-adjusted under user authorization, not re-verified by human.',
        'inference_wall_ms':{'mean':float(np.mean(latencies)),'median':float(np.median(latencies)),'p95':float(np.percentile(latencies,95)),
            'samples':len(rows),'warmup_excluded':True,'definition':'preprocess+forward+NMS with CUDA synchronization; image decode, WiFi and capture excluded'}}
    write_json(args.output/'predictions.json',{'images':items})
    write_json(args.output/'metrics.json',report)
    with (args.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['geometry','class_id','class_name','precision','recall','mAP50','mAP50-95','ground_truth','predicted','false_positive','false_negative'])
        writer.writeheader()
        for name,result in metrics.items():
            for label,m in result['per_class'].items():writer.writerow({'geometry':name,'class_name':label,**{k:m[k] for k in ['class_id','precision','recall','mAP50','mAP50-95']},**result['counts_at_operational_confidence'][label]})
    if sha(args.review/'manifest.json')!=manifest_hash:raise RuntimeError('Human review changed during evaluation; discard this evaluation')
    print(json.dumps({'images':len(rows),'model':args.model.name,'metrics':metrics,'latency':report['inference_wall_ms']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
