"""Send verified raw ESP32 JPEGs to the local API; keep camera timing unknown."""
import argparse,base64,hashlib,json,time
from collections import Counter
from pathlib import Path
import httpx
from dotenv import dotenv_values
try:
    from .common import ROOT,CLASSES,timestamp,write_json
    from .evaluate import match_errors,overlap
    from .evaluate_camera import reviewed_rows
except ImportError:
    from common import ROOT,CLASSES,timestamp,write_json
    from evaluate import match_errors,overlap
    from evaluate_camera import reviewed_rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,default=ROOT/'data/processed/esp32_review_v1')
    parser.add_argument('--api-url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--last-index',type=int)
    parser.add_argument('--indices',type=int,nargs='+',help='Optional reviewed image indexes for a smaller HTTP smoke')
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a fresh output directory')
    manifest,digest=reviewed_rows(args.review)
    if any(r['original_size']!=r['processed_size'] or r['exif_orientation']!=1 for r in manifest['images']):
        parser.error('Raw HTTP comparison requires identical raw/review pixel orientation and dimensions')
    env=dotenv_values(ROOT/'.env');headers={'X-API-Key':env.get('API_KEY') or ''}
    args.output.mkdir(parents=True)
    rows=manifest['images']
    if args.indices is not None:
        if len(set(args.indices))!=len(args.indices) or not set(args.indices)<=set(r['index'] for r in rows):parser.error('Indexes must be unique and present in the review')
        rows=[r for r in rows if r['index'] in args.indices]
    if args.last_index is not None:rows=[r for r in rows if r['index']!=args.last_index]+[r for r in rows if r['index']==args.last_index]
    cases=[]; counters={k:Counter() for k in ['ground_truth','predicted','false_positive','false_negative']}
    with httpx.Client(base_url=args.api_url,headers=headers,timeout=60,trust_env=False) as client:
        client.get('/health').raise_for_status()
        info=client.get('/model/info');info.raise_for_status();info=info.json()
        if not info['loaded'] or not info['nms_class_agnostic']:raise RuntimeError('Restart backend to apply exclusive-face NMS before testing')
        telegram=client.get('/telegram/status');telegram.raise_for_status()
        if telegram.json().get('enabled'):raise RuntimeError('Disable Telegram for this private local smoke test')
        for row in rows:
            path=Path(row['source_file']);image_bytes=path.read_bytes()
            assert hashlib.sha256(image_bytes).hexdigest()==row['source_sha256']
            start=time.perf_counter()
            response=client.post('/predict',files={'file':(path.name,image_bytes,'image/jpeg')},data={'source':'test_client','include_image':'true'})
            roundtrip=(time.perf_counter()-start)*1000;response.raise_for_status();payload=response.json()
            token=payload.pop('report_token')
            annotated=payload.pop('annotated_image_base64')
            (args.output/f"{row['index']:02d}.jpg").write_bytes(base64.b64decode(annotated,validate=True))
            predicted=[{**d,'bbox':[d['bbox'][k] for k in ['x1','y1','x2','y2']]} for d in payload['detections']]
            assert len(predicted)==payload['detection_count']
            assert payload['width']==row['width'] and payload['height']==row['height']
            assert not any(overlap(a['bbox'],b['bbox'])>.451 for i,a in enumerate(predicted) for b in predicted[i+1:]),'Overlapping duplicate faces after NMS'
            errors=match_errors(row['boxes'],predicted)
            for key,boxes in [('ground_truth',row['boxes']),('predicted',predicted),('false_positive',errors['false_positives']),('false_negative',errors['false_negatives'])]:counters[key].update(b['class_id'] for b in boxes)
            metrics={'request_id':payload['request_id'],'report_token':token,'capture_ms':None,'roundtrip_ms':roundtrip,'total_ms':roundtrip,'upload_ms':None,'download_ms':None}
            client.post('/metrics/report',json=metrics).raise_for_status()
            cases.append({'index':row['index'],'source_file':row['source_relative'],'http_status':200,'roundtrip_local_ms':roundtrip,
                'ground_truth':row['boxes'],'predictions':predicted,'error_details':errors,'response':payload})
        latest=client.get('/results/latest');latest.raise_for_status();assert latest.json()['result']['request_id']==cases[-1]['response']['request_id']
    if hashlib.sha256((args.review/'manifest.json').read_bytes()).hexdigest()!=digest:raise RuntimeError('Review changed during test')
    report={'timestamp':timestamp(),'mode':'actual HTTP upload of raw camera JPEGs by laptop; source=test_client',
        'not_a_live_esp32_transport_test':True,'capture_wifi_esp32_total_ms':None,'model_sha256':info['metadata']['weights_sha256'],
        'nms_class_agnostic':True,'review_manifest_sha256':digest,'cases':cases,
        'counts':{label:{key:counters[key][cls] for key in counters} for cls,label in CLASSES.items()}}
    write_json(args.output/'report.json',report)
    print(json.dumps({'requests':len(cases),'counts':report['counts'],'last_image_index':cases[-1]['index']},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
