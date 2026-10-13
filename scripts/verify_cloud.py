"""Validate a deployed HTTPS cloud using a user-selected image. No Telegram."""
import argparse
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server.cloud_client import CloudConfig,CloudInferenceClient
from server.inference import MaskDetectionModel

def verify(image):
    env={**dotenv_values(ROOT/'.env'),**os.environ}
    client=CloudInferenceClient(CloudConfig(env.get('CLOUD_API_URL',''),env.get('CLOUD_API_KEY',''),env.get('CLOUD_MODEL_SHA256',''),float(env.get('CLOUD_TIMEOUT_SECONDS','90'))))
    data=Path(image).read_bytes();decoded=MaskDetectionModel('').decode(data)
    try:
        info=client.verify();result,annotation=client.predict(data,decoded.shape[1],decoded.shape[0])
    finally:client.close()
    directory=ROOT/'reports/cloud_v7';directory.mkdir(parents=True,exist_ok=True)
    private=directory/'private';private.mkdir(exist_ok=True)
    (private/'deployed_probe.jpg').write_bytes(annotation)
    report={k:v for k,v in result.items() if k not in {'annotated_image_base64','detections'}}
    report.update(verified_at=datetime.now(timezone.utc).isoformat(),test_location='real_https_cloud',model_info=info)
    (directory/'deployed_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    path=ROOT/'models/cloud_v7/model_manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest.update(cloud_deployed=True,cloud_uploaded=True,cloud_verification=str(directory/'deployed_verification.json'))
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Verified deployed model and exact-frame result. Report: reports/cloud_v7/deployed_verification.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True,help='Image that you choose to send to Render')
    verify(p.parse_args().image)
