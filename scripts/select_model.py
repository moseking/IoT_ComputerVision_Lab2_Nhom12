"""Select a saved model without overwriting checkpoints. Restart backend afterwards."""
import argparse,re,hashlib
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('variant',choices=['v3','camera-v7','camera-v8'])
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
relative={'v3':'models/face_mask_best.pt','camera-v7':'models/camera_v7/face_mask_best.pt','camera-v8':'models/camera_v8/face_mask_best.pt'}[args.variant]
checkpoint=root/relative
if not checkpoint.is_file():parser.error('Selected checkpoint missing')
env=root/'.env';text=env.read_text(encoding='utf-8')
updated,count=re.subn(r'^MODEL_PATH=.*$', 'MODEL_PATH='+relative,text,flags=re.M)
if count!=1:parser.error('Expected exactly one MODEL_PATH in local .env')
env.write_text(updated,encoding='utf-8')
print('Selected '+args.variant+'; checkpoint '+hashlib.sha256(checkpoint.read_bytes()).hexdigest())
print('Restart your backend terminal to load this checkpoint; do not reflash ESP32.')
