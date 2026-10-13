"""Generate a private shared key or set the Render URL in the local .env."""
import argparse
import json
import os
import secrets
from pathlib import Path
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parents[1]

def configure(url=None,init_key=False):
    from server.cloud_client import CloudConfig,CloudInferenceClient
    path=ROOT/'.env';values=dotenv_values(path) if path.exists() else {}
    sha=json.loads((ROOT/'models/cloud_v7/model_manifest.json').read_text(encoding='utf-8'))['model_sha256']
    key=values.get('CLOUD_API_KEY') or ''
    if init_key and not key:key=secrets.token_urlsafe(32)
    if key and len(key)<24:raise ValueError('CLOUD_API_KEY must contain at least 24 characters')
    if url is not None:CloudInferenceClient(CloudConfig(url,key,sha))
    updates={'CLOUD_MODEL_SHA256':sha,'CLOUD_TIMEOUT_SECONDS':values.get('CLOUD_TIMEOUT_SECONDS') or '90'}
    if url is not None:updates['CLOUD_API_URL']=url.rstrip('/')
    if key:updates['CLOUD_API_KEY']=key
    original=path.read_text(encoding='utf-8') if path.exists() else ''
    lines=original.splitlines();done=set();result=[]
    for line in lines:
        name=line.split('=',1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
        if name in updates:
            if name not in done:result.append(name+'='+updates[name]);done.add(name)
        else:result.append(line)
    result.extend(name+'='+value for name,value in updates.items() if name not in done)
    path.write_text('\n'.join(result)+'\n',encoding='utf-8')
    if key:
        private=ROOT/'.runtime/cloud/render_secrets.env';private.parent.mkdir(parents=True,exist_ok=True)
        private.write_text('CLOUD_API_KEY='+key+'\nCLOUD_MODEL_SHA256='+sha+'\n',encoding='utf-8')
        print('Private Render variables saved to .runtime/cloud/render_secrets.env; key is not printed.')
    print('Local .env updated. Restart the local backend to apply.')

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url');p.add_argument('--init-key',action='store_true')
    args=p.parse_args()
    if args.url is None and not args.init_key:p.error('Use --init-key or --url https://your-service.onrender.com')
    configure(args.url,args.init_key)
