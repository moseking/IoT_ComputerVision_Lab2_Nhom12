"""Store exactly the exported model in a private Supabase bucket; no image uploads."""
from pathlib import Path
import argparse,hashlib,json,logging,os,re,sys
from urllib.parse import quote,urlsplit
import httpx
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

class StorageError(RuntimeError):pass

def storage_settings(env):
    base=(env.get('SUPABASE_URL') or '').rstrip('/')
    parts=urlsplit(base)
    if parts.scheme!='https' or not parts.hostname or not parts.hostname.endswith('.supabase.co') or parts.path or parts.query or parts.fragment or parts.username:
        raise StorageError('Đặt SUPABASE_URL đúng project HTTPS trong .env')
    secret=env.get('SUPABASE_SECRET_KEY') or ''
    if not (secret.startswith('sb_secret_') or secret.startswith('eyJ')):raise StorageError('Đặt SUPABASE_SECRET_KEY phía server trong .env')
    bucket=env.get('SUPABASE_MODEL_BUCKET') or 'facemask-models'
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,62}',bucket):raise StorageError('Invalid private model bucket name')
    headers={'apikey':secret}
    if secret.startswith('eyJ'):headers['Authorization']='Bearer '+secret
    return base,bucket,headers

def publish(env,manifest,client=None,expires=604800):
    base,bucket,headers=storage_settings(env)
    if manifest.get('parity_verified') is not True:raise StorageError('Verify ONNX parity before uploading')
    model=Path(manifest['model_file']);data=model.read_bytes();digest=hashlib.sha256(data).hexdigest()
    if digest!=manifest['model_sha256']:raise StorageError('Local model SHA mismatch')
    object_name='v7/face_mask_v7_'+digest[:16]+'.onnx'
    endpoint=base+'/storage/v1'
    owned=client is None
    client=client or httpx.Client(timeout=90,trust_env=False,follow_redirects=False)
    try:
        response=client.get(endpoint+'/bucket/'+bucket,headers=headers)
        if response.status_code==404:
            response=client.post(endpoint+'/bucket',headers=headers,json={'id':bucket,'name':bucket,'public':False,
                                 'file_size_limit':32*1024*1024,'allowed_mime_types':['application/octet-stream']})
            if response.status_code not in {200,201}:raise StorageError(f'Create bucket HTTP {response.status_code}')
        elif response.status_code==200:
            if response.json().get('public') is not False:raise StorageError('Use a private model bucket; existing bucket is public')
        else:raise StorageError(f'Bucket info HTTP {response.status_code}')
        path=bucket+'/'+quote(object_name,safe='/')
        upload=client.post(endpoint+'/object/'+path,headers={**headers,'Content-Type':'application/octet-stream','x-upsert':'false'},content=data)
        if upload.status_code not in {200,201,400,409}:raise StorageError(f'Upload model HTTP {upload.status_code}')
        verified=client.get(endpoint+'/object/authenticated/'+path,headers=headers)
        if verified.status_code!=200 or hashlib.sha256(verified.content).hexdigest()!=digest:
            raise StorageError('Uploaded model verification failed; no success is reported')
        sign=client.post(endpoint+'/object/sign/'+path,headers=headers,json={'expiresIn':expires})
        if sign.status_code!=200:raise StorageError(f'Sign model URL HTTP {sign.status_code}')
        signed=sign.json().get('signedURL') or sign.json().get('signedUrl')
        if not isinstance(signed,str):raise StorageError('Missing signed download URL')
        if signed.startswith('/object/'):signed=endpoint+signed
        elif signed.startswith('/storage/'):signed=base+signed
        if urlsplit(signed).hostname!=urlsplit(base).hostname or urlsplit(signed).scheme!='https':
            raise StorageError('Unexpected signed model URL')
        return {'bucket':bucket,'object':object_name,'sha256':digest,'bytes':len(data),'verified':True,'expires_seconds':expires},signed
    except httpx.HTTPError as exc:raise StorageError('Storage network error: '+type(exc).__name__) from None
    finally:
        if owned:client.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['status','publish'])
    parser.add_argument('--expires',type=int,default=604800)
    args=parser.parse_args()
    env={**dotenv_values(ROOT/'.env'),**os.environ}
    manifest=json.loads((ROOT/'models/cloud_v7/model_manifest.json').read_text(encoding='utf-8'))
    if args.command=='status':
        print(json.dumps({'model_file':manifest['model_file'],'parity_verified':manifest['parity_verified'],
              'supabase_url_configured':bool(env.get('SUPABASE_URL')),'supabase_key_configured':bool(env.get('SUPABASE_SECRET_KEY')),
              'uploaded':manifest.get('cloud_uploaded',False),'deployed':manifest.get('cloud_deployed',False)}));return
    if not 60<=args.expires<=604800:raise StorageError('Signed link lifetime must be 60 seconds to 7 days')
    result,signed=publish(env,manifest,expires=args.expires)
    private=ROOT/'.runtime/cloud';private.mkdir(parents=True,exist_ok=True)
    # Scoped download link only; never copy the Supabase master secret to Render.
    (private/'render_model.env').write_text('CLOUD_MODEL_DOWNLOAD_URL='+signed+'\nCLOUD_MODEL_SHA256='+result['sha256']+'\n',encoding='utf-8')
    report=ROOT/'reports/cloud_v7';report.mkdir(parents=True,exist_ok=True)
    (report/'supabase_upload_verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    manifest['cloud_uploaded']=True;manifest['supabase_object']=result['object']
    (ROOT/'models/cloud_v7/model_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({**result,'signed_url_saved_locally':str(private/'render_model.env')}))

if __name__=='__main__':
    try:main()
    except (StorageError,OSError,ValueError) as exc:
        print('Cloud setup: '+str(exc),file=sys.stderr);raise SystemExit(1)
