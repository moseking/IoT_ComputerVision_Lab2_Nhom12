"""Build a complete Lab 2 source snapshot for a user-controlled Git push."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE_EXTENSIONS={'.py','.ps1','.js','.cjs','.html','.css','.ino','.h','.md','.json','.yaml','.yml','.txt'}
ROOT_FILES=('README.md','requirements.txt','requirements-cuda-tested.txt','.env.example','.gitignore',
            '.dockerignore','Dockerfile','Dockerfile.local','Dockerfile.local.dockerignore','render.yaml')
PINNED_MODELS={
    'models/camera_v7/face_mask_best.pt':'722614b198153c75c2248d1e0e9f7af9a0420123d62d3438f0911d9aa95bbaef',
    'models/cloud_v7/face_mask_v7.onnx':'f1fc04ca98bace9f1a1a5537d83fc9c52574514c0641959084c94a9f0750cbf4'}

def private_values():
    values=[]
    env=ROOT/'.env'
    if env.exists():
        for line in env.read_text(encoding='utf-8').splitlines():
            if line.lstrip().startswith('#') or '=' not in line:continue
            name,value=line.split('=',1);value=value.strip().strip('\"\'')
            if any(word in name.upper() for word in ('KEY','TOKEN','PASSWORD')) and len(value)>=8:
                values.append(value)
    config=ROOT/'esp32_cam/config.h'
    if config.exists():
        for name,value in re.findall(r'#define\s+(WIFI_SSID|WIFI_PASSWORD|API_KEY)\s+"([^"\n]+)"',config.read_text(encoding='utf-8')):
            if len(value)>=8 and not value.startswith('YOUR_'):values.append(value)
    return values

def prepare():
    files={}
    for name in ROOT_FILES:
        path=ROOT/name
        if path.is_file():files[name]=path.read_bytes()
    for folder in ('server','web','scripts','tests','docs','cloud'):
        for path in (ROOT/folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or '.git' in path.parts:continue
            if path.suffix.lower() in CODE_EXTENSIONS or path.name in {'Dockerfile','.dockerignore','.gitignore'}:
                files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for name in ('esp32_cam/esp32_cam.ino','esp32_cam/config.h.example'):
        files[name]=(ROOT/name).read_bytes()
    for name,sha in PINNED_MODELS.items():
        data=(ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Published model changed: '+name)
        files[name]=data
    manifest=json.loads((ROOT/'models/cloud_v7/model_manifest.json').read_text(encoding='utf-8'))
    manifest.update(source_checkpoint='models/camera_v7/face_mask_best.pt',model_file='models/cloud_v7/face_mask_v7.onnx')
    files['models/cloud_v7/model_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode('utf-8')
    files['data/.gitkeep']=b''
    secrets=private_values()
    findings=[]
    for name,data in files.items():
        if name in PINNED_MODELS:continue
        text=data.decode('utf-8')
        if any(secret in text for secret in secrets):findings.append({'file':name,'reason':'matches_local_secret'})
        if re.search(r'\b\d{6,12}:[A-Za-z0-9_-]{30,}\b|\bgh[pousr]_[A-Za-z0-9]{25,}|\bsb_secret_[A-Za-z0-9_-]{20,}',text):
            findings.append({'file':name,'reason':'credential_pattern'})
    report_dir=ROOT/'reports/full_repository';report_dir.mkdir(parents=True,exist_ok=True)
    if findings:
        (report_dir/'blocked_files.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
        raise ValueError('Public-source secret check failed; see blocked_files.json (values omitted)')
    target=ROOT/'.runtime/deployment/full_lab2/source'
    expected=set(files)
    if target.exists():
        existing={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file() and '.git' not in p.relative_to(target).parts}
        if existing-expected:raise ValueError('Source snapshot contains additional files; inspect before preparing again')
    for name,data in files.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    archive=report_dir/'full_lab2_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:
        for name,data in files.items():out.writestr(name,data)
    report={'source_directory':str(target),'archive':str(archive),'file_count':len(files),
            'files':sorted(files),'includes_personal_images':False,'includes_venv':False,'secret_check_passed':True,
            'models_sha256':PINNED_MODELS,'pushed':False}
    (report_dir/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2));return report

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');prepare()
