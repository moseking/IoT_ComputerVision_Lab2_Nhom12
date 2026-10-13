"""Build a minimal, secret-free private Render source bundle. No upload/deploy."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('server/__init__.py','server/inference.py','server/inference_onnx.py',
         'server/cloud_bootstrap.py','server/cloud_app.py','server/cloud_entry.py')

def prepare(output=None):
    manifest=json.loads((ROOT/'models/cloud_v7/model_manifest.json').read_text(encoding='utf-8'))
    model=ROOT/'models/cloud_v7/face_mask_v7.onnx'
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    if not manifest.get('parity_verified') or manifest['model_sha256']!=sha:
        raise ValueError('Exported model must pass parity and SHA-256 validation first')
    target=Path(output) if output else ROOT/'.runtime/deployment/render_v7/source'
    target=target.resolve()
    # Add/update only this explicit list. Never recursively delete or copy the project.
    files={name:(ROOT/name).read_bytes() for name in SOURCES}
    for name in ('Dockerfile','requirements.txt','render.yaml','.dockerignore','.gitignore','README.md'):
        files[name]=(ROOT/'cloud/render'/name).read_bytes()
    files['models/face_mask_v7.onnx']=model.read_bytes()
    public_manifest={k:v for k,v in manifest.items() if k not in {'source_checkpoint','model_file'}}
    public_manifest.update(model_file='models/face_mask_v7.onnx',cloud_uploaded=False,cloud_deployed=False)
    files['models/model_manifest.json']=(json.dumps(public_manifest,indent=2)+'\n').encode()
    expected=set(files)
    if target.exists():
        extra={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
        extra={p for p in extra if not p.startswith('.git/')}
        if extra-expected:raise ValueError('Deployment source has additional files; use a new --output directory')
    for name,data in files.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    report_dir=ROOT/'reports/cloud_v7';report_dir.mkdir(parents=True,exist_ok=True)
    archive=report_dir/'render_deploy_v7.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:
        for name,data in files.items():out.writestr(name,data)
    report={'model_sha256':sha,'model_size_bytes':len(files['models/face_mask_v7.onnx']),
            'source_directory':str(target),'archive':str(archive),'files':sorted(files),
            'cloud_deployed':False,'includes_secrets':False,'includes_dataset':False}
    (report_dir/'deployment_bundle.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output')
    prepare(parser.parse_args().output)
