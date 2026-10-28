"""Render CPU API: a pinned single ONNX model is bundled with the service."""
import asyncio
import base64
import hashlib
import io
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI,Depends,File,Header,HTTPException,UploadFile
from fastapi.responses import JSONResponse
from .cloud_bootstrap import ensure_model
from .inference import CLASSES,InvalidImage
from .inference_onnx import OnnxMaskDetectionModel

KEY=os.getenv('CLOUD_API_KEY','')
MODEL_PATH=Path(os.getenv('CLOUD_MODEL_PATH','/tmp/facemask/face_mask_v7.onnx'))
SHA=os.getenv('CLOUD_MODEL_SHA256','')
MAX_BYTES=5*1024*1024
model=OnnxMaskDetectionModel(MODEL_PATH,expected_sha256=SHA,max_pixels=4000000)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app):
    if len(KEY)<24:raise RuntimeError('CLOUD_API_KEY must contain at least 24 characters')
    ensure_model(MODEL_PATH,SHA,os.getenv('CLOUD_MODEL_DOWNLOAD_URL',''))
    model.load_model()
    yield

def authorize(x_api_key:Annotated[str|None,Header()]=None):
    if not KEY or not secrets.compare_digest(x_api_key or '',KEY):raise HTTPException(401,'Missing or invalid X-API-Key')

app=FastAPI(title='Face Mask V7 Cloud API',version='1.0',lifespan=lifespan)

class BodyLimit:
    def __init__(self,app):
        self.app=app;self.gate=asyncio.Semaphore(1)
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('method')!='POST':return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]))
        supplied=headers.get(b'x-api-key',b'').decode('latin1')
        if not KEY or not secrets.compare_digest(supplied,KEY):
            return await JSONResponse({'detail':'Missing or invalid X-API-Key'},401)(scope,receive,send)
        # One complete request at a time bounds memory on Render Free.
        if self.gate.locked():return await JSONResponse({'detail':'Cloud busy; retry shortly'},429)(scope,receive,send)
        await self.gate.acquire()
        try:return await self.bounded(scope,receive,send)
        finally:self.gate.release()

    async def bounded(self,scope,receive,send):
        chunks=[];size=0;deadline=time.perf_counter()+30
        while True:
            try:message=await asyncio.wait_for(receive(),max(.001,deadline-time.perf_counter()))
            except asyncio.TimeoutError:return await JSONResponse({'detail':'Upload timed out'},408)(scope,receive,send)
            if message['type']=='http.disconnect':return
            chunk=message.get('body',b'');size+=len(chunk)
            if size>MAX_BYTES+65536:return await JSONResponse({'detail':'Request exceeds upload limit'},413)(scope,receive,send)
            chunks.append(chunk)
            if not message.get('more_body',False):break
        used=False
        async def replay():
            nonlocal used
            if not used:used=True;return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        await self.app(scope,replay,send)
app.add_middleware(BodyLimit)

@app.get('/health')
def health():return {'status':'ok' if model.ready else 'model_unavailable','model_loaded':model.ready}

@app.get('/model/info',dependencies=[Depends(authorize)])
def info():return {'loaded':model.ready,'model_name':'YOLOv8n Face Mask V7','checkpoint':MODEL_PATH.name,
                   'classes':CLASSES,'image_size':512,'device':'cpu','runtime':'onnxruntime',
                   'model_sha256':model.artifact_sha256,'confidence_threshold':.35,'nms_iou':.45,
                   'nms_class_agnostic':True,'warmup_ms':model.warmup_ms}

@app.post('/predict',dependencies=[Depends(authorize)])
def predict(file:Annotated[UploadFile,File()]):
    start=time.perf_counter()
    try:
        data=file.file.read(MAX_BYTES+1)
        if len(data)>MAX_BYTES:raise HTTPException(413,'Image exceeds upload limit')
        image=model.decode(data);prediction=model.predict(image)
        annotated=model.annotate(image,prediction['detections'])
        return {'success':True,'request_id':uuid.uuid4().hex,'timestamp':datetime.now(timezone.utc).isoformat(),
                **prediction,'detections':prediction['detections'],'detection_count':len(prediction['detections']),
                'server_processing_ms':round((time.perf_counter()-start)*1000,3),
                'inference_location':'cloud','runtime':'onnxruntime','model_sha256':model.artifact_sha256,
                'input_sha256':hashlib.sha256(data).hexdigest(),'width':image.shape[1],'height':image.shape[0],
                'annotated_image_base64':base64.b64encode(annotated).decode('ascii')}
    except InvalidImage as exc:raise HTTPException(400,str(exc)) from None
    finally:file.file.close()
