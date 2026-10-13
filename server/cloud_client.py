"""Local gateway to the Render API; validates model identity and the exact frame."""
import base64
import hashlib
import ipaddress
import json
import math
import re
from datetime import datetime
import time
from dataclasses import dataclass,field
from urllib.parse import urlsplit
import httpx
from .inference import CLASSES
from .camera_client import validate_capture,CameraError

class CloudError(RuntimeError):pass

@dataclass(frozen=True)
class CloudConfig:
    url:str=''
    key:str=field(default='',repr=False)
    sha256:str=''
    timeout:float=90

class CloudInferenceClient:
    def __init__(self,config,client=None):
        self.config=config;self.client=client;self._owned=client is None
        self._verified=False;self._error=None;self._info=None
        if config.url:
            parts=urlsplit(config.url)
            if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path not in {'','/'}:
                raise ValueError('CLOUD_API_URL must be an HTTPS origin without credentials')
            try:address=ipaddress.ip_address(parts.hostname)
            except ValueError:address=None
            if address and not address.is_global:raise ValueError('CLOUD_API_URL must be a public cloud endpoint')

    @property
    def configured(self):return bool(self.config.url and len(self.config.key)>=24 and re.fullmatch(r'[0-9a-f]{64}',self.config.sha256) is not None)

    def status(self):
        return {'configured':self.configured,'verified':self._verified,'last_error':self._error,
                'model_sha256':self.config.sha256 if self.configured else None,
                'state':'ready' if self._verified else 'error' if self._error else 'configured' if self.configured else 'not_configured'}

    def _http(self):
        if not self.configured:raise CloudError('Cloud chưa cấu hình URL/key/model SHA-256')
        if self.client is None:
            self.client=httpx.Client(timeout=httpx.Timeout(self.config.timeout,connect=15),trust_env=False,follow_redirects=False)
        return self.client

    def close(self):
        if self._owned and self.client:self.client.close()

    def verify(self):
        try:
            response=self._http().get(self.config.url.rstrip('/')+'/model/info',headers={'X-API-Key':self.config.key})
            if response.status_code!=200:raise CloudError(f'Cloud model info HTTP {response.status_code}')
            info=response.json()
            if not isinstance(info,dict) or not isinstance(info.get('classes'),dict):raise CloudError('Cloud model info invalid')
            if not info.get('loaded') or info.get('model_sha256')!=self.config.sha256:
                raise CloudError('Cloud chưa load đúng model V7 đã xuất')
            if {int(k):v for k,v in info.get('classes',{}).items()}!=CLASSES:
                raise CloudError('Cloud model classes mismatch')
            self._verified=True;self._error=None;self._info=info
            return info
        except CloudError as exc:
            self._error=str(exc);self._verified=False;raise
        except (httpx.HTTPError,ValueError,TypeError) as exc:
            self._error=type(exc).__name__;self._verified=False
            raise CloudError('Cloud verification failed: '+type(exc).__name__) from None

    def predict(self,data,width,height):
        try:
            if not self._verified:self.verify()
            start=time.perf_counter()
            with self._http().stream('POST',self.config.url.rstrip('/')+'/predict',
                                     headers={'X-API-Key':self.config.key},
                                     files={'file':('frame.jpg',data,'application/octet-stream')}) as response:
                if response.status_code!=200:raise CloudError(f'Cloud predict HTTP {response.status_code}')
                raw=bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw)>12*1024*1024:raise CloudError('Cloud response exceeds limit')
            elapsed=(time.perf_counter()-start)*1000
            result=json.loads(raw)
            if not isinstance(result,dict):raise CloudError('Cloud result invalid')
            if result.get('success') is not True or result.get('model_sha256')!=self.config.sha256:
                raise CloudError('Cloud result model mismatch')
            if result.get('runtime')!='onnxruntime' or result.get('inference_location')!='cloud':raise CloudError('Cloud runtime invalid')
            if not re.fullmatch(r'[0-9a-f]{32}',result.get('request_id','')):raise CloudError('Cloud request ID invalid')
            timestamp=datetime.fromisoformat(result['timestamp'])
            if timestamp.tzinfo is None:raise CloudError('Cloud timestamp must include timezone')
            if result.get('input_sha256')!=hashlib.sha256(data).hexdigest():raise CloudError('Cloud input frame mismatch')
            if result.get('width')!=width or result.get('height')!=height:raise CloudError('Cloud image dimensions mismatch')
            detections=result.get('detections')
            if not isinstance(detections,list) or len(detections)>300 or result.get('detection_count')!=len(detections):
                raise CloudError('Cloud detections invalid')
            for d in detections:
                if not isinstance(d,dict) or not isinstance(d.get('bbox'),dict):raise CloudError('Cloud detection invalid')
                cls=d.get('class_id');conf=d.get('confidence');box=d.get('bbox',{})
                if type(cls)!=int or cls not in CLASSES or d.get('class_name')!=CLASSES[cls]:raise CloudError('Cloud class invalid')
                if type(conf) not in {int,float} or not math.isfinite(conf) or not 0<=conf<=1:raise CloudError('Cloud confidence invalid')
                coords=[box.get(k) for k in ['x1','y1','x2','y2']]
                if any(type(v) not in {int,float} or not math.isfinite(v) for v in coords):raise CloudError('Cloud bbox invalid')
                x1,y1,x2,y2=coords
                if not 0<=x1<=x2<=width or not 0<=y1<=y2<=height:raise CloudError('Cloud bbox out of frame')
            for key in ['inference_ms','server_processing_ms','queue_ms']:
                value=result.get(key)
                if type(value) not in {float,int} or not math.isfinite(value) or value<0:raise CloudError('Cloud timing invalid')
            if result['inference_ms']>result['server_processing_ms']+2:raise CloudError('Cloud timing inconsistent')
            annotated=base64.b64decode(result['annotated_image_base64'],validate=True)
            try:dimensions=validate_capture(annotated)[:2]
            except CameraError:raise CloudError('Cloud annotation invalid') from None
            if dimensions!=(width,height):raise CloudError('Cloud annotation frame dimensions mismatch')
            self._error=None
            return {**result,'cloud_roundtrip_ms':round(elapsed,3)},annotated
        except (httpx.HTTPError,ValueError,KeyError,TypeError) as exc:
            self._error=type(exc).__name__
            raise CloudError('Cloud inference failed: '+type(exc).__name__) from None
        except CloudError as exc:
            self._error=str(exc);raise
