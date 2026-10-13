"""Bounded camera control via the backend; no browser-to-camera CORS dependency."""
from ipaddress import IPv4Address, IPv4Network
from urllib.parse import urlsplit
import httpx
from .camera_client import CameraError, camera_endpoint


def lan_camera_url(value):
    try:
        parts=urlsplit(value.strip())
        address=IPv4Address(parts.hostname or '')
        port=parts.port
    except (ValueError,TypeError) as exc:
        raise CameraError('Camera URL phải là http(s)://IP_LAN_CAMERA, ví dụ http://192.168.137.28') from exc
    lan=any(address in IPv4Network(network) for network in ['10.0.0.0/8','172.16.0.0/12','192.168.0.0/16'])
    if (parts.scheme not in {'http','https'} or parts.username or parts.password
        or not lan
        or parts.query or parts.fragment or parts.path not in {'','/'}):
        raise CameraError('Dùng IP LAN camera, không kèm credentials/path/query; không dùng localhost hoặc link-local')
    if port is not None and not 1<=port<=65535:raise CameraError('Invalid camera port')
    return value.strip().rstrip('/')


def flash(client,base,enabled):
    base=lan_camera_url(base)
    try:
        with client.stream('GET',camera_endpoint(base,'/flash'),params={'on':'1' if enabled else '0'}) as response:
            if response.status_code!=200:raise CameraError(f'Camera /flash HTTP {response.status_code}')
            data=bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data)>4096:raise CameraError('Camera flash response exceeds limit')
        import json
        payload=json.loads(data)
        if not isinstance(payload,dict) or type(payload.get('flash')) is not bool:
            raise CameraError('Camera did not confirm flash state')
        if payload['flash'] is not enabled:raise CameraError('Camera confirmed a different flash state')
        return {'flash':payload['flash'],'confirmed':True}
    except httpx.TimeoutException as exc:
        raise CameraError('Camera flash timed out; trạng thái chưa xác nhận') from exc
    except httpx.RequestError as exc:
        raise CameraError('Camera unreachable; kiểm tra Wi-Fi/IP/port 80') from exc
    except (ValueError,UnicodeDecodeError) as exc:
        raise CameraError('Invalid camera flash JSON; trạng thái chưa xác nhận') from exc
