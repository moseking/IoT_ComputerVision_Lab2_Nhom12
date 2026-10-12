"""Download one pinned model from a scoped HTTPS link. Never log signed URLs."""
import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
import httpx


def ensure_model(path, expected_sha256, download_url, client=None):
    path=Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}",expected_sha256 or ""):
        raise ValueError("CLOUD_MODEL_SHA256 must pin the exported model")
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError("Existing cloud model SHA-256 mismatch")
        return path
    parts=urlsplit(download_url or "")
    if parts.scheme!="https" or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError("Configure a valid HTTPS CLOUD_MODEL_DOWNLOAD_URL")
    if not parts.hostname.endswith(".supabase.co"):
        raise ValueError("Model download must use the configured Supabase Storage signed URL")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.download')
    owned=client is None
    client=client or httpx.Client(timeout=httpx.Timeout(90,connect=10),trust_env=False,follow_redirects=False)
    try:
        with client.stream('GET',download_url) as response:
            if response.status_code!=200: raise ValueError(f"Model download HTTP {response.status_code}; refresh an expired signed URL")
            size=0;digest=hashlib.sha256()
            with temporary.open('wb') as output:
                for chunk in response.iter_bytes():
                    size+=len(chunk)
                    if size>32*1024*1024:raise ValueError("Cloud model exceeds 32 MiB")
                    digest.update(chunk);output.write(chunk)
        if digest.hexdigest()!=expected_sha256:raise ValueError("Downloaded model SHA-256 mismatch")
        temporary.replace(path)
        return path
    except httpx.HTTPError as exc:
        raise RuntimeError("Cloud model download failed: "+type(exc).__name__) from None
    finally:
        if temporary.exists():temporary.unlink()
        if owned:client.close()
