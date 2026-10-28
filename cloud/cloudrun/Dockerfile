FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080
COPY cloud/cloudrun/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip check
RUN useradd --create-home --uid 10001 appuser
COPY --chown=10001:10001 server/__init__.py server/inference.py server/inference_onnx.py server/cloud_bootstrap.py server/cloud_app.py server/cloud_entry.py ./server/
COPY --chown=10001:10001 scripts/verify_cloudrun.py ./scripts/verify_cloudrun.py
USER appuser
ENV CLOUD_MODEL_PATH=/tmp/facemask/face_mask_v7.onnx
EXPOSE 8080
CMD ["python", "scripts/verify_cloudrun.py", "serve"]
