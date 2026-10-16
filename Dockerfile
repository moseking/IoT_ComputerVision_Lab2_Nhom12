FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY cloud/render/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 appuser
COPY --chown=10001:10001 server/__init__.py server/inference.py server/inference_onnx.py server/cloud_bootstrap.py server/cloud_app.py server/cloud_entry.py server/latency_store.py server/telegram_service.py server/camera_client.py ./server/
COPY --chown=10001:10001 models/cloud_v7/face_mask_v7.onnx ./models/face_mask_v7.onnx
COPY --chown=10001:10001 web/index.html web/style.css web/app.js ./web/
USER appuser
ENV CLOUD_MODEL_PATH=/app/models/face_mask_v7.onnx
EXPOSE 10000
CMD ["python", "-m", "server.cloud_entry"]
