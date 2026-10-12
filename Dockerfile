FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 appuser
COPY --chown=10001:10001 server/ server/
COPY --chown=10001:10001 models/ models/
USER appuser
ENV CLOUD_MODEL_PATH=/app/models/face_mask_v7.onnx
EXPOSE 10000
CMD ["python", "-m", "server.cloud_entry"]
