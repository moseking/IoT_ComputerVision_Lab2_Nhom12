# Face Mask V7 — Render deployment source

This minimal repository contains one exported ONNX model and a CPU API.
The model is part of the Docker image, so Supabase is not required.
Use a **private GitHub repository** for this trained model. No camera images,
dataset, Wi-Fi credentials, Telegram credentials or local `.env` are included.

Create a Render Web Service from this repository, choose Docker and the **Free**
instance, and set these environment variables in Render's dashboard:

- `CLOUD_API_KEY`: the shared key you generated locally; never commit it.
- `CLOUD_MODEL_SHA256`: `f1fc04ca98bace9f1a1a5537d83fc9c52574514c0641959084c94a9f0750cbf4`

The `render.yaml` Blueprint defines the same free service. Keep the default
Docker command; the server binds to Render's `PORT` with one worker.

`GET /health` is public. `GET /model/info` and multipart `POST /predict`
require the `X-API-Key` header. `/predict` accepts JPEG/PNG up to 5 MiB and
4 million pixels. One complete prediction request is processed at a time;
concurrent predictions receive HTTP 429 and can be retried. The CPU model
uses image size 512, confidence 0.35 and class-agnostic NMS IoU 0.45.
The response includes boxes, an annotated image, the uploaded bytes' SHA-256,
model SHA-256, inference time and server processing time.

Keep the camera dashboard running locally: ESP32-CAM → local gateway → Render
→ local dashboard. The cloud API does not save uploaded pictures/results.
SQLite history and Telegram delivery run on the laptop.

This repository is prepared source; deployment and cloud performance must be
verified after creating the service. Render Free sleeps after 15 minutes of
inactivity, so the first request can take about a minute to wake it.
See [Render Free documentation](https://render.com/docs/free).
