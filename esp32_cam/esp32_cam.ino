/*
 * AI Thinker ESP32-CAM; Arduino-ESP32 3.x; ArduinoJson 7.x.
 * Model inference runs on FastAPI. Camera routes: port 80 / /capture /stream.
 * /stream redirects to a dedicated port 81 server so a live stream cannot
 * monopolize /capture or the IoT inference loop.
 */
#include <Arduino.h>
#include <time.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "config.h"

// Exact AI Thinker mapping from Espressif CameraWebServer.
static constexpr int PWDN_PIN=32, RESET_PIN=-1, XCLK_PIN=0, SIOD_PIN=26, SIOC_PIN=27;
static constexpr int Y9_PIN=35, Y8_PIN=34, Y7_PIN=39, Y6_PIN=36;
static constexpr int Y5_PIN=21, Y4_PIN=19, Y3_PIN=18, Y2_PIN=5;
static constexpr int VSYNC_PIN=25, HREF_PIN=23, PCLK_PIN=22;
static SemaphoreHandle_t cameraMutex;
static httpd_handle_t cameraServer=nullptr, streamServer=nullptr;
static bool cameraReady=false;
static bool flashState=false;
static bool backendUrlValid=true;
static bool wasWifiConnected=false;
static constexpr int FLASH_PIN=4;
static bool timeSyncStarted=false;
static uint32_t nextSend=0, lastWifiRetry=0;
static String pendingReport;
static int pendingRetries=0;

struct Frame {
  uint8_t *data=nullptr;
  size_t size=0;
};

static double elapsedMs(int64_t start, int64_t stop) {
  return static_cast<double>(stop-start)/1000.0;
}

static void releaseFrame(Frame &frame) {
  if (frame.data) free(frame.data);
  frame.data=nullptr;
  frame.size=0;
}

static bool captureFrame(Frame &frame) {
  if (!cameraReady || xSemaphoreTake(cameraMutex, pdMS_TO_TICKS(3000)) != pdTRUE) return false;
  camera_fb_t *fb=esp_camera_fb_get();
  if (!fb) {
    xSemaphoreGive(cameraMutex);
    return false;
  }
  // Copy to release the camera frame buffer before network operations.
  frame.data=static_cast<uint8_t*>(heap_caps_malloc(fb->len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!frame.data) frame.data=static_cast<uint8_t*>(malloc(fb->len));
  if (frame.data) {
    frame.size=fb->len;
    memcpy(frame.data, fb->buf, fb->len);
  }
  esp_camera_fb_return(fb);
  xSemaphoreGive(cameraMutex);
  return frame.data != nullptr;
}

static esp_err_t rootHandler(httpd_req_t *request) {
  const char *page=R"HTML(<!doctype html><html><head><meta name="viewport" content="width=device-width">
<title>ESP32-CAM Camera</title><style>body{font:18px system-ui;background:#101b2b;color:#fff;max-width:850px;margin:32px auto;padding:16px}button,a{padding:12px;margin:6px;color:#111;background:#69dec7;border:0;border-radius:6px}img{width:100%;margin-top:20px}p{color:#bccddd}</style></head>
<body><h1>ESP32-CAM Camera</h1><p>AI Thinker · Real IoT camera · Model runs on server</p>
<button onclick="document.getElementById('view').src='/stream'">Live Stream</button>
<button onclick="document.getElementById('view').src='/capture?t='+Date.now()">Capture</button>
<button onclick="document.getElementById('view').removeAttribute('src')">Stop</button>
<button onclick="toggleFlash()">Flash ON/OFF</button>
<p id="status">Loading real camera image...</p><img id="view" src="/capture" alt="Camera frame">
<script>
const view=document.getElementById('view'), status=document.getElementById('status');
view.onload=()=>status.textContent='Camera frame loaded';
view.onerror=()=>status.textContent='Camera error: check /capture, power supply, ribbon cable and Serial logs.';
async function toggleFlash(){try{const r=await fetch('/flash',{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);
const d=await r.json();status.textContent=d.flash?'Flash ON':'Flash OFF';}catch(e){status.textContent='Flash error: '+e.message;}}
</script></body></html>)HTML";
  httpd_resp_set_type(request, "text/html; charset=utf-8");
  return httpd_resp_send(request, page, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t captureHandler(httpd_req_t *request) {
  Frame frame;
  if (!captureFrame(frame)) { httpd_resp_set_status(request, "503 Service Unavailable"); return httpd_resp_send(request, "Camera busy/unavailable", HTTPD_RESP_USE_STRLEN); }
  httpd_resp_set_type(request, "image/jpeg");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  esp_err_t result=httpd_resp_send(request, reinterpret_cast<const char*>(frame.data), frame.size);
  releaseFrame(frame);
  return result;
}

static esp_err_t flashHandler(httpd_req_t *request) {
  char query[48]={}, value[8]={};
  if (httpd_req_get_url_query_str(request, query, sizeof(query))==ESP_OK &&
      httpd_query_key_value(query, "on", value, sizeof(value))==ESP_OK) {
    if (strcmp(value, "0")!=0 && strcmp(value, "1")!=0) {
      httpd_resp_set_status(request, "400 Bad Request");
      return httpd_resp_send(request, "Use on=0 or on=1", HTTPD_RESP_USE_STRLEN);
    }
    flashState=strcmp(value, "1")==0;
  } else flashState=!flashState;
  digitalWrite(FLASH_PIN, flashState ? HIGH : LOW);
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, flashState ? "{\"flash\":true}" : "{\"flash\":false}", HTTPD_RESP_USE_STRLEN);
}

static esp_err_t streamRedirect(httpd_req_t *request) {
  String location="http://" + WiFi.localIP().toString() + ":81/stream";
  httpd_resp_set_status(request, "302 Found");
  httpd_resp_set_hdr(request, "Location", location.c_str());
  return httpd_resp_send(request, nullptr, 0);
}

static esp_err_t streamHandler(httpd_req_t *request) {
  httpd_resp_set_type(request, "multipart/x-mixed-replace;boundary=frame");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  while (WiFi.status()==WL_CONNECTED) {
    Frame frame;
    if (!captureFrame(frame)) break;
    char header[96];
    int count=snprintf(header, sizeof(header), "\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", static_cast<unsigned>(frame.size));
    esp_err_t err=httpd_resp_send_chunk(request, header, count);
    if (err==ESP_OK) err=httpd_resp_send_chunk(request, reinterpret_cast<const char*>(frame.data), frame.size);
    releaseFrame(frame);
    if (err!=ESP_OK) break;
    // Throttle only the camera preview; excluded from prediction latency.
    vTaskDelay(pdMS_TO_TICKS(STREAM_INTERVAL_MS));
  }
  return ESP_OK;
}

static void startCameraWeb() {
  httpd_config_t config=HTTPD_DEFAULT_CONFIG();
  config.server_port=80;
  config.stack_size=8192;
  if (httpd_start(&cameraServer, &config)==ESP_OK) {
    httpd_uri_t root={};
    root.uri="/"; root.method=HTTP_GET; root.handler=rootHandler;
    httpd_register_uri_handler(cameraServer, &root);
    httpd_uri_t capture={};
    capture.uri="/capture"; capture.method=HTTP_GET; capture.handler=captureHandler;
    httpd_register_uri_handler(cameraServer, &capture);
    httpd_uri_t flash={};
    flash.uri="/flash"; flash.method=HTTP_GET; flash.handler=flashHandler;
    httpd_register_uri_handler(cameraServer, &flash);
    httpd_uri_t stream={};
    stream.uri="/stream"; stream.method=HTTP_GET; stream.handler=streamRedirect;
    httpd_register_uri_handler(cameraServer, &stream);
  }
  config.server_port=81;
  config.ctrl_port+=1;
  if (httpd_start(&streamServer, &config)==ESP_OK) {
    httpd_uri_t stream={};
    stream.uri="/stream"; stream.method=HTTP_GET; stream.handler=streamHandler;
    httpd_register_uri_handler(streamServer, &stream);
  }
}

static bool beginHttp(HTTPClient &http, WiFiClient &plain, WiFiClientSecure &secure, const char *url) {
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
  http.setReuse(false);
  bool ok;
  if (String(url).startsWith("https://")) {
    if (strlen(TLS_ROOT_CA)==0) {
      Serial.println("HTTPS requires TLS_ROOT_CA in config.h");
      return false;
    }
    if (time(nullptr) < 1700000000) { Serial.println("Waiting for NTP clock before HTTPS"); return false; }
    secure.setCACert(TLS_ROOT_CA);
    ok=http.begin(secure, url);
  } else {
    ok=http.begin(plain, url);
  }
  if (ok && strlen(API_KEY)>0) http.addHeader("X-API-Key", API_KEY);
  return ok;
}

static bool sendReport(const String &body) {
  HTTPClient http;
  WiFiClient plain;
  WiFiClientSecure secure;
  if (!beginHttp(http, plain, secure, METRICS_URL)) return false;
  http.addHeader("Content-Type", "application/json");
  int status=http.POST(body);
  String reply=http.getString();
  http.end();
  if (status!=200) Serial.printf("Metrics report failed HTTP=%d %s\n", status, reply.c_str());
  return status==200;
}

static void sendFrame() {
  const int64_t totalStart=esp_timer_get_time();
  const int64_t captureStart=totalStart;
  Frame frame;
  if (!captureFrame(frame)) {
    Serial.println("Capture failed; no prediction sent.");
    return;
  }
  const int64_t captureDone=esp_timer_get_time();
  const double captureMs=elapsedMs(captureStart, captureDone);
  String boundary="----ESP32Mask" + String(static_cast<uint32_t>(esp_random()), HEX);
  String prefix="--" + boundary + "\r\nContent-Disposition: form-data; name=\"source\"\r\n\r\nesp32\r\n";
  prefix+="--" + boundary + "\r\nContent-Disposition: form-data; name=\"capture_ms\"\r\n\r\n" + String(captureMs, 3) + "\r\n";
  prefix+="--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
  String suffix="\r\n--" + boundary + "--\r\n";
  size_t payloadSize=prefix.length() + frame.size + suffix.length();
  uint8_t *payload=static_cast<uint8_t*>(heap_caps_malloc(payloadSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!payload) payload=static_cast<uint8_t*>(malloc(payloadSize));
  if (!payload) {
    releaseFrame(frame);
    Serial.println("Not enough RAM for multipart upload");
    return;
  }
  memcpy(payload, prefix.c_str(), prefix.length());
  memcpy(payload + prefix.length(), frame.data, frame.size);
  memcpy(payload + prefix.length() + frame.size, suffix.c_str(), suffix.length());
  releaseFrame(frame);

  Serial.println("\n================================\nESP32-CAM MASK DETECTION\nSending image...");
  HTTPClient http;
  WiFiClient plain;
  WiFiClientSecure secure;
  const int64_t requestStart=esp_timer_get_time();
  if (!beginHttp(http, plain, secure, SERVER_URL)) {
    free(payload);
    return;
  }
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  int status=http.POST(payload, payloadSize);
  const int64_t headersReceived=esp_timer_get_time();
  String response=http.getString();
  const int64_t bodyReceived=esp_timer_get_time();
  free(payload);
  http.end();
  const double roundtripMs=elapsedMs(requestStart, bodyReceived);
  const double totalMs=elapsedMs(totalStart, bodyReceived);
  const double downloadMs=elapsedMs(headersReceived, bodyReceived);
  Serial.printf("Response: %d\n", status);
  if (status!=200) {
    Serial.println(status < 0 ? HTTPClient::errorToString(status) : response);
    return;
  }
  JsonDocument result;
  DeserializationError error=deserializeJson(result, response);
  if (error || !(result["success"] | false)) {
    Serial.println("Invalid prediction JSON");
    return;
  }
  Serial.printf("Request: %s\n", result["request_id"].as<const char*>());
  Serial.printf("Detected: %d faces\n", result["detection_count"].as<int>());
  int person=0;
  for (JsonObject detection : result["detections"].as<JsonArray>()) {
    Serial.printf("Person %d:\n  Class: %s\n  Confidence: %.1f%%\n", ++person,
      detection["class_name"].as<const char*>(), detection["confidence"].as<float>() * 100);
  }
  Serial.printf("Capture: %.3f ms\nInference: %.3f ms\nRound Trip: %.3f ms\nTotal: %.3f ms\n",
      captureMs, result["inference_ms"].as<double>(), roundtripMs, totalMs);
  Serial.printf("Body read after headers: %.3f ms\nUpload: unavailable separately\n================================\n", downloadMs);
  JsonDocument metrics;
  metrics["request_id"]=result["request_id"];
  metrics["report_token"]=result["report_token"];
  metrics["capture_ms"]=captureMs;
  metrics["roundtrip_ms"]=roundtripMs;
  metrics["total_ms"]=totalMs;
  metrics["upload_ms"]=nullptr;
  metrics["download_ms"]=downloadMs;
  String report;
  serializeJson(metrics, report);
  if (!sendReport(report)) {
    pendingReport=report;
    pendingRetries=0;
  }
}

static bool initializeCamera() {
  camera_config_t config={};
  config.ledc_channel=LEDC_CHANNEL_0;
  config.ledc_timer=LEDC_TIMER_0;
  config.pin_d0=Y2_PIN; config.pin_d1=Y3_PIN; config.pin_d2=Y4_PIN; config.pin_d3=Y5_PIN;
  config.pin_d4=Y6_PIN; config.pin_d5=Y7_PIN; config.pin_d6=Y8_PIN; config.pin_d7=Y9_PIN;
  config.pin_xclk=XCLK_PIN; config.pin_pclk=PCLK_PIN; config.pin_vsync=VSYNC_PIN; config.pin_href=HREF_PIN;
  config.pin_sccb_sda=SIOD_PIN; config.pin_sccb_scl=SIOC_PIN;
  config.pin_pwdn=PWDN_PIN; config.pin_reset=RESET_PIN;
  config.xclk_freq_hz=20000000;
  config.pixel_format=PIXFORMAT_JPEG;
  config.frame_size=CAMERA_FRAME_SIZE;
  config.jpeg_quality=JPEG_QUALITY;
  config.fb_count=psramFound() ? 2 : 1;
  config.grab_mode=psramFound() ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location=psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  if (!psramFound()) config.frame_size=FRAMESIZE_QVGA;
  esp_err_t status=esp_camera_init(&config);
  if (status!=ESP_OK) Serial.printf("Camera initialization failed 0x%x\n", status);
  return status==ESP_OK;
}

void setup() {
  Serial.begin(115200);
  pinMode(FLASH_PIN, OUTPUT);
  digitalWrite(FLASH_PIN, LOW);
  String backend(SERVER_URL);
  backendUrlValid=(backend.startsWith("http://") || backend.startsWith("https://")) &&
      backend.indexOf("://localhost")<0 && backend.indexOf("://127.")<0 && backend.indexOf("://0.0.0.0")<0;
  if (!backendUrlValid) Serial.println("Invalid SERVER_URL: use laptop LAN IP or cloud domain. Camera web remains available.");
  cameraMutex=xSemaphoreCreateMutex();
  cameraReady=cameraMutex && initializeCamera();
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  uint32_t started=millis();
  while (WiFi.status()!=WL_CONNECTED && millis()-started < 20000UL) delay(200);
  if (WiFi.status()==WL_CONNECTED) {
    Serial.printf("WiFi: Connected\nIP: %s\nCamera: http://%s/\n", WiFi.localIP().toString().c_str(), WiFi.localIP().toString().c_str());
  } else Serial.println("WiFi not connected; retrying in loop");
  startCameraWeb();
}

void loop() {
  if (WiFi.status()!=WL_CONNECTED) {
    wasWifiConnected=false;
    if (millis()-lastWifiRetry >= WIFI_RETRY_MS) {
      lastWifiRetry=millis();
      WiFi.reconnect();
      Serial.println("WiFi reconnect requested");
    }
    delay(10); // Scheduler yield, outside all measured intervals.
    return;
  }
  if (!wasWifiConnected) {
    wasWifiConnected=true;
    Serial.printf("WiFi connected/reconnected; Camera: http://%s/\n", WiFi.localIP().toString().c_str());
  }
  if (!timeSyncStarted && (String(SERVER_URL).startsWith("https://") || String(METRICS_URL).startsWith("https://"))) {
    configTime(0, 0, "pool.ntp.org", "time.google.com");
    timeSyncStarted=true;
  }
  if (static_cast<int32_t>(millis()-nextSend) >= 0) {
    if (pendingReport.length()>0) {
      if (sendReport(pendingReport)) pendingReport="";
      else if (++pendingRetries>=3) {
        Serial.println("Metrics retry exhausted; server history will retain null client timings");
        pendingReport="";
      }
    }
    // Do not replace a pending measurement before retry completes.
    if (pendingReport.length()==0 && cameraReady && backendUrlValid && INFERENCE_ENABLED) sendFrame();
    nextSend=millis() + SEND_INTERVAL_MS;
  }
  delay(5); // Rate control is not used to synthesize latency.
}
