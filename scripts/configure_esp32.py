"""Generate ignored firmware config.h from .env without printing secrets."""
import argparse
import json
import os
from urllib.parse import urlsplit
from dotenv import load_dotenv
from common import ROOT


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    values = {"WIFI_SSID": os.getenv("WIFI_SSID", ""), "WIFI_PASSWORD": os.getenv("WIFI_PASSWORD", ""),
              "SERVER_URL": os.getenv("ESP32_SERVER_URL", ""), "API_KEY": os.getenv("API_KEY", "")}
    if not all(values[k] for k in ("WIFI_SSID", "WIFI_PASSWORD", "SERVER_URL")):
        parser.error("Set WIFI_SSID, WIFI_PASSWORD, ESP32_SERVER_URL in .env first")
    url = urlsplit(values["SERVER_URL"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}:
        parser.error("Use the computer LAN IP or a reachable cloud hostname")
    values["METRICS_URL"] = values["SERVER_URL"].rstrip("/").rsplit("/", 1)[0] + "/metrics/report"
    output = ROOT / "esp32_cam/config.h"
    if output.exists() and not args.overwrite:
        parser.error("config.h exists; use --overwrite to replace")
    interval = int(os.getenv("SEND_INTERVAL_MS", "2000"))
    if not 100 <= interval <= 3600000:
        parser.error("SEND_INTERVAL_MS must be 100..3600000")
    lines = []
    for line in (ROOT / "esp32_cam/config.h.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("#define "):
            key = line.split()[1]
            if key in values:
                line = f"#define {key} {json.dumps(values[key], ensure_ascii=True)}"
            elif key == "SEND_INTERVAL_MS":
                line = f"#define SEND_INTERVAL_MS {interval}UL"
        lines.append(line)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("config.h generated (secrets hidden). Recompile/upload to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
