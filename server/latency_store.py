"""SQLite history: server metrics first, client measurements reported after response."""
from datetime import datetime, timezone
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path


COUNT_COLUMNS = ("with_mask_count", "without_mask_count", "incorrect_mask_count")
CLASS_NAMES = ("with_mask", "without_mask", "mask_weared_incorrect")


def recognition_counts(detections):
    """Counts come from the returned model detections, never the browser."""
    counts = [0, 0, 0]
    for detection in detections:
        class_id = detection["class_id"]
        if type(class_id) is not int or class_id not in (0, 1, 2):
            raise ValueError("Invalid recognition class ID")
        counts[class_id] += 1
    return dict(zip(COUNT_COLUMNS, counts))


class LatencyStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS latency_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    request_id TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    detection_count INTEGER NOT NULL,
                    inference_ms REAL NOT NULL,
                    server_processing_ms REAL NOT NULL,
                    roundtrip_ms REAL,
                    total_ms REAL,
                    capture_ms REAL,
                    upload_ms REAL,
                    download_ms REAL,
                    network_overhead_estimated_ms REAL,
                    image_size_bytes INTEGER NOT NULL,
                    report_token TEXT NOT NULL,
                    reported INTEGER NOT NULL DEFAULT 0
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_latency_timestamp ON latency_history(timestamp)")
            # Separate failure table preserves the original NOT NULL measurements and old rows.
            db.execute("""CREATE TABLE IF NOT EXISTS request_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                request_id TEXT UNIQUE NOT NULL, source TEXT NOT NULL,
                http_status INTEGER NOT NULL, error_code TEXT NOT NULL, image_size_bytes INTEGER)""")

            # Additive migration keeps all historical measurements and tokens intact.
            for table in ("latency_history", "request_failures"):
                columns = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
                additions = {"inference_location": "TEXT NOT NULL DEFAULT 'local'"}
                if table == "latency_history":
                    additions.update(cloud_roundtrip_ms="REAL", cloud_model_sha256="TEXT")
                    # NULL explicitly means the old request did not save its labels.
                    additions.update({name: "INTEGER" for name in COUNT_COLUMNS})
                for name, definition in additions.items():
                    if name not in columns:db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def insert(self, record):
        counts = [record.get(k) for k in COUNT_COLUMNS]
        if any(value is not None for value in counts):
            if any(type(value) is not int or value < 0 for value in counts) or sum(counts) != record["detection_count"]:
                raise ValueError("Recognition counts must match detection_count")
        token = secrets.token_urlsafe(24)
        columns = ["timestamp", "request_id", "source", "detection_count", "inference_ms",
                   "server_processing_ms", "capture_ms", "image_size_bytes",
                   "inference_location", "cloud_roundtrip_ms", "cloud_model_sha256", *COUNT_COLUMNS]
        values = [record.get(k, "local" if k == "inference_location" else None) for k in columns]
        with self.connect() as db:
            db.execute(f"INSERT INTO latency_history ({','.join(columns)},report_token) VALUES ({','.join('?' for _ in columns)},?)",
                       [*values, token])
        return token

    def report(self, request_id, token, values):
        with self.connect() as db:
            # Reserve the write transaction before SELECT: concurrent retries keep the first report.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM latency_history WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown request_id")
            if not secrets.compare_digest(row["report_token"], token):
                raise PermissionError("Invalid report_token")
            if row["reported"]:
                # Retry is idempotent; it never replaces measurements already persisted.
                return self.public(row)
            # The local gateway measures the WAN request. Browser reports only end-to-end.
            roundtrip = row["cloud_roundtrip_ms"] if row["inference_location"] == "cloud" and row["cloud_roundtrip_ms"] is not None else values["roundtrip_ms"]
            residual = roundtrip - row["server_processing_ms"]
            db.execute("""UPDATE latency_history SET roundtrip_ms=?,total_ms=?,capture_ms=?,
                       upload_ms=?,download_ms=?,network_overhead_estimated_ms=?,reported=1 WHERE request_id=?""",
                       (roundtrip, values["total_ms"], values.get("capture_ms"),
                        values.get("upload_ms"), values.get("download_ms"),
                        residual if residual >= 0 else None, request_id))
            return self.public(db.execute("SELECT * FROM latency_history WHERE request_id=?", (request_id,)).fetchone())

    @staticmethod
    def public(row):
        result = {k: row[k] for k in row.keys() if k != "report_token"}
        counts = [result.get(k) for k in COUNT_COLUMNS]
        result["class_counts"] = dict(zip(CLASS_NAMES, counts)) if all(value is not None for value in counts) else None
        result["end_to_end_ms"] = result.get("total_ms")
        result.setdefault("status", "success")
        result.setdefault("http_status", 200)
        result.setdefault("error_code", None)
        return result

    def history(self, limit=50, offset=0):
        with self.connect() as db:
            rows = [self.public(r) for r in db.execute("SELECT * FROM latency_history ORDER BY id DESC LIMIT ?", (limit+offset,))]
            failures = [dict(r) for r in db.execute("SELECT * FROM request_failures ORDER BY id DESC LIMIT ?", (limit+offset,))]
        for failure in failures:
            failure.update({k: None for k in ("detection_count", "inference_ms", "server_processing_ms",
                "roundtrip_ms", "total_ms", "end_to_end_ms", "capture_ms", "upload_ms", "download_ms",
                "network_overhead_estimated_ms", *COUNT_COLUMNS)})
            failure.update(status="failure", reported=0, class_counts=None)
        return sorted(rows+failures, key=lambda r: (r["timestamp"], r["id"]), reverse=True)[offset:offset+limit]

    def get(self, request_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM latency_history WHERE request_id=?", (request_id,)).fetchone()
            return self.public(row) if row else None

    def latest_device_timestamp(self):
        with self.connect() as db:
            row = db.execute("SELECT timestamp FROM latency_history WHERE source IN ('esp32','camera_pull') ORDER BY id DESC LIMIT 1").fetchone()
            return row["timestamp"] if row else None

    def summary(self):
        with self.connect() as db:
            result = {"total_requests": db.execute("SELECT COUNT(*) FROM latency_history").fetchone()[0]}
            for name in ("inference_ms", "roundtrip_ms", "total_ms", "server_processing_ms"):
                row = db.execute(f"SELECT COUNT({name}) AS samples, AVG({name}) AS average, MIN({name}) AS min, MAX({name}) AS max FROM latency_history").fetchone()
                result[name] = dict(row)
            complete = " AND ".join(name + " IS NOT NULL" for name in COUNT_COLUMNS)
            labeled = db.execute(f"SELECT COUNT(*) FROM latency_history WHERE {complete}").fetchone()[0]
            recognition = {"labeled_requests": labeled, "unlabeled_requests": result["total_requests"] - labeled,
                           "total_faces": 0, "classes": {}}
            for name, column in zip(CLASS_NAMES, COUNT_COLUMNS):
                values = db.execute(f"SELECT COALESCE(SUM({column}),0) AS detections, COALESCE(SUM(CASE WHEN {column}>0 THEN 1 ELSE 0 END),0) AS requests FROM latency_history WHERE {complete}").fetchone()
                recognition["classes"][name] = dict(values)
                recognition["total_faces"] += values["detections"]
            result["recognition"] = recognition
            result["failed_requests"] = db.execute("SELECT COUNT(*) FROM request_failures").fetchone()[0]
            result["total_events"] = result["total_requests"] + result["failed_requests"]
            result["end_to_end_ms"] = result["total_ms"]
            result.update({"average_inference_ms": result["inference_ms"]["average"],
                           "average_roundtrip_ms": result["roundtrip_ms"]["average"],
                           "average_total_ms": result["total_ms"]["average"]})
            return result

    def record_failure(self, request_id, source, http_status, error_code, image_size_bytes=None, inference_location="local"):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO request_failures (timestamp,request_id,source,http_status,error_code,image_size_bytes,inference_location) VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(),request_id,source,http_status,error_code,image_size_bytes,inference_location))

    def latest_failure(self, source):
        with self.connect() as db:
            row = db.execute("SELECT timestamp,request_id,http_status,error_code FROM request_failures WHERE source=? ORDER BY id DESC LIMIT 1", (source,)).fetchone()
            return dict(row) if row else None
