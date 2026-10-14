"""Loopback web UI: no source-file routes, model imports, or cross-rater endpoints."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from pydantic import ValidationError

from .store import ConflictError, ReviewStore, encode

STATIC = Path(__file__).with_name("static")
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/styles.css": ("styles.css", "text/css; charset=utf-8")}


def make_server(store, host="127.0.0.1", port=8766):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log tokens or user-supplied URLs.

        def send(self, status, data, content_type="application/json; charset=utf-8", attachment=None):
            if not isinstance(data, bytes):
                data = encode(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(data)

        def origin_ok(self):
            origin = self.headers.get("Origin")
            return origin is None or origin in {f"http://{self.headers.get('Host')}", f"https://{self.headers.get('Host')}"}

        def rater(self):
            if not self.origin_ok():
                self.send(403, {"error": "다른 사이트에서의 요청은 허용되지 않습니다."})
                return None
            auth = self.headers.get("Authorization", "")
            user = store.authenticate(auth[7:] if auth.startswith("Bearer ") else "")
            if user is None:
                self.send(401, {"error": "참여 코드를 확인해 주세요."})
            return user

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ASSETS:
                name, mime = ASSETS[path]
                self.send(200, (STATIC / name).read_bytes(), mime)
                return
            user = self.rater()
            if user is None:
                return
            if path == "/api/session":
                self.send(200, {**store.session(user["rater_id"]), "rater_label": user["label"]})
            elif path == "/api/export":
                rows = store.export_rater(user["rater_id"])
                self.send(200, "".join(encode(r) + "\n" for r in rows).encode(),
                          "application/x-ndjson; charset=utf-8", "my-reviews.jsonl")
            elif re.fullmatch(r"/api/cases/[a-f0-9]{32}", path):
                try:
                    self.send(200, store.case(user["rater_id"], path.rsplit("/", 1)[-1]))
                except KeyError:
                    self.send(404, {"error": "평가 항목을 찾을 수 없습니다."})
            else:
                self.send(404, {"error": "요청한 페이지가 없습니다."})

        def do_PUT(self):
            user = self.rater()
            if user is None:
                return
            path = urlsplit(self.path).path
            if not re.fullmatch(r"/api/cases/[a-f0-9]{32}/review", path):
                self.send(404, {"error": "요청한 페이지가 없습니다."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    self.send(413, {"error": "저장할 내용이 너무 크거나 비어 있습니다."})
                    return
                if self.headers.get_content_type() != "application/json":
                    self.send(415, {"error": "JSON 형식이 필요합니다."})
                    return
                data = json.loads(self.rfile.read(length))
                saved = store.save(user["rater_id"], path.split("/")[3], data)
                self.send(200, {"review": saved})
            except ConflictError as exc:
                self.send(409, {"error": str(exc)})
            except KeyError:
                self.send(404, {"error": "평가 항목을 찾을 수 없습니다."})
            except (ValueError, ValidationError, UnicodeError):
                self.send(400, {"error": "입력을 확인해 주세요. 평가 완료에는 네 기준의 판정과 근거가 모두 필요합니다."})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
