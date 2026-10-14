"""One frozen GPU reference scorer; separate cache and no policy/training load."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from .common import ReferenceScores, write_json
from .luna_v2_data import ROOT


def serve():
    score = ReferenceScores(root=ROOT/'reference')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = score(body['question'], body['text'])
                status = 200
            except Exception as exc:
                status = 500; result = {'error': type(exc).__name__, 'detail': str(exc)}
            payload = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
    write_json(ROOT/'score_service.json', {'pid': os.getpid(), 'port': 8064, 'physical_gpu': 1, 'training': False})
    HTTPServer(('127.0.0.1', 8064), Handler).serve_forever()


if __name__ == '__main__': serve()
