import json
from http.server import BaseHTTPRequestHandler

from main import MATERIALS, TEST_MATERIALS, TRAIN_MATERIALS


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "materials": sorted(MATERIALS.keys()),
            "train_materials": TRAIN_MATERIALS,
            "test_materials": TEST_MATERIALS,
        }
        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
