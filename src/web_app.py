"""Test UI backend: serves public/index.html and a /api/predict endpoint.

Locally:  .venv/Scripts/python src/web_app.py   then open http://127.0.0.1:8765
On Vercel, api/predict.py reuses Handler below and public/ is served as static
files. Needs only NumPy and OpenCV (models run through src/inference.py)."""
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Refuse to decode images bigger than ~40 megapixels: a small PNG can claim huge
# dimensions and make the decoder allocate gigabytes. Must be set before cv2 loads.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(40_000_000))

import cv2
import numpy as np

import inference
from preprocess import FRAMING, preprocess

HOST, PORT = "127.0.0.1", 8765
INDEX_HTML = Path(__file__).resolve().parent.parent / "public" / "index.html"
# Vercel rejects request bodies over 4.5 MB; the page shrinks photos well below this
MAX_UPLOAD_BYTES = 4 * 1024 * 1024


def run_prediction(image_bytes, script):
    if script not in FRAMING:
        raise ValueError(f"Unknown script {script!r}")
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read that file as an image")

    tensor, final_img = preprocess(image, script)
    if tensor is None:
        return {"found": False}

    probabilities = inference.load_model(script).predict(tensor)[0]
    _, png = cv2.imencode(".png", final_img)
    return {
        "found": True,
        "digit": int(np.argmax(probabilities)),
        "confidence": float(np.max(probabilities)),
        "probabilities": [round(float(p), 4) for p in probabilities],
        "preprocessed": "data:image/png;base64," + base64.b64encode(png).decode(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        self._send(200, "text/html; charset=utf-8", INDEX_HTML.read_bytes())

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self._json(413, {"error": "Send an image under 4 MB"})
            return
        try:
            body = json.loads(self.rfile.read(length))
            # Browsers send images as data URLs: "data:image/png;base64,...."
            image_bytes = base64.b64decode(body["image"].split(",", 1)[-1])
            self._json(200, run_prediction(image_bytes, body.get("script", "western")))
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self._json(400, {"error": str(e)})
        except cv2.error:
            self._json(400, {"error": "That image is too large or damaged to read"})

    def _json(self, status, payload):
        self._send(status, "application/json", json.dumps(payload).encode())

    def _send(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # the camera posts several frames a second; keep the console quiet


def main():
    print(f"Numeral Recogniser UI running at http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
