"""Local test UI: serves web/index.html and a /api/predict endpoint.

Run:  .venv/Scripts/python src/web_app.py   then open http://127.0.0.1:8765
Only listens on this machine (127.0.0.1); it is a testing tool, not a server
to expose on a network."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import keras
import numpy as np

from preprocess import DEVANAGARI_MODEL_PATH, FRAMING, MNIST_MODEL_PATH, preprocess

HOST, PORT = "127.0.0.1", 8765
INDEX_HTML = Path(__file__).resolve().parent.parent / "web" / "index.html"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MODEL_PATHS = {"western": MNIST_MODEL_PATH, "devanagari": DEVANAGARI_MODEL_PATH}

_models = {}
# Keras models are not safe to call from several request threads at once
_model_lock = threading.Lock()


def _model(script):
    if script not in _models:
        _models[script] = keras.models.load_model(MODEL_PATHS[script])
    return _models[script]


def run_prediction(image_bytes, script):
    if script not in FRAMING:
        raise ValueError(f"Unknown script {script!r}")
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read that file as an image")

    tensor, final_img = preprocess(image, script)
    if tensor is None:
        return {"found": False}

    with _model_lock:
        probabilities = _model(script).predict(tensor, verbose=0)[0]
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
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self._json(413, {"error": "Image must be under 15 MB"})
            return
        try:
            body = json.loads(self.rfile.read(length))
            # Browsers send images as data URLs: "data:image/png;base64,...."
            image_bytes = base64.b64decode(body["image"].split(",", 1)[-1])
            self._json(200, run_prediction(image_bytes, body.get("script", "western")))
        except (ValueError, KeyError, TypeError) as e:
            self._json(400, {"error": str(e)})

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
    for script in MODEL_PATHS:
        _model(script)
    print(f"Numeral Recogniser UI running at http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
