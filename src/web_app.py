"""Test UI backend: serves public/index.html and a /api/predict endpoint.

Locally:  .venv/Scripts/python src/web_app.py   then open http://127.0.0.1:8765
On Vercel, the root app.py exposes the same WSGI `app`. Needs only NumPy and
OpenCV (models run through src/inference.py)."""
import base64
import json
import os
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

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


def app(environ, start_response):
    """WSGI app: the same object runs locally (main below) and on Vercel (app.py)."""
    method, path = environ["REQUEST_METHOD"], environ.get("PATH_INFO", "/")
    if method == "GET" and path in ("/", "/index.html"):
        return _send(start_response, "200 OK", "text/html; charset=utf-8", INDEX_HTML.read_bytes())
    if method == "POST" and path == "/api/predict":
        status, payload = _predict_request(environ)
        return _send(start_response, status, "application/json", json.dumps(payload).encode())
    return _send(start_response, "404 Not Found", "text/plain", b"Not found")


def _predict_request(environ):
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        length = 0
    if not 0 < length <= MAX_UPLOAD_BYTES:
        return "413 Content Too Large", {"error": "Send an image under 4 MB"}
    try:
        body = json.loads(environ["wsgi.input"].read(length))
        # Browsers send images as data URLs: "data:image/png;base64,...."
        image_bytes = base64.b64decode(body["image"].split(",", 1)[-1])
        return "200 OK", run_prediction(image_bytes, body.get("script", "western"))
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return "400 Bad Request", {"error": str(e)}
    except cv2.error:
        return "400 Bad Request", {"error": "That image is too large or damaged to read"}


def _send(start_response, status, content_type, data):
    start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(data)))])
    return [data]


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True  # the camera sends several frames a second


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, fmt, *args):
        pass


def main():
    server = make_server(HOST, PORT, app, _ThreadingWSGIServer, _QuietHandler)
    print(f"Numeral Recogniser UI running at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
