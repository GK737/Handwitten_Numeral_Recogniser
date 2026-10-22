import subprocess
import sys
from pathlib import Path

import pytest

from web_app import run_prediction

REPO = Path(__file__).resolve().parent.parent


def test_uploaded_photo_returns_digit_probabilities_and_model_input():
    photo = (REPO / "IRL_TEST" / "digit_7_yadu.png").read_bytes()
    result = run_prediction(photo, "western")
    assert result["found"] is True
    assert result["digit"] == 7
    assert len(result["probabilities"]) == 10
    assert result["preprocessed"].startswith("data:image/png;base64,")


def test_blank_image_reports_no_digit():
    import cv2
    import numpy as np
    _, png = cv2.imencode(".png", np.full((200, 200), 230, np.uint8))
    assert run_prediction(png.tobytes(), "western")["found"] is False


def run_fresh(code):
    # A fresh interpreter, so imports and cv2 settings start the way they do on Vercel
    return subprocess.run([sys.executable, "-c", code], cwd=REPO / "src",
                          capture_output=True, text=True, timeout=120)


def test_server_code_does_not_load_tensorflow():
    # TensorFlow is far too big for the Vercel function
    r = run_fresh("import sys, web_app; print('tensorflow' in sys.modules or 'keras' in sys.modules)")
    assert r.stdout.strip() == "False", r.stderr


def test_image_claiming_huge_dimensions_is_refused():
    # 8000x6000 blank PNG: a few KB on the wire, ~144 MB once decoded
    r = run_fresh(
        "import web_app, cv2, numpy as np\n"
        "_, png = cv2.imencode('.png', np.zeros((6000, 8000), np.uint8))\n"
        "try:\n"
        "    web_app.run_prediction(png.tobytes(), 'western'); print('decoded')\n"
        "except (ValueError, cv2.error):\n"
        "    print('refused')\n")
    assert r.stdout.strip() == "refused", r.stderr


def test_unknown_script_and_garbage_bytes_are_rejected():
    photo = (REPO / "IRL_TEST" / "digit_7_yadu.png").read_bytes()
    with pytest.raises(ValueError):
        run_prediction(photo, "klingon")
    with pytest.raises(ValueError):
        run_prediction(b"not an image", "western")
