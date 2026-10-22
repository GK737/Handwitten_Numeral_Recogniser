import base64
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


def test_unknown_script_and_garbage_bytes_are_rejected():
    photo = (REPO / "IRL_TEST" / "digit_7_yadu.png").read_bytes()
    with pytest.raises(ValueError):
        run_prediction(photo, "klingon")
    with pytest.raises(ValueError):
        run_prediction(b"not an image", "western")
