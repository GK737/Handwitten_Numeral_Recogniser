from pathlib import Path

import cv2
import keras
import numpy as np
import pytest

import inference
from preprocess import DEVANAGARI_MODEL_PATH, MNIST_MODEL_PATH, preprocess

REPO = Path(__file__).resolve().parent.parent


def sample_inputs(script):
    if script == "western":
        files = sorted((REPO / "IRL_TEST").glob("digit_*_yadu.png"))
    else:
        files = [sorted((REPO / "datasets/Devanagiri_Dataset/Test" / f"digit_{d}").glob("*.png"))[0]
                 for d in range(10)]
    return np.concatenate([preprocess(cv2.imread(str(f)), script)[0] for f in files])


@pytest.mark.parametrize("script, keras_path", [
    ("western", MNIST_MODEL_PATH),
    ("devanagari", DEVANAGARI_MODEL_PATH),
])
def test_exported_model_matches_keras(script, keras_path):
    x = sample_inputs(script)
    expected = keras.models.load_model(keras_path).predict(x, verbose=0)
    actual = np.concatenate([inference.load_model(script).predict(x[i:i+1]) for i in range(len(x))])
    np.testing.assert_allclose(actual, expected, atol=1e-4)
