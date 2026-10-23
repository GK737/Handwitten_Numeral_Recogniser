from pathlib import Path

import cv2
import keras
import numpy as np
import pytest

import inference
from preprocess import DEVANAGARI_MODEL_PATH, MALAYALAM_MODEL_PATH, MNIST_MODEL_PATH, preprocess

REPO = Path(__file__).resolve().parent.parent


def sample_inputs(script, western_photos):
    if script == "western":
        return np.concatenate([preprocess(img, script)[0] for img, _ in western_photos[::15]])
    if script == "devanagari":
        files = [sorted((REPO / "datasets/Devanagiri_Dataset/Test" / f"digit_{d}").glob("*.png"))[0]
                 for d in range(10)]
    else:
        files = [sorted((REPO / "datasets/Malayalam_Dataset/malayalam_digits/test" / str(d)).glob("*.png"))[0]
                 for d in range(10)]
    return np.concatenate([preprocess(cv2.imread(str(f)), script)[0] for f in files])


@pytest.mark.parametrize("script, keras_path", [
    ("western", MNIST_MODEL_PATH),
    ("devanagari", DEVANAGARI_MODEL_PATH),
    ("malayalam", MALAYALAM_MODEL_PATH),
])
def test_exported_model_matches_keras(script, keras_path, western_photos):
    x = sample_inputs(script, western_photos)
    expected = keras.models.load_model(keras_path).predict(x, verbose=0)
    actual = np.concatenate([inference.load_model(script).predict(x[i:i+1]) for i in range(len(x))])
    np.testing.assert_allclose(actual, expected, atol=1e-4)
