"""Accuracy of the Devanagari and Malayalam models, through the same NumPy
models and preprocessing the deployed site uses."""
from pathlib import Path

import cv2
import numpy as np
import pytest

import inference
from preprocess import preprocess
from synth import DIGITS, font_files, render_photo

REPO = Path(__file__).resolve().parent.parent
MALAYALAM_TEST = REPO / "datasets/Malayalam_Dataset/malayalam_digits/test"


def accuracy(script, images, labels):
    model = inference.load_model(script)
    preds = []
    for image in images:
        tensor, _ = preprocess(image, script)
        preds.append(-1 if tensor is None else int(np.argmax(model.predict(tensor))))
    return np.mean(np.array(preds) == np.array(labels))


def held_out_font_photos(script, per_font_digit=20):
    """Distorted photos of digits in fonts the model never trained on."""
    rng = np.random.default_rng(1234)
    images, labels = [], []
    for font in font_files(script, held_out=True):
        for digit, char in enumerate(DIGITS[script]):
            for _ in range(per_font_digit):
                images.append(render_photo(font, char, rng))
                labels.append(digit)
    return images, labels


@pytest.mark.parametrize("script", ["devanagari", "malayalam"])
def test_digits_in_unseen_fonts_are_recognised(script):
    images, labels = held_out_font_photos(script)
    acc = accuracy(script, images, labels)
    assert acc >= 0.90, f"{script} held-out fonts: {acc:.3f}"


def test_devanagari_handwritten_test_set_does_not_regress():
    images, labels = [], []
    for digit in range(10):
        files = sorted((REPO / "datasets/Devanagiri_Dataset/Test" / f"digit_{digit}").glob("*.png"))
        images += [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files[::3]]
        labels += [digit] * len(files[::3])
    acc = accuracy("devanagari", images, labels)
    assert acc >= 0.97, f"Devanagari handwritten test set: {acc:.3f}"


def test_malayalam_dataset_test_split_is_recognised():
    from train_script import clean_malayalam_scan
    images, labels = [], []
    for digit in range(10):
        for f in sorted((MALAYALAM_TEST / str(digit)).glob("*.png")):
            images.append(clean_malayalam_scan(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)))
            labels.append(digit)
    acc = accuracy("malayalam", images, labels)
    assert acc >= 0.95, f"Malayalam dataset test split: {acc:.3f}"
