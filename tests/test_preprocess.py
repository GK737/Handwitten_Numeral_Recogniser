from pathlib import Path

import cv2
import keras
import numpy as np
import pytest

from preprocess import DEVANAGARI_MODEL_PATH, MNIST_MODEL_PATH, preprocess

REPO = Path(__file__).resolve().parent.parent
IRL_PHOTOS = sorted(p for p in (REPO / "IRL_TEST").glob("digit_*.png")
                    if "_preprocessed" not in p.name)


def label_of(path):
    # digit_4_yadu.png -> 4
    return int(path.name.split("_")[1])


@pytest.fixture(scope="module")
def mnist_model():
    return keras.models.load_model(MNIST_MODEL_PATH)


def predict(model, image):
    tensor, _ = preprocess(image)
    assert tensor is not None, "no digit found"
    return int(np.argmax(model.predict(tensor, verbose=0)))


def test_real_photos_of_dark_digits_on_paper_are_recognised(mnist_model):
    assert len(IRL_PHOTOS) == 10
    correct = [p.name for p in IRL_PHOTOS
               if predict(mnist_model, cv2.imread(str(p))) == label_of(p)]
    assert len(correct) >= 8, f"only recognised {correct}"


def test_light_ink_on_dark_background_reads_the_same_as_dark_on_light(mnist_model):
    differing = []
    for p in IRL_PHOTOS:
        photo = cv2.imread(str(p))
        if predict(mnist_model, 255 - photo) != predict(mnist_model, photo):
            differing.append(p.name)
    assert differing == []


def fake_paper_photo(scan, rng):
    """Dataset digits are clean white-on-black scans; make one look like a phone
    photo of dark ink on grey paper: upscaled, uneven light, sensor noise."""
    big = cv2.resize(scan, (224, 224), interpolation=cv2.INTER_CUBIC).astype("float32") / 255
    paper = 185 - 40 * np.linspace(0, 1, 224)[None, :]  # shadow across the page
    photo = paper * (1 - big) + 55 * big + rng.normal(0, 6, big.shape)
    return np.clip(photo, 0, 255).astype(np.uint8)


def test_devanagari_digits_photographed_on_paper_are_recognised():
    model = keras.models.load_model(DEVANAGARI_MODEL_PATH)
    rng = np.random.default_rng(0)
    scans, labels = [], []
    for digit in range(10):
        files = sorted((REPO / "datasets/Devanagiri_Dataset/Test" / f"digit_{digit}").glob("*.png"))
        for f in files[::10]:  # 30 per digit keeps the test quick
            scans.append(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
            labels.append(digit)
    labels = np.array(labels)

    # Baseline: the scans fed the way the model saw them in training
    clean = np.stack([cv2.resize(s, (28, 28), interpolation=cv2.INTER_AREA) for s in scans])
    clean = np.expand_dims(clean.astype("float32") / 255.0, -1)
    clean_acc = np.mean(model.predict(clean, verbose=0).argmax(1) == labels)

    photos = [preprocess(fake_paper_photo(s, rng), script="devanagari")[0] for s in scans]
    assert all(p is not None for p in photos)
    photo_acc = np.mean(model.predict(np.concatenate(photos), verbose=0).argmax(1) == labels)

    assert photo_acc >= clean_acc - 0.10, f"clean {clean_acc:.2f}, photo {photo_acc:.2f}"
