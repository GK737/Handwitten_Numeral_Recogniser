"""Train the Devanagari or Malayalam model on real scans plus synthetic
font-based photos, all passed through preprocess() so training sees exactly
what the app sees at prediction time.

    .venv/Scripts/python src/train_script.py malayalam
    .venv/Scripts/python src/train_script.py devanagari
    .venv/Scripts/python src/export_weights.py      # then refresh models/*.npz

Data:
- Devanagari: datasets/Devanagiri_Dataset/Train (handwritten, 1700 per digit)
- Malayalam: datasets/Malayalam_Dataset (CC BY 4.0, anand-venugopal on
  Hugging Face; one typed glyph per digit, rotated), so most variety comes
  from the synthetic photos
- Both: synthetic photos from datasets/fonts (see src/synth.py). Fonts in
  synth.HELD_OUT are never trained on; tests/test_scripts.py measures them."""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from preprocess import DEVANAGARI_MODEL_PATH, MALAYALAM_MODEL_PATH, preprocess
from synth import DIGITS, font_files, render_photo

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "datasets" / "cache"
MODEL_PATHS = {"devanagari": DEVANAGARI_MODEL_PATH, "malayalam": MALAYALAM_MODEL_PATH}
# Synthetic photos per training font per digit
SYNTH_PER_FONT_DIGIT = {"devanagari": 150, "malayalam": 350}


def clean_malayalam_scan(gray):
    """The Malayalam dataset images are a white square rotated on black, so
    black corners surround the paper. Paint everything outside the paper white
    and upscale, so preprocess() sees an ordinary photo of a digit."""
    paper = (gray > 128).astype(np.uint8)
    hull = cv2.convexHull(cv2.findNonZero(paper))
    inside = np.zeros_like(paper)
    cv2.fillConvexPoly(inside, hull, 1)
    # Pull in from the edge: the rotated paper's border is a grey line that
    # would otherwise survive as a stray stroke
    inside = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=2)
    out = np.where(inside == 1, gray, 255).astype(np.uint8)
    return cv2.resize(out, (112, 112), interpolation=cv2.INTER_CUBIC)


def _single_threaded():
    # Each pool worker is one process; OpenCV's own thread pool on top of
    # that makes the workers fight over the cores
    cv2.setNumThreads(1)


def _framed(args):
    image, script = args
    return preprocess(image, script)[1]


def _synthetic(args):
    font, char, script, seed = args
    rng = np.random.default_rng(seed)
    return preprocess(render_photo(font, char, rng), script)[1]


def real_images(script):
    if script == "devanagari":
        root = REPO / "datasets/Devanagiri_Dataset/Train"
        for digit in range(10):
            for f in sorted((root / f"digit_{digit}").glob("*.png")):
                yield cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), digit
    else:
        root = REPO / "datasets/Malayalam_Dataset/malayalam_digits/train"
        for digit in range(10):
            for f in sorted((root / str(digit)).glob("*.png")):
                yield clean_malayalam_scan(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)), digit


def build_dataset(script):
    cache = CACHE_DIR / f"{script}.npz"
    if cache.exists():
        with np.load(cache) as d:
            return d["x"], d["y"]
    real = list(real_images(script))
    jobs = [(font, char, script, hash((font.name, digit, i)) % 2**32)
            for font in font_files(script, held_out=False)
            for digit, char in enumerate(DIGITS[script])
            for i in range(SYNTH_PER_FONT_DIGIT[script])]
    labels = [d for _, d in real] + [DIGITS[script].index(j[1]) for j in jobs]
    with Pool(initializer=_single_threaded) as pool:
        frames = pool.map(_framed, [(img, script) for img, _ in real], chunksize=64)
        frames += pool.map(_synthetic, jobs, chunksize=64)
    keep = [i for i, f in enumerate(frames) if f is not None]
    x = np.stack([frames[i] for i in keep])[..., None]
    y = np.array([labels[i] for i in keep])
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    print(f"{script}: {len(real)} real + {len(jobs)} synthetic, {len(frames) - len(keep)} had no digit")
    return x, y


def make_model():
    import keras
    from keras import layers
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])


def train(script, epochs=15):
    import keras
    x, y = build_dataset(script)
    x = x.astype("float32") / 255.0
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x))
    split = int(len(x) * 0.9)
    train_idx, val_idx = order[:split], order[split:]

    model = make_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x[train_idx], y[train_idx], validation_data=(x[val_idx], y[val_idx]),
              epochs=epochs, batch_size=128, verbose=2,
              callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
                         keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    model.save(MODEL_PATHS[script])
    print(f"Saved {MODEL_PATHS[script]}")


if __name__ == "__main__":
    train(sys.argv[1])
