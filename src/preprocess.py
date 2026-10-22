from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MNIST_MODEL_PATH = MODELS_DIR / "mnist_cnn.keras"
DEVANAGARI_MODEL_PATH = MODELS_DIR / "devanagari_cnn.keras"
MALAYALAM_MODEL_PATH = MODELS_DIR / "malayalam_cnn.keras"

# Pieces smaller than this fraction of the largest blob are treated as noise
MIN_PIECE_RATIO = 0.15
# Larger inputs are shrunk to this longest side before any processing
MAX_SIDE = 512

# How each training set framed its digits in 28x28:
# (box size in pixels, keep aspect ratio, centre by mass instead of by box).
# MNIST fits digits in 20x20 keeping their shape and centres by mass; the
# Devanagari scans stretch every digit to fill 28 of 32 pixels both ways
# (24 of 28 once resized), so a wide digit must be stretched to square too.
FRAMING = {
    "western": (20, True, True),
    "devanagari": (24, False, False),
    # Trained on this pipeline's own output (src/train_script.py). Malayalam
    # digits vary a lot in width (൦ vs ൬), so their shape is kept.
    "malayalam": (22, True, False),
}


def preprocess(image, script="western"):
    """Turn a photo of one digit (dark-on-light or light-on-dark) into the
    28x28 white-on-black tensor the given script's model was trained on.
    Returns (tensor, 28x28 uint8 image), or (None, None) if no digit is found."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Phone photos can be 4000px; the digit only needs to end up 28px
    scale = MAX_SIDE / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Polarity: the border is background, and ink is whichever extreme sits
    #    furthest from it. Flip so the ink is always the dark part below.
    if _ink_is_light(gray):
        gray = 255 - gray

    # 2. Flatten lighting: estimate the paper (ink closed away) and divide it out,
    #    so shadows and folds don't survive thresholding. The window must be
    #    wider than the thickest stroke, or the stroke leaks into the estimate.
    k = max(15, (min(gray.shape) // 3) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    flat = cv2.divide(gray, paper, scale=255)

    # 3. Otsu picks the ink/paper cutoff from the image itself; INV makes ink white
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    digit = _keep_digit(mask)
    if digit is None:
        return None, None
    final_img = _to_frame(digit, *FRAMING[script])
    tensor = np.expand_dims(final_img.astype("float32") / 255.0, axis=(0, -1))
    return tensor, final_img


def _ink_is_light(gray):
    paper = np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]))
    darkest, brightest = np.percentile(gray, [0.5, 99.5])
    return brightest - paper > paper - darkest


def _keep_digit(mask):
    """Keep the largest blob plus any sizeable pieces of the same digit
    (a 4 or a Devanagari numeral can be drawn in more than one stroke)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    h, w = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = areas.max()
    if largest < 0.0005 * h * w:
        return None
    keep = [i + 1 for i in range(areas.size) if areas[i] >= MIN_PIECE_RATIO * largest]
    digit = np.isin(labels, keep).astype(np.uint8) * 255
    x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(digit))
    return digit[y:y+bh, x:x+bw]


def _to_frame(digit, box, keep_aspect, centre_by_mass):
    """Fit into a box x box square in the middle of a 28x28 frame."""
    h, w = digit.shape
    # Thin pen strokes vanish when shrunk; thicken them relative to digit size
    stroke = max(1, max(h, w) // 40)
    digit = cv2.dilate(digit, np.ones((stroke, stroke), np.uint8))
    if keep_aspect:
        scale = box / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    else:
        new_w = new_h = box
    small = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_img = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    final_img[oy:oy+new_h, ox:ox+new_w] = small

    m = cv2.moments(final_img)
    if centre_by_mass and m["m00"] > 0:
        dx = round(14 - m["m10"] / m["m00"])
        dy = round(14 - m["m01"] / m["m00"])
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        final_img = cv2.warpAffine(final_img, shift, (28, 28))
    return final_img
