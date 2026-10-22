"""Synthetic handwriting-like digit photos rendered from fonts.

There is no open dataset of handwritten Malayalam digits, and the Devanagari
set covers few writing styles, so we render digits from many fonts and distort
them the way handwriting and phone photos vary: stroke weight, elastic wobble,
slant, rotation, uneven paper, blur and noise. Images come out as "photos"
(ink on paper), so they go through the same preprocess() as real uploads."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "fonts"
DIGITS = {
    "devanagari": "०१२३४५६७८९",
    "malayalam": "൦൧൨൩൪൫൬൭൮൯",
}
# Fonts never used for training, so tests measure shapes the model has not seen.
# Each list holds one handwriting-style font plus regular ones.
# (Hind, Khand, Rozha One and Sarpanch were dropped from datasets/fonts: they
# draw Western-looking shapes for the Devanagari digit code points.)
HELD_OUT = {
    "devanagari": ("Kalam", "Amita", "Mukta"),
    "malayalam": ("Karumbi", "Meera", "AnekMalayalam"),
}


def font_files(script, held_out):
    """Training fonts (held_out=False) or test fonts (held_out=True)."""
    files = sorted((FONTS_DIR / script).glob("*.ttf"))
    is_held = lambda f: f.stem.split("-")[0] in HELD_OUT[script]
    return [f for f in files if is_held(f) == held_out]


def render_photo(font_path, char, rng):
    """One distorted 'photo' of a digit: dark ink on light paper (sometimes inverted)."""
    mask = _glyph_mask(font_path, char, rng)
    mask = _distort(mask, rng)
    return _onto_paper(mask, rng)


def _glyph_mask(font_path, char, rng, size=160):
    font = ImageFont.truetype(str(font_path), size)
    try:  # variable fonts: pick a random weight
        axes = font.get_variation_axes()
        font.set_variation_by_axes([rng.uniform(a["minimum"], a["maximum"]) for a in axes])
    except OSError:
        pass
    canvas = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(canvas).text((size, size), char, font=font, fill=255, anchor="mm")
    mask = np.array(canvas)
    ys, xs = np.nonzero(mask > 40)
    pad = size // 4
    return mask[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def _distort(mask, rng):
    h, w = mask.shape
    # Stroke weight: pens and markers vary more than font weights do.
    # Thinning stays mild; heavy erosion snaps strokes into pieces.
    if rng.random() < 0.7:
        k = int(rng.integers(1, 9))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    else:
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8))

    # Elastic wobble: a smooth random displacement field, like an unsteady hand
    alpha, sigma = rng.uniform(4, 14) * max(h, w) / 160, max(h, w) / rng.uniform(8, 14)
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha / 0.05
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha / 0.05
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mask = cv2.remap(mask, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)

    # Slant, rotation and uneven scaling
    angle, shear = rng.uniform(-15, 15), rng.uniform(-0.3, 0.3)
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    affine = rot @ np.array([[sx, shear, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    affine[:, 2] += (np.array([w / 2, h / 2]) - affine[:, :2] @ np.array([w / 2, h / 2]))
    return cv2.warpAffine(mask, affine, (w, h), borderValue=0)


def _onto_paper(mask, rng):
    h, w = mask.shape
    m = mask.astype(np.float32) / 255
    paper = rng.uniform(150, 245)
    shade = rng.uniform(0, 45) * np.linspace(0, 1, w)[None, :]  # light falling across the page
    if rng.random() < 0.5:
        shade = shade.T[:h, :w] if h <= w else rng.uniform(0, 45) * np.linspace(0, 1, h)[:, None]
    bg = paper - shade
    ink = rng.uniform(10, 90)
    photo = bg * (1 - m) + ink * m + rng.normal(0, rng.uniform(1, 8), (h, w))
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.5, 1.8))
    if rng.random() < 0.15:  # chalk on a board / white pen on black paper
        photo = 255 - photo
    return photo
