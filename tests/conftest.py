import cv2
import numpy as np
import pytest

from synth import FONTS_DIR, render_photo

# Handwriting-style fonts that include Western digits
WESTERN_FONTS = ["Kalam-Regular.ttf", "Kalam-Light.ttf", "Amita-Regular.ttf"]


@pytest.fixture(scope="session")
def western_photos():
    """Synthetic phone photos of handwriting-style Western digits: dark ink on
    uneven grey paper. The original adaptive-threshold pipeline got 17% of these
    right, the fixed one 98.7%. Returns [(BGR image, digit)]."""
    rng = np.random.default_rng(42)
    return [(cv2.cvtColor(render_photo(FONTS_DIR / "devanagari" / font, str(d), rng), cv2.COLOR_GRAY2BGR), d)
            for font in WESTERN_FONTS for d in range(10) for _ in range(5)]


@pytest.fixture(scope="session")
def photo_of_seven(western_photos):
    image = next(img for img, d in western_photos if d == 7)
    return cv2.imencode(".png", image)[1].tobytes()
