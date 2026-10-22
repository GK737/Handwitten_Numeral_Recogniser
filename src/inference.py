"""Run the exported models (models/*.npz, made by src/export_weights.py) with
NumPy only. Gives the same probabilities as Keras, without TensorFlow, so the
deployed site stays small enough for a serverless function."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from preprocess import MODELS_DIR

_cache = {}


def load_model(script):
    if script not in _cache:
        _cache[script] = NumpyModel(MODELS_DIR / f"{script}.npz")
    return _cache[script]


class NumpyModel:
    def __init__(self, path):
        with np.load(path) as data:
            self.kinds = [str(k) for k in data["kinds"]]
            self.arrays = {k: data[k] for k in data.files if k != "kinds"}

    def predict(self, x):
        """x: (1, 28, 28, 1) float32 -> (1, 10) probabilities."""
        out = x[0].astype(np.float32)
        for i, kind in enumerate(self.kinds):
            name, _, activation = kind.partition(":")
            if name == "conv":
                out = _conv3x3_same(out, self.arrays[f"{i}_kernel"]) + self.arrays[f"{i}_bias"]
            elif name == "batchnorm":
                out = out * self.arrays[f"{i}_scale"] + self.arrays[f"{i}_shift"]
            elif name == "maxpool":
                h, w, c = out.shape
                out = out[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))
            elif name == "flatten":
                out = out.reshape(-1)
            elif name == "dense":
                out = out @ self.arrays[f"{i}_kernel"] + self.arrays[f"{i}_bias"]
            out = _activate(out, activation)
        return out[None, :]


def _conv3x3_same(x, kernel):
    # x: (H, W, Cin), kernel: (3, 3, Cin, Cout) as Keras stores it
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    windows = sliding_window_view(padded, (3, 3), axis=(0, 1))  # (H, W, Cin, 3, 3)
    return np.einsum("hwcij,ijco->hwo", windows, kernel, optimize=True)


def _activate(x, activation):
    if activation == "relu":
        return np.maximum(x, 0)
    if activation == "softmax":
        e = np.exp(x - x.max())
        return e / e.sum()
    return x
