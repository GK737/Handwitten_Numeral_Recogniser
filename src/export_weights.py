"""Export the trained Keras models to small .npz files that src/inference.py
can run with NumPy alone (no TensorFlow), which is what the deployed site uses.

Run again after retraining:  .venv/Scripts/python src/export_weights.py"""
import keras
import numpy as np

from preprocess import DEVANAGARI_MODEL_PATH, MNIST_MODEL_PATH, MODELS_DIR

# Layers that only act during training (augmentation, dropout)
TRAINING_ONLY = ("RandomRotation", "RandomTranslation", "RandomZoom", "Dropout")


def flat_layers(model):
    for layer in model.layers:
        if isinstance(layer, keras.Sequential):
            yield from flat_layers(layer)
        else:
            yield layer


def export(model_path, out_path):
    model = keras.models.load_model(model_path)
    arrays, kinds = {}, []
    for layer in flat_layers(model):
        kind = type(layer).__name__
        if kind in TRAINING_ONLY:
            continue
        i = len(kinds)
        cfg = layer.get_config()
        if kind == "Conv2D":
            assert cfg["padding"] == "same" and cfg["strides"] == (1, 1)
            kernel, bias = layer.get_weights()
            arrays[f"{i}_kernel"], arrays[f"{i}_bias"] = kernel, bias
            kinds.append(f"conv:{cfg['activation']}")
        elif kind == "BatchNormalization":
            gamma, beta, mean, var = layer.get_weights()
            # Fold into one multiply-add: y = x * scale + shift
            scale = gamma / np.sqrt(var + cfg["epsilon"])
            arrays[f"{i}_scale"], arrays[f"{i}_shift"] = scale, beta - mean * scale
            kinds.append("batchnorm")
        elif kind == "MaxPooling2D":
            assert cfg["pool_size"] == (2, 2) and cfg["padding"] == "valid"
            kinds.append("maxpool")
        elif kind == "Flatten":
            kinds.append("flatten")
        elif kind == "Dense":
            kernel, bias = layer.get_weights()
            arrays[f"{i}_kernel"], arrays[f"{i}_bias"] = kernel, bias
            kinds.append(f"dense:{cfg['activation']}")
        else:
            raise ValueError(f"{model_path.name}: no NumPy version of layer {kind}")
    np.savez_compressed(out_path, kinds=np.array(kinds), **arrays)
    print(f"Wrote {out_path} ({len(kinds)} layers)")


if __name__ == "__main__":
    export(MNIST_MODEL_PATH, MODELS_DIR / "western.npz")
    export(DEVANAGARI_MODEL_PATH, MODELS_DIR / "devanagari.npz")
