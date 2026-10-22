# Handwitten_Numeral_Recogniser
Handwritten numeral recognition is the ability of a computer to identify and classify human numerical digits (Generally 0 through 9) from images or scanned documents. This project deals with the real life problem of now implementing the same for regional scripts. Now work in progrss 

## Setup

```
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

## Test it in the browser

```
.venv/Scripts/python src/web_app.py
```

Then open http://127.0.0.1:8765. The same page is deployed on Vercel (see Deploy below). You can upload a photo, use the webcam or draw a digit, for Western, Devanagari or Malayalam numerals. The page shows the prediction, the confidence and the 28x28 image the model actually sees.

## Command line

- `src/predict.py` predicts from an image file (add `--debug` to save the preprocessed image to `debug/`)
- `src/live_predict.py` predicts live from the webcam

## Tests

```
.venv/Scripts/python -m pytest
```

## Deploy

The site runs on Vercel as one Python function: `app.py` exposes the WSGI app from `src/web_app.py`, which serves `public/index.html` and `/api/predict`. It uses `src/inference.py`, which runs the models with NumPy alone, so the function doesn't need TensorFlow. After retraining, re-export the weights before deploying:

```
.venv/Scripts/python src/export_weights.py
vercel deploy
```

## Training the Devanagari and Malayalam models

```
.venv/Scripts/python src/train_script.py devanagari
.venv/Scripts/python src/train_script.py malayalam
.venv/Scripts/python src/export_weights.py
```

There's no open dataset of handwritten Malayalam digits, so training mixes real images with synthetic "photos" rendered from many fonts and then warped like handwriting: stroke weight, wobble, slant, rotation, uneven paper and noise (`src/synth.py`). A few fonts are never used for training, and `tests/test_scripts.py` checks accuracy on them, so the score reflects shapes the model hasn't seen.

## Data sources and credits

- **MNIST**: Yann LeCun, Corinna Cortes and Christopher J.C. Burges.
- **Devanagari**: the digit classes of the Devanagari Handwritten Character Dataset (UCI).
- **Malayalam**: [Malayalam Digit Dataset](https://huggingface.co/datasets/anand-venugopal/malayalam-digits-dataset) by Anand Venugopal, licensed CC BY 4.0. Used as-is apart from cleaning the rotated borders at training time.
- **Fonts** (`datasets/fonts/`), all under the SIL Open Font License: Google Fonts (Chilanka, Manjari, Gayathri, Baloo Chettan 2, Noto Sans/Serif Malayalam, Anek Malayalam, Kalam, Amita, Yatra One, Sahitya, Noto Sans Devanagari, Tiro Devanagari Hindi, Mukta, Martel, Baloo 2, Karma, Laila, Biryani, Eczar, Palanquin, Kurale) and Swathanthra Malayalam Computing (Karumbi, Uroob, Dyuthi, Rachana, Meera).
