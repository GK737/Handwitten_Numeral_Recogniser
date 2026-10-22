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

Then open http://127.0.0.1:8765. The same page is deployed on Vercel (see Deploy below). You can upload a photo, use the webcam or draw a digit, for Western or Devanagari numerals. The page shows the prediction, the confidence and the 28x28 image the model actually sees.

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
