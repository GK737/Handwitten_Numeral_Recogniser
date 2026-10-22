# Handwitten_Numeral_Recogniser
Handwritten numeral recognition is the ability of a computer to identify and classify human numerical digits (Generally 0 through 9) from images or scanned documents. This project deals with the real life problem of now implementing the same for regional scripts. Now work in progrss 

## Setup

```
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Test it in the browser

```
.venv/Scripts/python src/web_app.py
```

Then open http://127.0.0.1:8765. You can upload a photo, use the webcam or draw a digit, for Western or Devanagari numerals. The page shows the prediction, the confidence and the 28x28 image the model actually sees.

## Command line

- `src/predict.py` predicts from an image file (add `--debug` to save the preprocessed image to `debug/`)
- `src/live_predict.py` predicts live from the webcam

## Tests

```
.venv/Scripts/python -m pytest
```
