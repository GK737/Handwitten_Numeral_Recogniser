import argparse
import os
from pathlib import Path

import cv2
import keras
import numpy as np

from preprocess import DEVANAGARI_MODEL_PATH, MNIST_MODEL_PATH, preprocess

DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"


def save_preprocessed_image(image, image_path):
    DEBUG_DIR.mkdir(exist_ok=True)
    output_path = DEBUG_DIR / (Path(image_path).stem + "_preprocessed.png")
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not save preprocessed image to {output_path}")
    print(f"Saved preprocessed image to {output_path}")


def predict_digit(image_path, model_path, script, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    processed_image, final_img = preprocess(image, script)
    if processed_image is None:
        return None, 0.0
    if debug:
        save_preprocessed_image(final_img, image_path)
    model = keras.models.load_model(model_path)
    prediction = model.predict(processed_image, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="save each preprocessed 28x28 image to debug/")
    args = parser.parse_args()

    while True:
        print("\n--- Numeral Recogniser Menu ---")
        print("1. Predict Western Digit (MNIST)")
        print("2. Predict Devanagari Digit")
        print("3. Exit")

        choice = input("Select an option: ")

        if choice == '3':
            print("Exiting...")
            break

        if choice not in ['1', '2']:
            print("Invalid choice, please try again.")
            continue

        image_path = input("Enter the path to the image file: ").strip('"')

        if not os.path.exists(image_path):
            print("File not found!")
            continue

        try:
            if choice == '1':
                model_path, script = MNIST_MODEL_PATH, "western"
            else:
                model_path, script = DEVANAGARI_MODEL_PATH, "devanagari"
            digit, confidence = predict_digit(image_path, model_path, script, args.debug)

            if digit is None:
                print("\nNo digit found in the image.")
            else:
                print(f"\nPredicted digit: {digit}")
                print(f"Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
