from PIL import Image, ImageOps
import numpy as np
import keras
import cv2
import os

MNIST_MODEL_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\mnist_cnn.keras"
DEVANAGARI_MODEL_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\devanagari_cnn.keras"

def save_preprocessed_image(image, image_path):
    output_path = os.path.splitext(image_path)[0] + "_preprocessed.png"
    if not cv2.imwrite(output_path, image):
        raise IOError(f"Could not save preprocessed image to {output_path}")

def preprocess_image(image_path):
    # 1. Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # 2. Contrast Enhancement
    # Makes dark ink darker and white paper brighter
    img = cv2.equalizeHist(img)

    # 3. Noise Reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 4. Adaptive Thresholding (Inverted)
    # Converts Dark-on-Light to Light-on-Dark (required for MNIST models)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 5)

    # 5. Find Bounding Box
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # Digit Validation (Anti-Blob/Face)
        roi_crop = img[y:y+h, x:x+w]
        white_pixels = cv2.countNonZero(roi_crop)
        total_pixels = w * h
        fill_ratio = white_pixels / total_pixels

        if fill_ratio > 0.50 or fill_ratio < 0.01:
            # We can't return None in predict_image because the main loop expects an array.
            # We'll return a blank image if it's not a valid digit.
            final_img = np.zeros((28, 28), dtype=np.uint8)
            save_preprocessed_image(final_img, image_path)
            img_array = final_img.astype("float32") / 255.0
            return np.expand_dims(img_array, axis=(0, -1))

        pad = 10
        y_start = max(0, y - pad)
        x_start = max(0, x - pad)
        y_end = min(img.shape[0], y + h + pad)
        x_end = min(img.shape[1], x + w + pad)
        img = img[y_start:y_end, x_start:x_end]
    else:
        final_img = np.zeros((28, 28), dtype=np.uint8)
        save_preprocessed_image(final_img, image_path)
        img_array = final_img.astype("float32") / 255.0
        return np.expand_dims(img_array, axis=(0, -1))

    # 6. Maintain Aspect Ratio and Resize to 20x20
    h, w = img.shape
    aspect = w / h
    if aspect > 1:
        new_w, new_h = 20, int(20 / aspect)
    else:
        new_h, new_w = 20, int(20 * aspect)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 7. Paste onto 28x28 black background
    final_img = np.zeros((28, 28), dtype=np.uint8)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    final_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img

    save_preprocessed_image(final_img, image_path)
    img_array = final_img.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=(0, -1))

def predict_digit(image_path, model_path):
    model = keras.models.load_model(model_path)
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence

def main():
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
            model_path = MNIST_MODEL_PATH if choice == '1' else DEVANAGARI_MODEL_PATH
            digit, confidence = predict_digit(image_path, model_path)

            print(f"\nPredicted digit: {digit}")
            print(f"Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
