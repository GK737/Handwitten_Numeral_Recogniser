from PIL import Image, ImageOps
import numpy as np
import keras
import cv2

model = keras.models.load_model("E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\mnist_cnn.keras")

def preprocess_image(image_path):
    # 1. Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Noise Reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Otsu's Thresholding (Better for single digits than Adaptive)
    # It finds the global optimal threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Find Bounding Box
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add a small padding (2-4 pixels) so the digit doesn't touch the edge
        pad = 4
        y_start = max(0, y - pad)
        x_start = max(0, x - pad)
        y_end = min(img.shape[0], y + h + pad)
        x_end = min(img.shape[1], x + w + pad)
        img = img[y_start:y_end, x_start:x_end]
    
    # 5. Maintain Aspect Ratio and Resize to 20x20
    h, w = img.shape
    aspect = w / h
    if aspect > 1:
        new_w, new_h = 20, int(20 / aspect)
    else:
        new_h, new_w = 20, int(20 * aspect)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 6. Paste onto 28x28 black background
    final_img = np.zeros((28, 28), dtype=np.uint8)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    final_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img

    # 7. Normalize and Expand Dimensions

    img_array = final_img.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=(0, -1))

image_path = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\src\\my_digit_2.png"
processed_image = preprocess_image(image_path)

prediction = model.predict(processed_image)
digit = np.argmax(prediction)
confidence = np.max(prediction)

print("Predicted digit:", digit)
print("Confidence:", confidence)