import cv2
import numpy as np
import keras
import os

MNIST_MODEL_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\mnist_cnn.keras"
DEVANAGARI_MODEL_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\devanagari_cnn.keras"

def preprocess_frame(frame):
    # 1. Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Noise Reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Adaptive Thresholding
    # Adjusted parameters to be slightly more restrictive to avoid "ghost" lines
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    # 4. Find Bounding Box
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # REMOVED: The strict size filter that was causing "Object too large" errors.
        # We now process whatever is found, as long as it's not the entire frame (noise).
        if w == 0 or h == 0:
            return None, None

        pad = 10
        y_start = max(0, y - pad)
        x_start = max(0, x - pad)
        y_end = min(img.shape[0], y + h + pad)
        x_end = min(img.shape[1], x + w + pad)
        img = img[y_start:y_end, x_start:x_end]
    else:
        return None, None

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

    img_array = final_img.astype("float32") / 255.0
    processed_tensor = np.expand_dims(img_array, axis=(0, -1))

    return processed_tensor, final_img

def main():
    print("Select Model:")
    print("1. MNIST (Western)")
    print("2. Devanagari")
    choice = input("Choice: ")

    model_path = MNIST_MODEL_PATH if choice == '1' else DEVANAGARI_MODEL_PATH
    model = keras.models.load_model(model_path)

    cap = cv2.VideoCapture(0)

    # Prediction state
    frame_count = 0
    last_digit = None
    last_confidence = 0.0

    # ROI configuration
    roi_w, roi_h = 300, 300

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dynamic ROI Centering
        h, w = frame.shape[:2]
        roi_x = (w - roi_w) // 2
        roi_y = (h - roi_h) // 2

        # Extract ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Predict every 3rd frame
        if frame_count % 3 == 0:
            processed, debug_img = preprocess_frame(roi)
            if processed is not None:
                prediction = model.predict(processed, verbose=0)
                last_digit = np.argmax(prediction)
                last_confidence = np.max(prediction)
            else:
                last_digit = None
                last_confidence = 0.0

        frame_count += 1

        # --- UI Logic ---
        if last_digit is not None:
            status_text = "Status: Digit Detected!"
            if last_confidence > 0.8:
                box_color = (0, 255, 0)  # Green
            elif last_confidence > 0.5:
                box_color = (0, 255, 255) # Yellow
            else:
                box_color = (0, 0, 255)   # Red

            result_text = f"Digit: {last_digit} ({last_confidence:.2f})"
            cv2.putText(frame, result_text, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        else:
            status_text = "Status: Scanning..."
            box_color = (255, 0, 0) # Blue for scanning

        # Draw ROI Box
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), box_color, 2)

        # Draw Status and Instructions
        cv2.putText(frame, status_text, (roi_x, roi_y + roi_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        cv2.putText(frame, "Place digit inside the box", (w // 2 - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Debug Visualization ---
        if 'debug_img' in locals() and debug_img is not None:
            debug_resized = cv2.resize(debug_img, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Debug - Preprocessed Input", debug_resized)

        cv2.imshow("Live Numeral Recogniser", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
