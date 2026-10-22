import cv2
import numpy as np
import keras

from preprocess import DEVANAGARI_MODEL_PATH, MALAYALAM_MODEL_PATH, MNIST_MODEL_PATH, preprocess

SCRIPTS = {
    "1": (MNIST_MODEL_PATH, "western"),
    "2": (DEVANAGARI_MODEL_PATH, "devanagari"),
    "3": (MALAYALAM_MODEL_PATH, "malayalam"),
}

def main():
    print("Select Model:")
    print("1. MNIST (Western)")
    print("2. Devanagari")
    print("3. Malayalam")
    choice = input("Choice: ")

    model_path, script = SCRIPTS.get(choice, SCRIPTS["1"])
    model = keras.models.load_model(model_path)
    debug_img = None

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
            processed, debug_img = preprocess(roi, script)
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
        if debug_img is not None:
            debug_resized = cv2.resize(debug_img, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Debug - Preprocessed Input", debug_resized)

        cv2.imshow("Live Numeral Recogniser", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
