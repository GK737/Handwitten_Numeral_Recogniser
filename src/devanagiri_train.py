import numpy as np
import cv2
import keras
from pathlib import Path
from keras import layers, regularizers

# --- CONFIGURATION ---
DATASET_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\datasets\\Devanagiri_Dataset\\Train"
PRETRAINED_MODEL_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\mnist_cnn.keras"
SAVE_MODEL_PATH = "E:\\Useless_Project\\Numeral Recogniser\\Handwitten_Numeral_Recogniser\\models\\devanagari_cnn.keras"

def preprocess_image(image_path):
    """The OpenCV pipeline that ensured the predict.py worked."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 4
        img = img[max(0, y-pad):min(img.shape[0], y+h+pad), 
                  max(0, x-pad):min(img.shape[1], x+w+pad)]
    
    h, w = img.shape
    aspect = w / h
    if aspect > 1:
        new_w, new_h = 20, int(20 / aspect)
    else:
        new_h, new_w = 20, int(20 * aspect)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final_img = np.zeros((28, 28), dtype=np.uint8)
    final_img[(28-new_h)//2 : (28-new_h)//2 + new_h, 
              (28-new_w)//2 : (28-new_w)//2 + new_w] = img
    
    return final_img.astype("float32") / 255.0

def load_devanagari_dataset(base_path):
    X, y = [], []
    for digit in range(10):
        digit_folder = Path(base_path) / f"digit_{digit}"
        print(f"Loading digit {digit}...")
        for img_path in digit_folder.glob("*.*"):
            processed = preprocess_image(str(img_path))
            if processed is not None:
                X.append(processed)
                y.append(digit)
    return np.expand_dims(np.array(X), -1), np.array(y)

# 1. Load Custom Data
X_dev, y_dev = load_devanagari_dataset(DATASET_PATH)
print(f"Loaded {len(X_dev)} images.")

# 2. Transfer Learning: Load the MNIST model
model = keras.models.load_model(PRETRAINED_MODEL_PATH)

# 3. Freeze Early Layers
# We freeze the Conv2D layers to keep the basic edge-detection skills
for layer in model.layers:
    if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# 4. Re-compile with a LOWER learning rate for stability
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

# 5. Train (Fine-tune)
history = model.fit(
    X_dev, y_dev,
    batch_size=32, # Smaller batch size for smaller datasets
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks
)

# 6. Save the Regional Model
model.save(SAVE_MODEL_PATH)
print(f"Devanagari model saved to {SAVE_MODEL_PATH}")