import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "DevanagariHandwrittenCharacterDataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "Train")
TEST_PATH = os.path.join(DATASET_PATH, "Test")

IMAGE_SIZE = (28, 28)
BATCH_SIZE = 64
EPOCHS = 15

SEED = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="int",
    class_names=[
        "digit_0",
        "digit_1",
        "digit_2",
        "digit_3",
        "digit_4",
        "digit_5",
        "digit_6",
        "digit_7",
        "digit_8",
        "digit_9"
    ],
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    labels="inferred",
    label_mode="int",
    class_names=[
        "digit_0",
        "digit_1",
        "digit_2",
        "digit_3",
        "digit_4",
        "digit_5",
        "digit_6",
        "digit_7",
        "digit_8",
        "digit_9"
    ],
    color_mode="grayscale",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 3. CREATE VALIDATION DATA
# ============================================================

# Take 80% of training batches for training
# and 20% for validation

total_batches = tf.data.experimental.cardinality(train_dataset)

validation_batches = int(total_batches.numpy() * 0.2)

validation_dataset = train_dataset.take(validation_batches)

training_dataset = train_dataset.skip(validation_batches)


print("\nDataset loaded successfully.")


# ============================================================
# 4. NORMALIZATION
# ============================================================

normalization_layer = layers.Rescaling(1.0 / 255)

training_dataset = training_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    )
])


# ============================================================
# 6. CNN MODEL
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(shape=(28, 28, 1)),

    # Data augmentation
    data_augmentation,

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout to reduce overfitting
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 7. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# ============================================================
# 11. TEST MODEL
# ============================================================

print("\nEvaluating model on test dataset...\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n====================================")
print("Devanagari Test Accuracy:",
      test_accuracy * 100, "%")
print("Devanagari Test Loss:",
      test_loss)
print("====================================")


# ============================================================
# 12. SAVE MODEL
# ============================================================

model.save("devanagari_numeral_cnn.keras")

print("\nModel saved as:")
print("devanagari_numeral_cnn.keras")

