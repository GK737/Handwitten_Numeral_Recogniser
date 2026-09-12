import numpy as np
import matplotlib.pyplot as plt
import keras
from pathlib import Path

from keras import layers
from keras import regularizers

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("Training:", x_train.shape)
print("Testing :", x_test.shape)

# CNN model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.1, 0.1),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    ),
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax")
])

model.summary()

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    ),
]

# Train
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=30,
    validation_split=0.1,
    callbacks=callbacks,
)

# Test
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Test accuracy:", test_accuracy)

# Save
Path("models").mkdir(exist_ok=True)
model.save("models/mnist_cnn.keras")