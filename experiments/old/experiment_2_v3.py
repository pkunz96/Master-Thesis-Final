import keras
import numpy as np
import tensorflow as tf


def generate_demo_data(input_size: int, num_samples: int, num_classes: int):
    """
    Generate demo data for testing.
    :param input_size: Number of input features (e.g., 32).
    :param num_samples: Number of data samples to generate (e.g., 1000).
    :param num_classes: Number of output classes (e.g., 10).
    :return: Tuple of input data (X) and labels (y).
    """
    X = np.random.rand(num_samples, input_size).astype(np.float32)  # Random features
    y = np.random.randint(0, num_classes, size=(num_samples,))       # Random class labels
    y = keras.utils.to_categorical(y, num_classes)  # One-hot encode the labels
    return X, y


model = keras.Sequential(
    [
        keras.layers.Dense(10, activation="relu", name="layer1"),
        keras.layers.Dense(10, activation="relu", name="layer2"),
        keras.layers.Dense(10, activation="relu", name="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

x, y = generate_demo_data(10, 10, 10)
print(x)
print(y)
model.fit(x, y)

