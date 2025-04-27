import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate demo data for classification task
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

# Create a simple feedforward model
def create_feedforward_model(input_size: int, num_classes: int):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_size,)),  # Input layer
        layers.Dense(128, activation='relu'),           # Hidden layer with 128 units
        layers.Dense(64, activation='relu'),            # Another hidden layer with 64 units
        layers.Dense(num_classes, activation='softmax') # Output layer for classification
    ])
    return model

# Parameters
input_size = 10  # Number of features in the input data
num_samples = 1000  # Number of training samples
num_classes = 10  # Number of output classes (for classification)

# Generate demo data
X_train, y_train = generate_demo_data(input_size, num_samples, num_classes)

# Create the feedforward model
model = create_feedforward_model(input_size, num_classes)

# Compile the model with Sparse Categorical Crossentropy for classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to check the architecture
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32)
