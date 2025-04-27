import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

# Generate synthetic demo data
num_samples = 1000
input_shape = 20  # Input feature dimension
num_classes = 3   # Number of classes for classification

# Random input data
x_train = np.random.rand(num_samples, input_shape).astype(np.float32)

# Binary labels for the intermediate layer (e.g., 0 or 1)
binary_labels = np.random.randint(0, 2, size=(num_samples, 64)).astype(np.float32)  # 64 matches intermediate layer size

# Classification labels (one-hot encoded)
classification_labels = tf.keras.utils.to_categorical(
    np.random.randint(0, num_classes, size=(num_samples,)), num_classes=num_classes
)

# Define the model
input_layer = Input(shape=(input_shape,))
intermediate_layer = Dense(64, activation='relu')(input_layer)  # Intermediate layer
output_layer = Dense(num_classes, activation='softmax')(intermediate_layer)  # Final output layer

# Create a model that outputs both the final output and intermediate layer
model = Model(inputs=input_layer, outputs=[output_layer, intermediate_layer])

# Custom loss function
def custom_loss(binary_labels, classification_labels):
    def loss(y_true, y_pred):
        # y_pred is a list containing [output_layer, intermediate_layer]
        output_layer_pred, intermediate_layer_pred = y_pred

        # Classification loss (e.g., categorical crossentropy)
        classification_loss = tf.keras.losses.categorical_crossentropy(
            classification_labels, output_layer_pred
        )

        # Custom loss using intermediate layer and binary labels
        intermediate_loss = tf.keras.losses.binary_crossentropy(
            binary_labels, intermediate_layer_pred
        )

        # Combine the losses (you can weight them as needed)
        total_loss = classification_loss + 0.5 * intermediate_loss
        return total_loss
    return loss

# Compile the model
model.compile(optimizer='adam', loss=None)  # Set loss to None since we use add_loss

# Add the custom loss
model.add_loss(custom_loss(binary_labels, classification_labels))

# Train the model
model.fit(x_train, epochs=10, batch_size=32)