from typing import List, Tuple, Dict, Callable

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras import models

encoder_descriptor = [(10, 'relu')]
classifier_descriptor = [(10, 'relu'), (10, 'softmax')]


def create_encoder(input_dim: int) -> Tuple[keras.Layer, keras.Layer]:
    input = layers.Input(shape=(input_dim,))
    predecessor = input
    for index in range(0, len(encoder_descriptor)):
        descriptor = encoder_descriptor[index]
        if index == len(encoder_descriptor) - 1:
            predecessor = layers.Dense(descriptor[0], activation=descriptor[1], name="latent_space")(predecessor)
        else:
            predecessor = layers.Dense(descriptor[0], activation=descriptor[1])(predecessor)
    return input, predecessor


def create_classifier(latent_layer: keras.Layer) -> keras.Layer:
    predecessor = latent_layer
    for index in range(0, len(encoder_descriptor)):
        descriptor = encoder_descriptor[index]
        if index == len(encoder_descriptor) - 1:
            predecessor = layers.Dense(descriptor[0], activation=descriptor[1], name="class_layer")(predecessor)
        else:
            predecessor = layers.Dense(descriptor[0], activation=descriptor[1])(predecessor)
    return predecessor


def pretraining_representation_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    latent_output = y_pred[1]
    loss = (1 - latent_output)**2*latent_output**2
    return loss


def pretraining_classification_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    pass


def fine_tuning_contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    pass


def fine_tuning_classification_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    final_output = y_pred[0]





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


in_layer, latent_layer = create_encoder(10)
out_layer = create_classifier(latent_layer)

model = keras.Model(inputs=in_layer, outputs=[latent_layer, out_layer])
model.compile(optimizer="adam", loss={"latent_space": pretraining_classification_loss, "class_layer": "categorical_crossentropy"}, metrics=["accuracy"])



#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

x, y = generate_demo_data(10, 10, 10)

model.fit(x, y, epochs=2000)

