import base64
import itertools
import os
import pickle
import random
import time
from datetime import datetime
from typing import Dict, Callable, Optional, List, Tuple
import re

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from joblib import dump, load
import os

import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Categorical

from algorithms.environment_oop import Environment
from experiments import sampling
from experiments.sampling import ParameterSet


class Layer(tf.Module):

    activation_function_dict: Dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
        "sigmoid": tf.nn.sigmoid,
        "relu": tf.nn.relu,
        "softmax": tf.nn.softmax
    }

    @staticmethod
    def assure_is_valid_input_dim(input_dim: int):
        if input_dim is None or input_dim < 1:
            raise ValueError("Input dimension have not been initialized properly")

    @staticmethod
    def assure_is_valid_output_dim(output_dim: int):
        if output_dim is None or output_dim < 1:
            raise ValueError("Output dimension have not been initialized properly")

    @staticmethod
    def assure_is_valid_activation_function(activation_func: str):
        if activation_func not in Layer.activation_function_dict:
            raise ValueError("Specified activation function (" + activation_func + ") is not supported.")

    def __init__(self, name: str, input_dim: int, output_dim: int, forward: bool, loss_function: Optional[Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]], loss_weight: float = 1.0, activation_function_name: str = "relu"):

        super().__init__()

        # Validation Logic
        Layer.assure_is_valid_input_dim(input_dim)
        Layer.assure_is_valid_input_dim(input_dim)
        Layer.assure_is_valid_activation_function(activation_function_name)

        # Meta-Information
        self.layer_name = name

        # Architecture

        self.successor: Optional[Layer] = None
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.forward: bool = forward
        self.activation_function: Callable[[tf.Tensor], tf.Tensor] = Layer.activation_function_dict[activation_function_name]
        self.activation_function_name: str = activation_function_name

        # Loss
        self.loss_function: Optional[Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = loss_function
        self.loss_weight = tf.constant(loss_weight)
        self.cur_loss = None

        # Weight Initialization
        if forward:
            self.w1: tf.Variable = tf.Variable(tf.random.normal([input_dim-1, output_dim-1]))
            self.b1: tf.Variable = tf.Variable(tf.random.normal([1, output_dim-1]))
        else:
            self.w1: tf.Variable = tf.Variable(tf.random.normal([input_dim, output_dim]))
            self.b1: tf.Variable = tf.Variable(tf.random.normal([1, output_dim]))

    def get_name(self):
        return self.layer_name

    def clear_successor(self):
        self.successor = None

    def add_successor(self, successor: "Layer") -> "Layer":
        self.successor = successor
        return successor

    def assure_is_compatible(self, successor: "Layer") -> None:
        if self.output_dim != successor.input_dim:
            raise ValueError("Succeeding layer's input dimension does not match this layer's output dimension.")

    def run(self, in_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        out_val: tf.Tensor = self.eval(in_val)
        suc_loss: tf.Tensor = tf.constant(0.0)
        cur_loss: tf.Tensor = self.calc_loss(out_val, x, x_index,  y)
        if self.successor is not None:
            suc_loss, out_val = self.successor.run(out_val, x, x_index, y)
        self.cur_loss = cur_loss
        return suc_loss + cur_loss, out_val

    def get_loss(self):
        return self.cur_loss

    def predict(self, in_val: tf.Tensor) -> tf.Tensor:
        activation_value: tf.Tensor = self.eval(in_val)
        if self.successor is not None:
            return self.successor.predict(activation_value)
        else:
            return activation_value

    def eval(self, in_val: tf.Tensor) -> tf.Tensor:
        in_val_internal: tf.Tensor = in_val
        if self.forward:
            in_val_internal = in_val[:, 1:]
        linear = in_val_internal @ self.w1 + self.b1
        activated = self.activation_function(linear)
        if self.forward:
            activated = tf.concat([in_val[:, : 1], activated], axis=1)
        return activated

    def calc_loss(self, out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor):
        loss: tf.Tensor = tf.constant(0.0)
        if self.loss_function is not None:
            if self.forward:
                loss = self.loss_function(out_val[:, 1:], x[:, 1:], x_index, y)
            else:
                loss = self.loss_function(out_val, x, x_index, y)
            if self.loss_weight is not None:
                loss = tf.constant(self.loss_weight) * loss
        return loss

    def copy(self) -> "Layer":
        copy: Layer = Layer(self.layer_name, self.input_dim, self.output_dim, self.forward, self.loss_function, self.loss_weight.numpy(),  activation_function_name=self.activation_function_name)
        copy.w1 = tf.Variable(self.w1, trainable=self.w1.trainable)
        copy.b1 = tf.Variable(self.b1, trainable=self.b1.trainable)
        return copy


class BinaryLayer(Layer):

    def __init__(self, name: str, input_dim: int):
        super().__init__(name, input_dim, 0, True, None, 1.0, "sigmoid")
        weight_matrix = None
        for inner_index in range(1, input_dim):
            row = None
            for outer_index in range(inner_index + 1, input_dim):
                row = [-1.0 * float(x == inner_index) + 1.0*float(x == outer_index) for x in range(1, input_dim)]
                if weight_matrix is None:
                    weight_matrix = tf.constant([row])
                else:
                    weight_matrix = tf.concat([weight_matrix, tf.constant([row])], axis=0)
        self.w1 = tf.Variable(tf.transpose(tf.constant(weight_matrix)), trainable=False)
        self.b1 = tf.Variable(tf.zeros((1, self.w1.shape[1]), dtype=tf.float32), trainable=False)
        self.output_dim = self.w1.shape[0]


def categorical_cross_entropy_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(out_val, y)


def binary_representation_loss_original(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    binary_loss = tf.math.log((tf.reduce_sum(tf.square(tf.subtract(out_val, 1)) * tf.square(out_val))) + tf.constant(1.0))
    return binary_loss


def binary_representation_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    binary_loss = tf.math.log((tf.reduce_sum(tf.square(tf.subtract(out_val, 1)) * tf.square(out_val))) + tf.constant(1.0))
    return binary_loss


def disabled_binary_representation_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.constant(0.0)


def create_contrastive_loss(cluster_labels: List[int], margin: float = 1.0) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:

    cluster_labels_tf = tf.constant(cluster_labels, dtype=tf.int32)

    def contrastive_loss_original(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

        batch_size = tf.shape(out_val)[0]
        indices = tf.where(tf.range(batch_size)[:, None] < tf.range(batch_size))

        out_val_1 = tf.gather(out_val, indices[:, 0])  # First element of each pair
        out_val_2 = tf.gather(out_val, indices[:, 1])

        distances = tf.sqrt(tf.reduce_sum((out_val_1 - out_val_2) ** 2, axis=1) + tf.constant(10**-2))

        def get_label(index_pair):
            i_0 = x_index[index_pair[0]][0]
            i_1 = x_index[index_pair[1]][0]
            return int(cluster_labels[i_0] == cluster_labels[i_1])

        with ThreadPoolExecutor() as executor:
           labels = list(executor.map(lambda index_pair: get_label(index_pair), indices))

        c_margin = tf.constant(margin, dtype=tf.float32)

        loss_similar = tf.cast(labels, tf.float32) * (distances ** 2)
        loss_dissimilar = (1 - tf.cast(labels, tf.float32)) * tf.square(tf.maximum(0.0, c_margin - distances))

        loss = tf.reduce_mean(loss_similar + loss_dissimilar)
        return loss + tf.math.log((tf.reduce_sum(tf.square(tf.subtract(out_val, 1)) * tf.square(out_val))) + tf.constant(1.0))

    def contrastive_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

        batch_size = tf.shape(out_val)[0]
        indices = tf.where(tf.range(batch_size)[:, None] < tf.range(batch_size))

        out_val_1 = tf.gather(out_val, indices[:, 0])  # First element of each pair
        out_val_2 = tf.gather(out_val, indices[:, 1])

        distances = tf.sqrt(tf.reduce_sum((out_val_1 - out_val_2) ** 2, axis=1) + tf.constant(10**-2))

        cluster_1 = tf.gather(cluster_labels_tf, tf.gather(x_index[:, 0], indices[:, 0]))
        cluster_2 = tf.gather(cluster_labels_tf, tf.gather(x_index[:, 0], indices[:, 1]))
        labels = tf.cast(tf.equal(cluster_1, cluster_2), tf.float32)

        c_margin = tf.constant(margin, dtype=tf.float32)

        loss_similar = tf.cast(labels, tf.float32) * (distances ** 2)
        loss_dissimilar = (1 - tf.cast(labels, tf.float32)) * tf.square(tf.maximum(0.0, c_margin - distances))

        loss = tf.reduce_mean(loss_similar + loss_dissimilar)
        return loss + tf.math.log((tf.reduce_sum(tf.square(tf.subtract(out_val, 1)) * tf.square(out_val))) + tf.constant(1.0))

    return contrastive_loss

def create_disabled_contrastive_loss(cluster_labels: List[int], margin: float = 1.0) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
    def contrastive_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.constant(0.0)

    return contrastive_loss


class Procedure:

    optimizer_dict: Dict[str, tf.keras.optimizers.Optimizer] = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD
    }

    metrics_list = ["training_loss", "validation_loss", "training_accuracy", "validation_accuracy"]

    @staticmethod
    def compile_model(layer_arr: List[Layer]) -> Layer:
        if len(layer_arr) == 0:
            raise ValueError("The passed layer array must not be empty")
        for layer in layer_arr:
            layer.clear_successor()
        for index in range(0, len(layer_arr) - 1):
            cur_layer: Layer = layer_arr[index]
            suc_layer: Layer = layer_arr[index + 1]
            cur_layer.add_successor(suc_layer)
        return layer_arr[0]

    @staticmethod
    def extract_loss_per_layer(model: Layer) -> Dict[str, tf.Tensor]:
        layer_loss_dict: Dict[str, tf.Tensor] = {}
        cur_layer = model
        while cur_layer is not None:
            if cur_layer.loss_function is not None:
                layer_loss_dict[cur_layer.layer_name] = cur_layer.cur_loss
            cur_layer = cur_layer.successor
        return layer_loss_dict

    @staticmethod
    def merge_training_data(training_data_list: List[Tuple[tf.Tensor, tf.Tensor]]) -> Tuple[tf.Tensor, tf.Tensor]:
        x_data = tf.constant(training_data_list[0][0])
        y_data = tf.constant(training_data_list[0][1])
        for index in range(1, len(training_data_list)):
            cur_x_data, cur_y_data = training_data_list[index]
            x_data = tf.concat([cur_x_data, x_data], axis=0)
            y_data = tf.concat([cur_y_data, y_data], axis=0)
        return x_data, y_data

    @staticmethod
    def merge_training_data_dict(training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> Tuple[tf.Tensor, tf.Tensor]:
        return Procedure.merge_training_data(list(training_data_dict.values()))

    @staticmethod
    def create_batch(x_data: tf.Tensor, y_data: tf.Tensor, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        data_size = tf.shape(x_data)[0]
        if batch_size is None or batch_size > data_size:
            indices = tf.range(data_size)
        else:
            indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=data_size, dtype=tf.int32)
        x_batch = tf.gather(x_data, indices)
        y_batch = tf.gather(y_data, indices)
        return x_batch, y_batch, tf.expand_dims(indices, axis=-1)

    @staticmethod
    def calc_multi_data_set_accuracy(data_set_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        data_set_accuracy_dict = {}
        for data_set_name in data_set_dict:
            x, y = data_set_dict[data_set_name]
            # y_pred_classes = tf.argmax(model.predict(x), axis=1)
            y_pred_classes = tf.argmax(x, axis=1)
            y_true_classes = tf.argmax(y, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype=tf.float32))
            data_set_accuracy_dict[data_set_name] = accuracy
        return data_set_accuracy_dict

    @staticmethod
    def calc_accuracy(model: Layer, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return Procedure.calc_multi_data_set_accuracy({"str": (x, y)})["str"]

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], epochs: int = 100, learning_rate: float = 1.0, batch_size: int = 128, optimizer: str = "adam", early_stopping: int = 50, cache_validation_data: bool = False):

        # Hyperparameters
        self.layer: Layer = layer
        self.epochs: int = epochs
        self.alpha: float = learning_rate
        self.batch_size: int = batch_size
        self.optimizer: tf.keras.optimizers.Optimizer = Procedure.optimizer_dict[optimizer](learning_rate=learning_rate)

        self.early_stopping: int  = early_stopping
        # Training Data

        self.cache: bool = cache_validation_data

        self.training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = training_data_dict
        self.validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = validation_data_dict
        self.test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = test_data_dict

        # Metrics
        self.metrics = Procedure.metrics_list

        # Meta-Data
        self.epoch_arr: List[int] = []

        # Training loss
        self.training_loss_list: List[tf.Tensor] = []
        # Training loss w.r.t. individual layers
        self.layer_name_training_loss_dict: Dict[str, List[tf.Tensor]] = {}
        # Average validation loss
        self.validation_loss_list: List[tf.Tensor] = []
        # Average validation loss w.r.t. individual layers
        self.layer_name_validation_loss_dict: Dict[str, List[tf.Tensor]] = {}
        # Validation loss w.r.t. individual data sets
        self.data_name_validation_loss_dict: dict[str, List[tf.Tensor]] = {}

        # Training accuracy
        self.training_accuracy_list: List[tf.Tensor] = []
        # Average validation accuracy
        self.validation_accuracy_list: List[tf.Tensor] = []
        # Validation accuracy w. r. t. individual validation data sets
        self.data_name_validation_accuracy_dict: Dict[str, List[tf.Tensor]] = {}

        # Average test accuracy
        self.average_test_accuracy: Optional[tf.Tensor] = None
        # Test accuracy w. r. t. individual test data sets
        self.test_data_name_accuracy: Dict[str, tf.Tensor] = {}

        self.test_data_name_class_accuracy_dict = None

        # Graphics

        self.training_loss_line: Optional[Line2D] = None
        self.validation_loss_line: Optional[Line2D] = None
        self.layer_name_training_loss_line_dict: Dict[str, Line2D] = {}
        self.layer_name_validation_loss_line_dict: Dict[str, Line2D] = {}

        self.data_name_validation_loss_line_dict: Dict[str, Line2D] = {}

        self.training_accuracy_line: Optional[Line2D] = None
        self.validation_accuracy_line: Optional[Line2D] = None
        self.data_name_validation_accuracy_line_dict: Dict[str, Line2D] = {}

        self.loss_fig: Optional[Figure] = None
        self.loss_ax: Optional[Axes] = None

        self.data_set_loss_fig: Optional[Figure] = None
        self.data_set_loss_ax: Optional[Axes] = None

        self.accuracy_fig: Optional[Figure] = None
        self.accuracy_ax: Optional[Axes] = None

    def train(self) -> Layer:
        self.clear()
        best_validation_loss = None
        best_param_list = None
        no_improvement_counter = 0
        epoch_counter = 0
        for epoch in range(self.epochs):
            epoch_counter += 1
            self._execute_training_step()
            #self._validate_model_original()
            self._validate_model(epoch)

            self.epoch_arr.append(epoch)
            self._update_visualization()
            if self.early_stopping is not None:
                cur_loss = self.validation_loss_list[epoch]
                if best_validation_loss is None or cur_loss < best_validation_loss:
                    best_validation_loss = cur_loss
                    no_improvement_counter = 0
                    best_param_list = []
                    cur_layer = self.layer
                    while cur_layer is not None:
                        best_param_list.append((tf.Variable(cur_layer.w1), tf.Variable(cur_layer.b1)))
                        cur_layer = cur_layer.successor
                else:
                    no_improvement_counter += 1
                if no_improvement_counter > self.early_stopping:
                    cur_layer = self.layer
                    for w, b in best_param_list:
                        cur_layer.w1 = w
                        cur_layer.b1 = b
                        cur_layer = cur_layer.successor
                    break
        self.epochs = epoch_counter
        self._test_model()
        plt.ioff()
        return self.layer

    def _on_training_step_completed(self, training_loss: tf.Tensor, training_accuracy: tf.Tensor, layer_name_training_loss_dict: Dict[str, tf.Tensor]) -> None:
        self.training_loss_list.append(training_loss)
        self.training_accuracy_list.append(training_accuracy)
        for layer_name in layer_name_training_loss_dict:
            if layer_name not in self.layer_name_training_loss_dict:
                self.layer_name_training_loss_dict[layer_name] = []
            self.layer_name_training_loss_dict[layer_name].append(layer_name_training_loss_dict[layer_name])

    @staticmethod
    def continue_list(list):
        if list is not None and len(list) > 0:
            list.append(list[len(list) - 1])

    def _on_validation_step_completed(self, validation_loss: tf.Tensor, data_name_validation_loss_dict: Dict[str, tf.Tensor], layer_name_validation_loss_dict: Dict[str, tf.Tensor], validation_accuracy: tf.Tensor, data_name_validation_accuracy_dict: Dict[str, tf.Tensor]) -> None:

        if validation_loss is None:
            Procedure.continue_list(self.validation_loss_list)
        else:
            self.validation_loss_list.append(validation_loss)

        if data_name_validation_loss_dict is None:
            for data_name in self.data_name_validation_loss_dict:
                Procedure.continue_list(self.data_name_validation_loss_dict[data_name])
        else:
            for data_name in data_name_validation_loss_dict:
                if data_name not in self.data_name_validation_loss_dict:
                    self.data_name_validation_loss_dict[data_name] = []
                self.data_name_validation_loss_dict[data_name].append(data_name_validation_loss_dict[data_name])

        if layer_name_validation_loss_dict is None:
            for layer_name in self.layer_name_validation_loss_dict:
                Procedure.continue_list(self.layer_name_validation_loss_dict[layer_name])
        else:
            for layer_name in layer_name_validation_loss_dict:
                if layer_name not in self.layer_name_validation_loss_dict:
                    self.layer_name_validation_loss_dict[layer_name] = []
                self.layer_name_validation_loss_dict[layer_name].append(layer_name_validation_loss_dict[layer_name])

        if validation_accuracy is None:
            Procedure.continue_list(self.validation_accuracy_list)
        else:
            self.validation_accuracy_list.append(validation_accuracy)

        if data_name_validation_accuracy_dict is None:
            for layer_name in self.data_name_validation_accuracy_dict:
                Procedure.continue_list(self.data_name_validation_accuracy_dict[layer_name])
        else:
            for data_name in data_name_validation_accuracy_dict:
                if data_name not in self.data_name_validation_accuracy_dict:
                    self.data_name_validation_accuracy_dict[data_name] = []
                self.data_name_validation_accuracy_dict[data_name].append(data_name_validation_accuracy_dict[data_name])

    def _on_validation_step_completed_original(self, validation_loss: tf.Tensor, data_name_validation_loss_dict: Dict[str, tf.Tensor], layer_name_validation_loss_dict: Dict[str, tf.Tensor], validation_accuracy: tf.Tensor, data_name_validation_accuracy_dict: Dict[str, tf.Tensor]) -> None:
        self.validation_loss_list.append(validation_loss)

        for data_name in data_name_validation_loss_dict:
            if data_name not in self.data_name_validation_loss_dict:
                self.data_name_validation_loss_dict[data_name] = []
            self.data_name_validation_loss_dict[data_name].append(data_name_validation_loss_dict[data_name])

        for data_name in layer_name_validation_loss_dict:
            if data_name not in self.layer_name_validation_loss_dict:
                self.layer_name_validation_loss_dict[data_name] = []
            self.layer_name_validation_loss_dict[data_name].append(layer_name_validation_loss_dict[data_name])

        self.validation_accuracy_list.append(validation_accuracy)

        for data_name in data_name_validation_accuracy_dict:
            if data_name not in self.data_name_validation_accuracy_dict:
                self.data_name_validation_accuracy_dict[data_name] = []
            self.data_name_validation_accuracy_dict[data_name].append(data_name_validation_accuracy_dict[data_name])

    def _on_training_completed(self, average_test_accuracy: tf.Tensor, test_data_name_accuracy: Dict[str, tf.Tensor], test_data_name_class_accuracy_dict) -> None:
        self.average_test_accuracy = average_test_accuracy
        self.test_data_name_accuracy = test_data_name_accuracy
        self.test_data_name_class_accuracy_dict = test_data_name_class_accuracy_dict

    def _execute_training_step(self):
        loss = None
        x_data, y_data = self.merge_training_data_dict(self.training_data_dict)
        x_sample, y_sample, x_index = Procedure.create_batch(x_data, y_data, self.batch_size)
        with tf.GradientTape(persistent=False) as tape:
            layer_params = []
            current_layer = self.layer
            while current_layer is not None:
                tape.watch(current_layer.w1)
                tape.watch(current_layer.b1)
                layer_params.append((current_layer.w1, current_layer.b1))
                current_layer = current_layer.successor
            loss, prediction = self.layer.run(x_sample, x_sample, x_index, y_sample)
        dw_list = [param[0] for param in layer_params]
        db_list = [param[1] for param in layer_params]
        gradients = tape.gradient(loss, dw_list + db_list)
        self.optimizer.apply_gradients(zip(gradients, dw_list + db_list))
        layer_loss_dict = Procedure.extract_loss_per_layer(self.layer)
        accuracy = Procedure.calc_accuracy(self.layer, self.layer.predict(x_sample), y_sample)
        self._on_training_step_completed(loss, accuracy, layer_loss_dict)
        return layer_loss_dict, loss, accuracy

    def _test_model(self) -> None:
        test_data_name_accuracy: Dict[str, tf.Tensor] = {}
        test_data_list: List[Tuple[tf.Tensor, tf.Tensor]] = []
        for data_name in self.test_data_dict:
            x_data, y_data = self.test_data_dict[data_name]
            test_data_list.append((x_data, y_data))
            test_data_name_accuracy[data_name] = Procedure.calc_accuracy(self.layer, self.layer.predict(x_data), y_data)
        x_data, y_data = Procedure.merge_training_data(test_data_list)
        average_test_accuracy = Procedure.calc_accuracy(self.layer, self.layer.predict(x_data), y_data)

        test_data_name_class_accuracy_dict = dict()
        for data_name in self.test_data_dict:
            test_data_name_class_accuracy_dict[data_name] = dict()
            response_predictor = dict()
            x_data, y_data = self.test_data_dict[data_name]
            for index in range(len(y_data)):
                response = tuple(y_data[index].numpy())
                if response not in response_predictor:
                    response_predictor[response] = []
                response_predictor[response].append(x_data[index].numpy())
            for response in response_predictor:
                predictor_arr = response_predictor[response]
                response_arr = np.array([response for x in predictor_arr])
                cur_x_data = tf.constant(predictor_arr)
                cur_y_data = tf.constant(response_arr)
                accuracy = Procedure.calc_accuracy(self.layer, self.layer.predict(cur_x_data), cur_y_data)
                test_data_name_class_accuracy_dict[data_name][response] = accuracy

        self._on_training_completed(average_test_accuracy, test_data_name_accuracy, test_data_name_class_accuracy_dict)

    def _validate_model(self, epoch) -> None:
        # Average
        x_val, y_val = Procedure.merge_training_data([x for x in self.validation_data_dict.values()])
        x_val_sample, y_val_sample, indices = Procedure.create_batch(x_val, y_val, 64)
        average_val_loss, prediction = self.layer.run(x_val_sample, x_val_sample, indices, y_val_sample)

        average_accuracy = None
        layer_name_validation_loss_dict = None
        data_name_validation_loss_dict = None
        data_name_validation_accuracy_dict = None

        #IMPORTANT: Epoch counting must start with 0, otherwise data structures are not properly initialized.
        if not self.cache or epoch % 15 == 0:
            average_accuracy = Procedure.calc_accuracy(self.layer, y_val_sample, prediction)
            layer_name_validation_loss_dict = Procedure.extract_loss_per_layer(self.layer)
            data_name_validation_loss_dict: Dict[str, tf.Tensor] = {}
            data_name_validation_accuracy_dict: Dict[str, tf.Tensor] = {}
            for data_name in self.validation_data_dict:
                x_data, y_data = self.validation_data_dict[data_name]
                x_data_sample, y_data_sample, indices = Procedure.create_batch(x_data, y_data, 64)
                loss, prediction = self.layer.run(x_data_sample, y_data_sample, indices, y_data_sample)
                data_accuracy = Procedure.calc_accuracy(self.layer, y_data_sample, prediction)
                data_name_validation_loss_dict[data_name] = loss
                data_name_validation_accuracy_dict[data_name] = data_accuracy

        self._on_validation_step_completed(average_val_loss, data_name_validation_loss_dict,
                                               layer_name_validation_loss_dict, average_accuracy,
                                               data_name_validation_accuracy_dict)


    def _validate_model_original(self) -> None:
        # Average
        x_val, y_val = Procedure.merge_training_data([x for x in self.validation_data_dict.values()])
        x_val_sample, y_val_sample, indices = Procedure.create_batch(x_val, y_val, 64)
        average_val_loss, prediction = self.layer.run(x_val_sample, x_val_sample, indices, y_val_sample)

        average_accuracy = Procedure.calc_accuracy(self.layer, y_val_sample, prediction)
        layer_name_validation_loss_dict = Procedure.extract_loss_per_layer(self.layer)

        data_name_validation_loss_dict: Dict[str, tf.Tensor] = {}
        data_name_validation_accuracy_dict: Dict[str, tf.Tensor] = {}

        for data_name in self.validation_data_dict:
            x_data, y_data = self.validation_data_dict[data_name]
            x_data_sample, y_data_sample, indices = Procedure.create_batch(x_data, y_data, 64)
            loss , prediction = self.layer.run(x_data_sample, y_data_sample, indices, y_data_sample)
            data_accuracy = Procedure.calc_accuracy(self.layer, y_data_sample, prediction)
            data_name_validation_loss_dict[data_name] = loss
            data_name_validation_accuracy_dict[data_name] = data_accuracy
        self._on_validation_step_completed(average_val_loss, data_name_validation_loss_dict, layer_name_validation_loss_dict, average_accuracy, data_name_validation_accuracy_dict)

    def _init_visualization_context(self):
        plt.ion()
        # Loss figure
        self.loss_fig, self.loss_ax = plt.subplots()
        self.loss_ax.set_xlabel('Epochs')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Average Training and Validation Loss')
        self.training_loss_line = self.loss_ax.plot([], [], label='Training Loss', marker='o')[0]
        self.validation_loss_line = self.loss_ax.plot([], [], label='Validation Loss', marker='o')[0]
        for layer_name in Procedure.extract_loss_per_layer(self.layer):
            self.layer_name_training_loss_line_dict[layer_name] = self.loss_ax.plot([], [], label='Training Loss @ ' + layer_name, marker='o')[0]
        for layer_name in Procedure.extract_loss_per_layer(self.layer):
            self.layer_name_validation_loss_line_dict[layer_name] = self.loss_ax.plot([], [], label='Validation Loss @ ' + layer_name, marker='o')[0]
        self.loss_ax.legend()
        self.loss_ax.grid(True)
        self.data_set_loss_fig, self.data_set_loss_ax = plt.subplots()
        self.data_set_loss_ax.set_xlabel('Epochs')
        self.data_set_loss_ax.set_ylabel('Loss')
        self.data_set_loss_ax.set_title('Training Loss for Validation Data Sets')
        for data_name in self.validation_data_dict:
            self.data_name_validation_loss_line_dict[data_name] = self.data_set_loss_ax.plot([], [], label='Validation Loss for ' + data_name, marker='o')[0]
        self.data_set_loss_ax.legend()
        self.data_set_loss_ax.grid(True)

        # Accuracy figure
        self.accuracy_fig, self.accuracy_ax = plt.subplots()
        self.accuracy_ax.set_xlabel('Epochs')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.set_title('Accuracy')
        self.training_accuracy_line = self.accuracy_ax.plot([], [], label='Training Accuracy', marker='o')[0]
        self.validation_accuracy_line = self.accuracy_ax.plot([], [], label='Validation Accuracy', marker='o')[0]
        for data_name in self.validation_data_dict:
            self.data_name_validation_accuracy_line_dict[data_name] = self.accuracy_ax.plot([], [], label='Validation Accuracy for ' + data_name, marker='o')[0]
        self.accuracy_ax.legend()
        self.accuracy_ax.grid(True)
        plt.show()

    def _update_visualization(self):
        self.training_loss_line.set_xdata(self.epoch_arr)
        self.training_loss_line.set_ydata(self.training_loss_list)

        self.validation_loss_line.set_xdata(self.epoch_arr)
        self.validation_loss_line.set_ydata(self.validation_loss_list)

        for name in self.layer_name_training_loss_dict:
            self.layer_name_training_loss_line_dict[name].set_xdata(self.epoch_arr)
            self.layer_name_training_loss_line_dict[name].set_ydata(self.layer_name_training_loss_dict[name])

        for name in self.layer_name_validation_loss_dict:
            self.layer_name_validation_loss_line_dict[name].set_xdata(self.epoch_arr)
            self.layer_name_validation_loss_line_dict[name].set_ydata(self.layer_name_validation_loss_dict[name])

        for name in self.data_name_validation_loss_dict:
            self.data_name_validation_loss_line_dict[name].set_xdata(self.epoch_arr)
            self.data_name_validation_loss_line_dict[name].set_ydata(self.data_name_validation_loss_dict[name])

        self.training_accuracy_line.set_xdata(self.epoch_arr)
        self.training_accuracy_line.set_ydata(self.training_accuracy_list)

        self.validation_accuracy_line.set_xdata(self.epoch_arr)
        self.validation_accuracy_line.set_ydata(self.validation_accuracy_list)

        for name in self.data_name_validation_accuracy_dict:
            self.data_name_validation_accuracy_line_dict[name].set_xdata(self.epoch_arr)
            self.data_name_validation_accuracy_line_dict[name].set_ydata(self.data_name_validation_accuracy_dict[name])

        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        self.data_set_loss_ax.relim()
        self.data_set_loss_ax.autoscale_view()
        self.accuracy_ax.relim()
        self.accuracy_ax.autoscale_view()

        plt.draw()
        plt.pause(0.1)

    def clear(self):
        self.epoch_arr: List[int] = []
        self.training_loss_list: List[tf.Tensor] = []
        self.layer_name_training_loss_dict: Dict[str, List[tf.Tensor]] = {}
        self.validation_loss_list: List[tf.Tensor] = []
        self.layer_name_validation_loss_dict: Dict[str, List[tf.Tensor]] = {}
        self.data_name_validation_loss_dict: dict[str, List[tf.Tensor]] = {}
        self.training_accuracy_list: List[tf.Tensor] = []
        self.validation_accuracy_list: List[tf.Tensor] = []
        self.data_name_validation_accuracy_dict: Dict[str, List[tf.Tensor]] = {}
        self.average_test_accuracy: Optional[tf.Tensor] = None
        self.test_data_name_accuracy: Dict[str, tf.Tensor] = {}
        self.training_loss_line: Optional[Line2D] = None
        self.validation_loss_line: Optional[Line2D] = None
        self.layer_name_validation_loss_line_dict: Dict[str, Line2D] = {}
        self.data_name_validation_loss_line_dict: Dict[str, Line2D] = {}
        self.training_accuracy_line: Optional[Line2D] = None
        self.validation_accuracy_line: Optional[Line2D] = None
        self.data_name_validation_accuracy_line_dict: Dict[str, Line2D] = {}
        self.loss_fig: Optional[Figure] = None
        self.loss_ax: Optional[Axes] = None
        self.data_set_loss_fig: Optional[Figure] = None
        self.data_set_loss_ax: Optional[Axes] = None
        self.accuracy_fig: Optional[Figure] = None
        self.accuracy_ax: Optional[Axes] = None
        self._init_visualization_context()


class MLDGProcedure(Procedure):

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], epochs: int = 100, batch_size: int = 128, optimizer: str = "adam",  early_stopping: int = 50, alpha: float = 0.1, beta: float =0.1, gamma :float = 0.1):
        super().__init__(layer, training_data_dict, validation_data_dict, test_data_dict, epochs, alpha, batch_size, optimizer, early_stopping)
        self.meta_optimizer: tf.keras.optimizers.Optimizer = Procedure.optimizer_dict[optimizer](learning_rate=gamma)
        self.layer_cpy: Layer = self._copy_model()
        self.gamma = tf.constant(gamma)
        self.beta = tf.constant(beta)

    def _execute_training_step(self):

        x_training_data, y_training_data, x_meta_training_data, y_meta_training_date = self._merge_training_data()

        x_training_sample, y_training_sample, x_training_index = Procedure.create_batch(x_training_data, y_training_data, self.batch_size)
        x_meta_training_sample, y_meta_training_sample, x_meta_training_index = Procedure.create_batch(x_meta_training_data, y_meta_training_date, self.batch_size)

        model: Layer = self.layer
        model_cpy: Layer = self.layer_cpy
        self._sync()

        model_params = []
        model_params_cpy = []

        current_layer = model
        while current_layer is not None:
            model_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        current_layer = model_cpy
        while current_layer is not None:
            model_params_cpy.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        with tf.GradientTape(persistent=True) as tape:
            for var in model_params:
                tape.watch(var)
            loss, prediction = model.run(x_training_sample, x_training_sample, x_training_index, y_training_sample)

        dw_list = [param[0] for param in model_params]
        db_list = [param[1] for param in model_params]
        gradients = tape.gradient(loss, dw_list + db_list)

        dw_list_cpy = [param[0] for param in model_params_cpy]
        db_list_cpy = [param[1] for param in model_params_cpy]
        self.optimizer.apply_gradients(zip(gradients, dw_list_cpy + db_list_cpy))

        with tf.GradientTape(persistent=True) as meta_tape:
            for var in model_params_cpy:
                meta_tape.watch(var)
            meta_loss, _ = model_cpy.run(x_meta_training_sample, x_meta_training_sample, x_meta_training_index, y_meta_training_sample)
        meta_gradients = meta_tape.gradient(meta_loss, dw_list_cpy + db_list_cpy)

        gradient_sums = []
        for index in range(len(gradients)):
            gradient_sums.append(gradients[index] + self.beta * meta_gradients[index])

        self.meta_optimizer.apply_gradients(zip(gradient_sums, dw_list + db_list))

        layer_loss_dict = Procedure.extract_loss_per_layer(self.layer)
        accuracy = Procedure.calc_accuracy(self.layer, self.layer.predict(x_training_sample), y_training_sample)
        self._on_training_step_completed(loss, accuracy, layer_loss_dict)

        del tape
        del meta_tape

    def _sync(self):
        model_params = []
        model_params_cpy = []

        current_layer = self.layer
        while current_layer is not None:
            model_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        current_layer = self.layer_cpy
        while current_layer is not None:
            model_params_cpy.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        for index in range(len(model_params)):
            model_params_cpy[index][0].assign(model_params[index][0])
            model_params_cpy[index][1].assign(model_params[index][1])

    def _copy_model(self):
        layer_arr: List[Layer] = []
        cur_layer: Layer = self.layer
        while cur_layer is not None:
            layer_arr.append(cur_layer)
            cur_layer = cur_layer.successor
        model_cpy = list(map(lambda layer: layer.copy(), layer_arr))
        return Procedure.compile_model(model_cpy)

    def _merge_training_data(self):
        training_data_list = []
        meta_training_data_list = []
        while len(training_data_list) == 0 or len(meta_training_data_list) == 0:
            for name in self.training_data_dict:
                value = np.random.randint(0, 2, size=1)[0]
                rand = int(value)
                if rand == 0:
                    training_data_list.append(self.training_data_dict[name])
                else:
                    meta_training_data_list.append(self.training_data_dict[name])
        x_training_data, y_training_data = AbstractSearch.stack_training_data(training_data_list)
        x_meta_training_data, y_meta_training_data = AbstractSearch.stack_training_data(meta_training_data_list)
        return x_training_data, y_training_data, x_meta_training_data, y_meta_training_data


class DIBEPretraining(Procedure):

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], epochs: int = 100, batch_size: int = 128, optimizer: str = "adam",  early_stopping: int = 50, alpha: float = 0.1, beta: float =0.1, gamma :float = 0.1):
        super().__init__(layer, training_data_dict, validation_data_dict, test_data_dict, epochs, alpha, batch_size, optimizer, early_stopping)
        self.meta_optimizer: tf.keras.optimizers.Optimizer = Procedure.optimizer_dict[optimizer](learning_rate=gamma)
        self.layer_cpy: Layer = self._copy_model()
        self.gamma = tf.constant(gamma)
        self.beta = tf.constant(beta)

    def _execute_training_step(self):

        x_training_data, y_training_data, x_meta_training_data, y_meta_training_date = self._merge_training_data()

        x_training_sample, y_training_sample, x_training_index = Procedure.create_batch(x_training_data, y_training_data, self.batch_size)
        x_meta_training_sample, y_meta_training_sample, x_meta_training_index = Procedure.create_batch(x_meta_training_data, y_meta_training_date, self.batch_size)

        model: Layer = self.layer
        model_cpy: Layer = self.layer_cpy
        self._sync()

        model_params = []
        model_params_cpy = []

        current_layer = model
        while current_layer is not None:
            model_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        current_layer = model_cpy
        while current_layer is not None:
            model_params_cpy.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        with tf.GradientTape(persistent=True) as tape:
            for var in model_params:
                tape.watch(var)
            loss, prediction = model.run(x_training_sample, x_training_sample, x_training_index, y_training_sample)

        dw_list = [param[0] for param in model_params]
        db_list = [param[1] for param in model_params]
        gradients = tape.gradient(loss, dw_list + db_list)

        dw_list_cpy = [param[0] for param in model_params_cpy]
        db_list_cpy = [param[1] for param in model_params_cpy]
        self.optimizer.apply_gradients(zip(gradients, dw_list_cpy + db_list_cpy))

        with tf.GradientTape(persistent=True) as meta_tape:
            for var in model_params_cpy:
                meta_tape.watch(var)
            meta_loss, _ = model_cpy.run(x_meta_training_sample, x_meta_training_sample, x_meta_training_index, y_meta_training_sample)
        meta_gradients = meta_tape.gradient(meta_loss, dw_list_cpy + db_list_cpy)

        gradient_sums = []
        for index in range(len(gradients)):
            gradient_sums.append(gradients[index] + self.beta * meta_gradients[index])

        self.meta_optimizer.apply_gradients(zip(gradient_sums, dw_list + db_list))

        layer_loss_dict = Procedure.extract_loss_per_layer(self.layer)
        accuracy = Procedure.calc_accuracy(self.layer, self.layer.predict(x_training_sample), y_training_sample)
        self._on_training_step_completed(loss, accuracy, layer_loss_dict)

        del tape
        del meta_tape

    def _sync(self):
        model_params = []
        model_params_cpy = []

        current_layer = self.layer
        while current_layer is not None:
            model_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        current_layer = self.layer_cpy
        while current_layer is not None:
            model_params_cpy.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        for index in range(len(model_params)):
            model_params_cpy[index][0].assign(model_params[index][0])
            model_params_cpy[index][1].assign(model_params[index][1])

    def _copy_model(self):
        layer_arr: List[Layer] = []
        cur_layer: Layer = self.layer
        while cur_layer is not None:
            layer_arr.append(cur_layer)
            cur_layer = cur_layer.successor
        model_cpy = list(map(lambda layer: layer.copy(), layer_arr))
        return Procedure.compile_model(model_cpy)

    def _merge_training_data(self):
        training_data_list = []
        meta_training_data_list = []
        while len(training_data_list) == 0 or len(meta_training_data_list) == 0:
            for name in self.training_data_dict:
                value = np.random.randint(0, 2, size=1)[0]
                rand = int(value)
                if rand == 0:
                    training_data_list.append(self.training_data_dict[name])
                else:
                    meta_training_data_list.append(self.training_data_dict[name])
        x_training_data, y_training_data = AbstractSearch.stack_training_data(training_data_list)
        x_meta_training_data, y_meta_training_data = AbstractSearch.stack_training_data(meta_training_data_list)
        return x_training_data, y_training_data, x_meta_training_data, y_meta_training_data


class FishProcedure(Procedure):

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], epochs: int = 100, batch_size: int = 128, optimizer: str = "adam", early_stopping: int = 50, alpha: float = 0.01, beta: float = 0.01):
        super().__init__(layer, training_data_dict, validation_data_dict, test_data_dict, epochs, alpha, batch_size, optimizer, early_stopping=early_stopping)
        self.beta = tf.constant(beta)
        self.model_cpy = self._copy_model()

    def _execute_training_step(self) -> None:

        x_data, y_data = Procedure.merge_training_data_dict(self.training_data_dict)
        x_training_sample, y_training_sample, x_training_index = Procedure.create_batch(x_data, y_data, self.batch_size)
        pre_update_loss, prediction = self.layer.run(x_training_sample, x_training_sample, x_training_index, y_training_sample)
        pre_update_layer_loss_dict = Procedure.extract_loss_per_layer(self.layer)

        self._sync()
        model_params_cpy = []
        current_cpy_layer = self.model_cpy
        while current_cpy_layer is not None:
            model_params_cpy.append((current_cpy_layer.w1, current_cpy_layer.b1))
            current_cpy_layer = current_cpy_layer.successor

        training_data_set_names = list(self.training_data_dict)
        random.shuffle(training_data_set_names)
        for data_set_name in training_data_set_names:
            x_training_data, y_training_data = self.training_data_dict[data_set_name]
            x_training_sample, y_training_sample, x_training_index = Procedure.create_batch(x_training_data, y_training_data, self.batch_size)
            with tf.GradientTape(persistent=True) as tape:
                for var in model_params_cpy:
                    tape.watch(var[0])
                    tape.watch(var[1])
                loss, prediction = self.model_cpy.run(x_training_sample, x_training_sample, x_training_index, y_training_sample)
            dw_list_cpy = [param[0] for param in model_params_cpy]
            db_list_cpy = [param[1] for param in model_params_cpy]
            gradients = tape.gradient(loss, dw_list_cpy + db_list_cpy)

            for i in range(len(dw_list_cpy)):
                dw = gradients[i]
                db = gradients[len(dw_list_cpy) + i]
                if dw is not None:
                    dw_list_cpy[i].assign_sub(self.alpha * dw)
                if db is not None:
                    db_list_cpy[i].assign_sub(self.alpha * db)
            #self.optimizer.apply_gradients(zip(gradients, dw_list_cpy + db_list_cpy))
            del tape
        current_layer = self.layer
        current_cpy_layer = self.model_cpy
        while current_layer is not None:
            w1_update = current_layer.w1 + tf.constant(self.beta, dtype=current_layer.w1.dtype) * (current_cpy_layer.w1 - current_layer.w1)
            b1_update = current_layer.b1 + tf.constant(self.beta, dtype=current_layer.w1.dtype) * (current_cpy_layer.b1 - current_layer.b1)
            current_layer.w1.assign(w1_update)
            current_layer.b1.assign(b1_update)
            current_layer = current_layer.successor
            current_cpy_layer = current_cpy_layer.successor
        x_data, y_data = Procedure.merge_training_data_dict(self.training_data_dict)
        accuracy = Procedure.calc_accuracy(self.layer, self.layer.predict(x_data), y_data)
        self._on_training_step_completed(pre_update_loss, accuracy, pre_update_layer_loss_dict)

    def _save_parameters(self) -> List[Tuple[tf.Variable, tf.Variable]]:
        model_params = []
        current_layer = self.layer
        while current_layer is not None:
            model_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor
        return model_params

    def _copy_model(self):
        layer_arr: List[Layer] = []
        cur_layer: Layer = self.layer
        while cur_layer is not None:
            layer_arr.append(cur_layer)
            cur_layer = cur_layer.successor
        model_cpy = list(map(lambda layer: layer.copy(), layer_arr))
        return Procedure.compile_model(model_cpy)

    def _sync(self):
        model_params = []
        model_params_cpy = []

        current_layer = self.layer
        while current_layer is not None:
            model_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        current_layer = self.model_cpy
        while current_layer is not None:
            model_params_cpy.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor

        for index in range(len(model_params)):
            model_params_cpy[index][0].assign(model_params[index][0])
            model_params_cpy[index][1].assign(model_params[index][1])


class ERMGradientMatchingProcedure(Procedure):

    @staticmethod
    def remove_kth_loss_function(model: Layer, k: int):
        cur_layer: Layer = model
        layer_index = 0
        while cur_layer is not None and k > 0:
            if k == 1 and cur_layer.loss_function is not None:
                cur_loss_func = cur_layer.loss_function
                cur_layer.loss_function = None
                return cur_loss_func, layer_index
            elif cur_layer.loss_function is not None:
                k -= 1
                cur_layer = cur_layer.successor
                layer_index += 1

    @staticmethod
    def add_loss_function(model: Layer, layer_index: int, loss_function):
        cur_layer: Layer = model
        cur_layer_index = 0
        while cur_layer is not None and cur_layer_index != layer_index:
            cur_layer = cur_layer.successor
            cur_layer_index += 1
        if cur_layer_index == layer_index:
            cur_layer.loss_function = loss_function

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], epochs: int = 100, alpha: float = 1.0, batch_size: int = 128, optimizer: str = "adam", gamma=.2, beta=.2, early_stopping: int = 50):
        super().__init__(layer, training_data_dict, validation_data_dict, test_data_dict, epochs, alpha, batch_size, optimizer, early_stopping=early_stopping)
        self.gamma = tf.constant(gamma)
        self.beta = tf.constant(beta)

    def _execute_training_step(self) -> None:
        x_data, y_data = self.merge_training_data_dict(self.training_data_dict)
        x_sample, y_sample, x_index = Procedure.create_batch(x_data, y_data, self.batch_size)
        model_parameter_list = []
        current_layer = self.layer
        while current_layer is not None:
            model_parameter_list.append((current_layer.w1, current_layer.b1))
        with tf.GradientTape(persistent=False) as primary_tape:
            for w, b in model_parameter_list:
                primary_tape.watch(w)
                primary_tape.watch(b)
            grad_loss_1 = None
            grad_loss_2 = None
            with tf.GradientTape(persistent=False) as sec_tape_1:
                for w, b in model_parameter_list:
                    sec_tape_1.watch(w)
                    sec_tape_1.watch(b)
                cur_loss_func, layer_index = ERMGradientMatchingProcedure.remove_kth_loss_function(self.layer, 2)
                loss_1, _ = self.layer.run(x_sample, x_sample, x_index, y_sample)
                dw_list = [param[0] for param in model_parameter_list]
                db_list = [param[1] for param in model_parameter_list]
                grad_loss_1 = sec_tape_1.gradient(loss_1, dw_list + db_list)
                ERMGradientMatchingProcedure.add_loss_function(self.layer, layer_index, cur_loss_func)
            with tf.GradientTape(persistent=False) as sec_tape_2:
                for w, b in model_parameter_list:
                    sec_tape_2.watch(w)
                    sec_tape_2.watch(b)
                cur_loss_func, layer_index = ERMGradientMatchingProcedure.remove_kth_loss_function(self.layer, 1)
                loss_2, _ = self.layer.run(x_sample, x_sample, x_index, y_sample)
                dw_list = [param[0] for param in model_parameter_list]
                db_list = [param[1] for param in model_parameter_list]
                grad_loss_2 = sec_tape_2.gradient(loss_1, dw_list + db_list)
                ERMGradientMatchingProcedure.add_loss_function(self.layer, layer_index, cur_loss_func)
            loss_grad_dot_product = tf.tensordot(grad_loss_1, grad_loss_2, axes=1)
            classification_loss, _ = self.layer.run(x_sample, x_sample, x_index, y_sample)
            loss = classification_loss + loss_grad_dot_product
            layer_loss_dict = Procedure.extract_loss_per_layer(self.layer)
            accuracy = Procedure.calc_accuracy(self.layer, x_sample, y_sample)
            self._on_training_step_completed(loss, accuracy, layer_loss_dict)
            dw_list = [param[0] for param in model_parameter_list]
            db_list = [param[1] for param in model_parameter_list]
            gradients = sec_tape_2.gradient(loss, dw_list + db_list)
            self.optimizer.apply_gradients(zip(gradients, dw_list + db_list))
            del sec_tape_1
            del sec_tape_2
        del primary_tape


class Pipeline:

    @staticmethod
    def _average_tensors(tensors: List[List[tf.Tensor]]):
        result: List[tf.Tensor] = []
        length: int = len(tensors[0])

        for length_index in range(0, length):
            sum_tensor: tf.Tensor = tf.constant(0.0)
            counter = 0
            for sample_index in range(len(tensors)):
                if length_index < len(tensors[sample_index]):
                    sum_tensor = sum_tensor + tensors[sample_index][length_index]
                    counter += 1
            sum_tensor = tf.constant(sum_tensor.numpy() / float(counter))
            result.append(sum_tensor)
        return result

    @staticmethod
    def _average_stage_iteration_tensor_dict(stage_iteration_dict: Dict[int, Dict[int, tf.Tensor]]) -> Dict[int, tf.Tensor]:
        stage_dict: Dict[int, tf.Tensor] = dict()
        for stage in stage_iteration_dict:
            value_list = []
            for iteration in stage_iteration_dict[stage]:
                value_list.append(stage_iteration_dict[stage][iteration])
            stage_dict[stage] = Pipeline._average_tensors([value_list])[0]
        return stage_dict

    @staticmethod
    def _average_stage_iteration_name_tensor_dict(stage_iteration_dict: Dict[int, Dict[int, Dict[str, tf.Tensor]]]) -> Dict[int, Dict[str, tf.Tensor]]:
        stage_dict: Dict[int, Dict[str, tf.Tensor]] = dict()
        for stage in stage_iteration_dict.keys():
            stage_dict[stage] = dict()
            for layer_name in stage_iteration_dict[stage][0]:
                value_list = []
                for iteration in range(len(stage_iteration_dict[stage])):
                    value_list.append(stage_iteration_dict[stage][iteration][layer_name])
                stage_dict[stage][layer_name] = Pipeline._average_tensors([value_list])[0]
        return stage_dict

    @staticmethod
    def _average_stage_iteration_list_dict(stage_iteration_dict: Dict[int, Dict[int, List[tf.Tensor]]]) -> Dict[int, List[tf.Tensor]]:
        stage_dict: Dict[int, List[tf.Tensor]] = dict()
        for stage in stage_iteration_dict:
            value_list = []
            for iteration in stage_iteration_dict[stage]:
                value_list.append(stage_iteration_dict[stage][iteration])
            stage_dict[stage] = Pipeline._average_tensors(value_list)
        return stage_dict

    @staticmethod
    def _average_stage_iteration_name_list_dict(stage_iteration_name_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]]) -> Dict[int, Dict[str, List[tf.Tensor]]]:
        stage_iteration_dict: Dict[int, Dict[str, List[tf.Tensor]]] = dict()
        for stage in stage_iteration_name_dict:
            stage_iteration_dict[stage] = dict()
            for layer_name in stage_iteration_name_dict[stage][0]:
                value_list = []
                for iteration in range(len(stage_iteration_name_dict[stage])):
                    value_list.append(stage_iteration_name_dict[stage][iteration][layer_name])
                stage_iteration_dict[stage][layer_name] = Pipeline._average_tensors(value_list)
        return stage_iteration_dict

    def __init__(self, initial_procedure_builder: Callable[[], Procedure], subsequent_procedure_builder_list: List[Callable[[Procedure], Procedure]], iterations: int = 1):

        self.initial_procedure_builder = initial_procedure_builder
        self.subsequent_procedure_builder_list = subsequent_procedure_builder_list
        self.iterations = iterations
        self.stage_iteration_epoch_dict: Dict[int, Dict[int, int]] = {}
        self.stage_iteration_training_loss_list: Dict[int, Dict[int, List[tf.Tensor]]] = {}
        self.stage_iteration_layer_name_training_loss_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]] = {}
        self.stage_iteration_validation_loss_list: Dict[int, Dict[int, List[tf.Tensor]]] = {}
        self.stage_iteration_layer_name_validation_loss_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]] = {}
        self.stage_iteration_data_name_validation_loss_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]] = {}
        self.stage_iteration_training_accuracy_list: Dict[int, Dict[int, List[tf.Tensor]]] = {}
        self.stage_iteration_validation_accuracy_list: Dict[int, Dict[int, List[tf.Tensor]]] = {}
        self.stage_iteration_data_name_validation_accuracy_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]] = {}
        self.stage_iteration_average_test_accuracy: Dict[int, Dict[int, Optional[tf.Tensor]]] = {}
        self.stage_iteration_test_data_name_accuracy: Dict[int, Dict[int, Dict[str, tf.Tensor]]] = {}

        self.stage_iteration_test_data_name_class_accuracy_dict = dict()

        self.stage_iteration_loss_fig_dict: Dict[int, Dict[int, Figure]] = {}
        self.stage_iteration_ds_loss_fig_dict: Dict[int, Dict[int, Figure]] = {}
        self.stage_iteration_accuracy_fig_dict: Dict[int, Dict[int, Figure]] = {}

        self.stage_iteration_training_data_dict = {}
        self.stage_iteration_validation_data_dict = {}
        self.stage_iteration_test_data_dict = {}

        self.stage_iteration_test_data_dict = {}
        self.stage_iteration_layer_param_dict = {}

        self.stage_average_test_accuracy = {}
        self.stage_test_data_name_accuracy = {}
        self.final_average_validation_accuracy = None
        self.final_average_test_accuracy = None

        for cur_stage in range(len(subsequent_procedure_builder_list) + 1):
            self.stage_iteration_epoch_dict[cur_stage] = {}
            self.stage_iteration_training_loss_list[cur_stage] = {}
            self.stage_iteration_layer_name_training_loss_dict[cur_stage] = {}
            self.stage_iteration_validation_loss_list[cur_stage] = {}
            self.stage_iteration_layer_name_validation_loss_dict[cur_stage] = {}
            self.stage_iteration_data_name_validation_loss_dict[cur_stage] = {}
            self.stage_iteration_training_accuracy_list[cur_stage] = {}
            self.stage_iteration_validation_accuracy_list[cur_stage] = {}
            self.stage_iteration_data_name_validation_accuracy_dict[cur_stage] = {}
            self.stage_iteration_average_test_accuracy[cur_stage] = {}
            self.stage_iteration_test_data_name_accuracy[cur_stage] = {}

            self.stage_iteration_test_data_name_class_accuracy_dict[cur_stage] = {}

            self.stage_iteration_loss_fig_dict[cur_stage] = {}
            self.stage_iteration_ds_loss_fig_dict[cur_stage] = {}
            self.stage_iteration_accuracy_fig_dict[cur_stage] = {}

            self.stage_iteration_training_data_dict[cur_stage] = {}
            self.stage_iteration_validation_data_dict[cur_stage] = {}
            self.stage_iteration_test_data_dict[cur_stage] = {}
            self.stage_iteration_layer_param_dict[cur_stage] = {}
            for cur_iteration in range(iterations):
                self.stage_iteration_epoch_dict[cur_stage][cur_iteration] = {}
                self.stage_iteration_training_loss_list[cur_stage][cur_iteration]= {}
                self.stage_iteration_layer_name_training_loss_dict[cur_stage][cur_iteration] = {}
                self.stage_iteration_validation_loss_list[cur_stage][cur_iteration] = {}
                self.stage_iteration_layer_name_validation_loss_dict[cur_stage][cur_iteration] = {}
                self.stage_iteration_data_name_validation_loss_dict[cur_stage][cur_iteration] = {}
                self.stage_iteration_training_accuracy_list[cur_stage][cur_iteration] = {}
                self.stage_iteration_validation_accuracy_list[cur_stage][cur_iteration] = {}
                self.stage_iteration_data_name_validation_accuracy_dict[cur_stage][cur_iteration] = {}
                self.stage_iteration_average_test_accuracy[cur_stage][cur_iteration] = {}
                self.stage_iteration_test_data_name_accuracy[cur_stage][cur_iteration] = {}
                self.stage_iteration_test_data_name_class_accuracy_dict[cur_stage][cur_iteration] = {}

        self.stage_epoch_dict = None
        self.stage_training_loss_list = None
        self.stage_layer_name_training_loss_dict = None
        self.stage_validation_loss_list = None
        self.stage_layer_name_validation_loss_dict = None
        self.stage_data_name_validation_loss_dict = None
        self.stage_training_accuracy_list = None
        self.stage_validation_accuracy_list = None
        self.stage_data_name_validation_accuracy_dict = None

    def run(self):
        for cur_iteration in range(self.iterations):
            cur_procedure: Optional[Procedure] = self.initial_procedure_builder()
            subsequent_procedure_builder_index = 0
            stage = 0
            while cur_procedure is not None:
                cur_procedure.train()
                self._extract_stats(cur_iteration, cur_procedure, stage)
                stage += 1
                if subsequent_procedure_builder_index < len(self.subsequent_procedure_builder_list):
                    cur_procedure = self.subsequent_procedure_builder_list[subsequent_procedure_builder_index](cur_procedure)
                    subsequent_procedure_builder_index += 1
                else:
                    cur_procedure = None
        self._average_stats()

    def _extract_stats(self, iteration: int, procedure: Procedure, stage: int):
        self.stage_iteration_epoch_dict[stage][iteration] = procedure.epochs
        self.stage_iteration_training_loss_list[stage][iteration] = procedure.training_loss_list
        self.stage_iteration_layer_name_training_loss_dict[stage][iteration] = procedure.layer_name_training_loss_dict
        self.stage_iteration_validation_loss_list[stage][iteration] = procedure.validation_loss_list
        self.stage_iteration_layer_name_validation_loss_dict[stage][iteration] = procedure.layer_name_validation_loss_dict
        self.stage_iteration_data_name_validation_loss_dict[stage][iteration] = procedure.data_name_validation_loss_dict
        self.stage_iteration_training_accuracy_list[stage][iteration] = procedure.training_accuracy_list
        self.stage_iteration_validation_accuracy_list[stage][iteration] = procedure.validation_accuracy_list
        self.stage_iteration_data_name_validation_accuracy_dict[stage][iteration] = procedure.data_name_validation_accuracy_dict
        self.stage_iteration_average_test_accuracy[stage][iteration] = procedure.average_test_accuracy
        self.stage_iteration_test_data_name_accuracy[stage][iteration] = procedure.test_data_name_accuracy
        self.stage_iteration_test_data_name_class_accuracy_dict[stage][iteration] = procedure.test_data_name_class_accuracy_dict

        self.stage_iteration_loss_fig_dict[stage][iteration] = procedure.loss_fig
        self.stage_iteration_ds_loss_fig_dict[stage][iteration] = procedure.data_set_loss_fig
        self.stage_iteration_accuracy_fig_dict[stage][iteration] = procedure.accuracy_fig

        self.stage_iteration_training_data_dict[stage][iteration] = procedure.training_data_dict
        self.stage_iteration_validation_data_dict[stage][iteration] = procedure.validation_data_dict
        self.stage_iteration_test_data_dict[stage][iteration] = procedure.test_data_dict

        layer_param_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = dict()
        cur_layer = procedure.layer
        while cur_layer is not None:
            w_cpy = tf.Variable(cur_layer.w1.numpy())
            b_cpy = tf.Variable(cur_layer.b1.numpy())
            layer_param_dict[cur_layer.layer_name] = (w_cpy, b_cpy)
            cur_layer = cur_layer.successor

        self.stage_iteration_layer_param_dict[stage][iteration] = layer_param_dict

    def _average_stats(self):
        self.stage_epoch_dict = dict()
        for stage in self.stage_iteration_epoch_dict:
            epoch_count: int = 0
            for iteration in self.stage_iteration_epoch_dict[stage]:
                cur_count = self.stage_iteration_epoch_dict[stage][iteration]
                epoch_count = epoch_count + cur_count
            self.stage_epoch_dict[stage] = epoch_count / len(self.stage_iteration_epoch_dict[stage])
        self.stage_training_loss_list = Pipeline._average_stage_iteration_list_dict(self.stage_iteration_training_loss_list)
        self.stage_layer_name_training_loss_dict = Pipeline._average_stage_iteration_name_list_dict(self.stage_iteration_layer_name_training_loss_dict)
        self.stage_validation_loss_list = Pipeline._average_stage_iteration_list_dict(self.stage_iteration_validation_loss_list)
        self.stage_layer_name_validation_loss_dict = Pipeline._average_stage_iteration_name_list_dict(self.stage_iteration_layer_name_validation_loss_dict)
        self.stage_data_name_validation_loss_dict = Pipeline._average_stage_iteration_name_list_dict(self.stage_iteration_data_name_validation_loss_dict)
        self.stage_training_accuracy_list = Pipeline._average_stage_iteration_list_dict(self.stage_iteration_training_accuracy_list)
        self.stage_validation_accuracy_list = Pipeline._average_stage_iteration_list_dict(self.stage_iteration_validation_accuracy_list)
        self.stage_data_name_validation_accuracy_dict = Pipeline._average_stage_iteration_name_list_dict(self.stage_iteration_data_name_validation_accuracy_dict)
        self.stage_average_test_accuracy = Pipeline._average_stage_iteration_tensor_dict(self.stage_iteration_average_test_accuracy)
        self.stage_test_data_name_accuracy = Pipeline._average_stage_iteration_name_tensor_dict(self.stage_iteration_test_data_name_accuracy)
        self.final_average_test_accuracy = self.stage_average_test_accuracy[len(self.stage_average_test_accuracy) - 1]


class Configuration:

    def __init__(self, neuron_count_list: List[int], activation_func_list: List[str], loss_func_list: List[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]], forwarding_list: List[bool], binary_layer_list: List[bool]):
        self.neuron_count_list: List[int] = neuron_count_list
        self.activation_func_list: List[str] = activation_func_list
        self.loss_funct_list: List[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = loss_func_list
        self.forwarding_list: List[bool] = forwarding_list
        self.binary_layer_list: List[bool] = binary_layer_list
    def as_string(self):
        return str(self.neuron_count_list) + "_" + str(self.activation_func_list) + "_" + str(self.loss_funct_list)


class Hyperparameters:

    def __init__(self, alpha, beta, gamma, epochs, batch_size, optimizer, early_stopping, sample_size, iterations):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.early_stopping = early_stopping
        self.sample_size = sample_size
        self.iterations = iterations

    def as_string(self):
        return "_" + str(self.alpha) \
            + "_" + str(self.beta) \
            + "_" + str(self.gamma) \
            + "_" + str(self.epochs) \
            + "_" + str(self.batch_size) \
            + "_" + str(self.optimizer) \
            + "_" + str(self.early_stopping) \
            + "_" + str(self.sample_size) \
            + "_" + str(self.iterations) \

class AbstractSearch:

    @staticmethod
    def stack_training_data(training_data_list: List[Tuple[tf.Tensor, tf.Tensor]]) -> Tuple[tf.Tensor, tf.Tensor]:
        x_data = tf.constant(training_data_list[0][0])
        y_data = tf.constant(training_data_list[0][1])
        for index in range(1, len(training_data_list)):
            cur_x_data, cur_y_data = training_data_list[index]
            x_data = tf.concat([cur_x_data, x_data], axis=0)
            y_data = tf.concat([cur_y_data, y_data], axis=0)
        return x_data, y_data

    @staticmethod
    def create_dir_if_not_existent(directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def _build_model(predictor_count: int, class_count, configuration: Configuration) -> Layer:
        print(configuration.as_string())
        model = []
        cur_in = predictor_count
        configuration.neuron_count_list[0] = predictor_count
        index = 0
        for neuron_count, activation_func, loss_func, forwarding, is_binary in zip(configuration.neuron_count_list, configuration.activation_func_list, configuration.loss_funct_list, configuration.forwarding_list, configuration.binary_layer_list):
            if index == len(configuration.neuron_count_list) - 1:
                neuron_count = class_count
                configuration.neuron_count_list[index] = class_count
            layer: Layer = Layer(
                name="layer_" + str(index),
                input_dim=cur_in,
                output_dim=neuron_count,
                forward=forwarding,
                loss_function=loss_func,
                loss_weight=1.0,
                activation_function_name=activation_func
            )
            index = index + 1
            cur_in = neuron_count
            model.append(layer)
        return Procedure.compile_model(model)

    @staticmethod
    def create_training_data(
            training_data_param_set: List[sampling.ParameterSet],
            validation_data_param_set: List[sampling.ParameterSet],
            test_data_param_set: List[sampling.ParameterSet],
            env: Environment,
            sort: Callable[[Environment], List[int]],
            sample_size: int) -> Tuple[Dict[sampling.ParameterSet, Tuple[tf.Tensor, tf.Tensor]], Dict[sampling.ParameterSet, Tuple[tf.Tensor, tf.Tensor]], Dict[sampling.ParameterSet, Tuple[tf.Tensor, tf.Tensor]], int]:

        if len(training_data_param_set) == 0 or len(validation_data_param_set) == 0 or len(test_data_param_set) == 0:
            raise ValueError("Training, validation and test data parameters must be specified.")

        training_data_param_index_list: List[Tuple[sampling.ParameterSet, int, int]] = []
        validation_data_param_index_list: List[Tuple[sampling.ParameterSet, int, int]] = []
        test_data_param_index_list: List[Tuple[sampling.ParameterSet, int, int]] = []
        # Sampling - By merging sampled data, we ensure a consistent one-hot-encoding across different parameters.
        mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(training_data_param_set[0], sort, env)
        mcs.init()
        x_np_data, y_np_data = mcs.sample(sample_size)
        cur_end_index: int = x_np_data.shape[0] - 1
        training_data_param_index_list.append((training_data_param_set[0], 0, x_np_data.shape[0]))

        for index in range(1, len(training_data_param_set)):
            cur_param_set: sampling.ParameterSet = training_data_param_set[index]
            mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(cur_param_set, sort, env)
            mcs.init()
            cur_x_np_data, cur_y_np_data = mcs.sample(sample_size)
            cur_begin_index = cur_end_index + 1
            cur_end_index = cur_end_index + cur_x_np_data.shape[0]
            x_np_data = np.vstack((x_np_data, cur_x_np_data))
            y_np_data = np.vstack((y_np_data, cur_y_np_data))
            training_data_param_index_list.append((cur_param_set, cur_begin_index, cur_end_index))
        for index in range(0, len(validation_data_param_set)):
            cur_param_set: sampling.ParameterSet = validation_data_param_set[index]
            mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(cur_param_set, sort, env)
            mcs.init()
            cur_x_np_data, cur_y_np_data = mcs.sample(sample_size)
            cur_begin_index = cur_end_index + 1
            cur_end_index = cur_end_index + cur_x_np_data.shape[0]
            x_np_data = np.vstack((x_np_data, cur_x_np_data))
            y_np_data = np.vstack((y_np_data, cur_y_np_data))
            validation_data_param_index_list.append((cur_param_set, cur_begin_index, cur_end_index))
        for index in range(0, len(test_data_param_set)):
            cur_param_set: sampling.ParameterSet = test_data_param_set[index]
            mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(cur_param_set, sort, env)
            mcs.init()
            cur_x_np_data, cur_y_np_data = mcs.sample(sample_size)
            cur_begin_index = cur_end_index + 1
            cur_end_index = cur_end_index + cur_x_np_data.shape[0]
            x_np_data = np.vstack((x_np_data, cur_x_np_data))
            y_np_data = np.vstack((y_np_data, cur_y_np_data))
            test_data_param_index_list.append((cur_param_set, cur_begin_index, cur_end_index))

        # Scaling
        x_np_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_np_data)
        # Conversion to TensorFlow Tensors
        x_tf_data, y_tf_data = tf.cast(tf.convert_to_tensor(x_np_data), dtype=tf.float32), tf.cast(tf.convert_to_tensor(y_np_data), dtype=tf.float32)
        # Swapping first and last column so that the state is passed through the first layers.
        x_tf_data = tf.gather(x_tf_data, list(range(x_tf_data.shape[1] - 1, -1, -1)), axis=1)
        # One-Hot-Encoding
        y_tf_data_reshaped = tf.reshape(y_tf_data, shape=(y_tf_data.shape[0],))
        unique_values, unique_encoding = tf.unique(y_tf_data_reshaped)
        y_tf_one_hot_data = tf.one_hot(unique_encoding, depth=unique_values.shape[0])
        # Returns the training data - x_tf_data and y_tf_one_hot_data - and the class count.
        training_param_set_data_dict = {}
        for param_set, begin_index, end_index in training_data_param_index_list:
            cur_x_tf_data = x_tf_data[begin_index: end_index + 1]
            cur_y_tf_data = y_tf_one_hot_data[begin_index: end_index + 1]
            training_param_set_data_dict[param_set] = (cur_x_tf_data, cur_y_tf_data)

        validation_param_set_data_dict = {}
        for param_set, begin_index, end_index in validation_data_param_index_list:
            cur_x_tf_data = x_tf_data[begin_index: end_index + 1]
            cur_y_tf_data = y_tf_one_hot_data[begin_index: end_index + 1]
            validation_param_set_data_dict[param_set] = (cur_x_tf_data, cur_y_tf_data)

        test_param_set_data_dict = {}
        for param_set, begin_index, end_index in test_data_param_index_list:
            cur_x_tf_data = x_tf_data[begin_index: end_index + 1]
            cur_y_tf_data = y_tf_one_hot_data[begin_index: end_index + 1]
            test_param_set_data_dict[param_set] = (cur_x_tf_data, cur_y_tf_data)

        return training_param_set_data_dict, validation_param_set_data_dict, test_param_set_data_dict, unique_values.shape[0]

    @staticmethod
    def generate_data_dict(hyperparameters: Hyperparameters, training_parameter_set_list: List[ParameterSet], validation_parameter_set_list: List[ParameterSet], test_parameter_set_list: List[ParameterSet], env: Environment, sort: Callable[[Environment], List[int]],) -> Tuple[Dict[str, Tuple[tf.Tensor, tf.Tensor]], Dict[str, Tuple[tf.Tensor, tf.Tensor]], Dict[str, Tuple[tf.Tensor, tf.Tensor]], int]:

        def param_set_to_str(param_set: ParameterSet):
            return "p_mu=" + str(param_set.problem_mu) + " p_sigma=" + str(param_set.problem_sigma) + " s_mu" + str(param_set.problem_size_mu) + " s_sigma=" + str(param_set.problem_size_sigma)

        training_param_set_data_dict, validation_param_set_data_dict, test_param_set_data_dict, class_count = AbstractSearch.create_training_data(training_parameter_set_list, validation_parameter_set_list, test_parameter_set_list, env, sort, hyperparameters.sample_size)

        training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = {}
        for param_set in training_param_set_data_dict:
            training_data_dict[param_set_to_str(param_set)] = training_param_set_data_dict[param_set]
        validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = {}
        for param_set in validation_param_set_data_dict:
            validation_data_dict[param_set_to_str(param_set)] = validation_param_set_data_dict[param_set]
        test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = {}
        for param_set in test_param_set_data_dict:
            test_data_dict[param_set_to_str(param_set)] = test_param_set_data_dict[param_set]

        return training_data_dict, validation_data_dict, test_data_dict, class_count

    def _build_search_space(self) -> List[Tuple[Configuration, Hyperparameters]]:
        pass

    def __init__(self, training_parameter_list: List[ParameterSet], validation_parameter_list: List[ParameterSet], test_parameter_list: List[ParameterSet],  base_dir: str, env: Environment, sort: Callable[[Environment], List[int]]):
        self.training_parameter_list = training_parameter_list
        self.validation_parameter_list = validation_parameter_list
        self.test_parameter_list = test_parameter_list
        self.base_dir = "./data_experiment_2/" + base_dir
        self.env: Environment = env
        self.sort: Callable[[Environment], List[int]] = sort

    def evaluate(self, configuration) -> float:
        pipeline: Pipeline = self._build_pipeline(configuration, self.env, self.sort)
        pipeline.run()
        self.save_result(configuration, pipeline)
        plt.close('all')
        accuracy_list = pipeline.stage_validation_accuracy_list[len(pipeline.stage_validation_accuracy_list) - 1]
        validation_accuracy_tf = accuracy_list[len(accuracy_list) - 1]
        validation_accuracy_np = validation_accuracy_tf.numpy()
        return float(validation_accuracy_np)

    def search(self):
        for configuration in self._build_search_space():
            self.evaluate(configuration)

    def _build_pipeline(self, configuration: Tuple[Configuration, Hyperparameters], env: Environment, sort: Callable[[Environment], List[int]]) -> Pipeline:
        conf, hyperparameters = configuration
        training_data, validation_data, test_data, class_count = AbstractSearch.generate_data_dict(hyperparameters, self.training_parameter_list, self.validation_parameter_list, self.test_parameter_list, env, sort)
        predictor_count = list(training_data.values())[0][0].shape[1]
        model: Layer = AbstractSearch._build_model(predictor_count, class_count, conf)
        initial_procedure_builder: Callable[[], Procedure] = self._create_initial_procedure_builder(model, hyperparameters, training_data, validation_data, test_data)
        subsequent_procedure_builders: List[Callable[[Procedure], Procedure]] = self._create_subsequent_procedure_builders(hyperparameters, training_data, validation_data, test_data)
        return Pipeline(initial_procedure_builder, subsequent_procedure_builders, iterations=hyperparameters.iterations)

    def _create_initial_procedure_builder(self, model: Layer, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> Callable[[], Procedure]:
        pass

    def _create_subsequent_procedure_builders(self, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> List[Callable[[Procedure], Procedure]]:
        pass

    @staticmethod
    def map_list(tensor_list: List[tf.Tensor]) -> List[float]:
        return [float(t.numpy()) for t in tensor_list]

    @staticmethod
    def map_list_dict(tensor_list_dict: Dict[str, List[tf.Tensor]]) -> Dict[str, List[float]]:
        result_dict: Dict[str, List[float]] = dict()
        for name in tensor_list_dict:
            result_dict[name] = AbstractSearch.map_list(tensor_list_dict[name])
        return result_dict

    @staticmethod
    def plot_list(x_label: str, y_label: str, title: str, label: str, tensor_list: List[tf.Tensor]) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.plot(list(range(1, len(tensor_list) + 1)), tensor_list, label=label, marker='o')
        ax.legend()
        return fig, ax

    @staticmethod
    def plot_dict_list(x_label: str, y_label: str, title: str, tensor_list_dict: Dict[str, List[tf.Tensor]]) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        for name in tensor_list_dict:
            value_list = tensor_list_dict[name]
            ax.plot(list(range(1, len(value_list) + 1)), value_list, label=name, marker='o')
        ax.legend()
        return fig, ax


    @staticmethod
    def clean_function_references(s):
        return re.sub(r"<function\s+([\w_]+)\s+at\s+0x[0-9a-fA-F]+>", r"\1", s)

    def save_result(self, configuration: Tuple[Configuration, Hyperparameters], pipeline: Pipeline) -> None:
        cur_dir: str = AbstractSearch.create_dir_if_not_existent(self.base_dir + "/" + str(datetime.now().strftime("%m-%d %H:%M")) + "_" + AbstractSearch.clean_function_references(configuration[0].as_string()) + "/")
        with open(cur_dir + "hyperparameters.txt", "w") as f:
            f.write("alpha: " + str(configuration[1].alpha) + "\n")
            f.write("beta: " + str(configuration[1].beta) + "\n")
            f.write("gamma: " + str(configuration[1].gamma) + "\n")
            f.write("iterations: " + str(configuration[1].iterations) + "\n")
            f.write("sample_size: " + str(configuration[1].sample_size) + "\n")
            f.write("epochs: " + str(configuration[1].epochs) + "\n")
            f.write("batch_size: " + str(configuration[1].batch_size) + "\n")
            f.write("optimizer: " + str(configuration[1].optimizer) + "\n")
            f.write("early_stopping: " + str(configuration[1].early_stopping) + "\n")
        with open(cur_dir + "configuration.txt", "w") as f:
            f.write("neuron_count_list: " + str(configuration[0].neuron_count_list) + "\n")
            f.write("activation_func_list: " + str(configuration[0].activation_func_list) + "\n")
            f.write("forwarding_list: " + str(configuration[0].forwarding_list) + "\n")
            f.write("loss_funct_list: " + str(configuration[0].loss_funct_list) + "\n")
            f.write("binary_layer_list: " + str(configuration[0].binary_layer_list) + "\n")
        for stage in range(0, len(pipeline.subsequent_procedure_builder_list) + 1):
            stage_dir: str = AbstractSearch.create_dir_if_not_existent(cur_dir + "stage_" + str(stage) + "/")
            for iteration in range(0, pipeline.iterations):
                iteration_dir = stage_dir + "iteration_" + str(iteration) + "/"
                iteration_dir = AbstractSearch.create_dir_if_not_existent(iteration_dir)
                with open(iteration_dir + "data_name_validation_loss.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list_dict(pipeline.stage_iteration_data_name_validation_loss_dict[stage][iteration])))
                with open(iteration_dir + "training_accuracy.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list(pipeline.stage_iteration_training_accuracy_list[stage][iteration])))
                with open(iteration_dir + "validation_accuracy.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list(pipeline.stage_iteration_validation_accuracy_list[stage][iteration])))
                with open(iteration_dir + "data_name_validation_accuracy.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list_dict(pipeline.stage_iteration_data_name_validation_accuracy_dict[stage][iteration])))
                with open(iteration_dir + "avg_test_accuracy.txt", "w") as f:
                    f.write(str((pipeline.stage_iteration_average_test_accuracy[stage][iteration].numpy())))
                with open(iteration_dir + "data_name_test_accuracy.txt", "a") as f:
                    for data_name in pipeline.stage_iteration_test_data_name_accuracy[stage][iteration]:
                        f.write(data_name + ": " + str(pipeline.stage_iteration_test_data_name_accuracy[stage][iteration][data_name].numpy()))

                with open(iteration_dir + "test_data_name_class_accuracy_dict.txt", "a") as f:
                    for data_name in pipeline.stage_iteration_test_data_name_class_accuracy_dict[stage][stage]:
                        f.write("---" + data_name + "---\n")
                        for class_name in pipeline.stage_iteration_test_data_name_class_accuracy_dict[stage][stage][data_name]:
                            f.write(str(class_name) + ": " + str(pipeline.stage_iteration_test_data_name_class_accuracy_dict[stage][stage][data_name][class_name]) +"\n")

                pipeline.stage_iteration_loss_fig_dict[stage][iteration].savefig(iteration_dir + "loss.png", dpi=600)
                pipeline.stage_iteration_ds_loss_fig_dict[stage][iteration].savefig(iteration_dir + "ds_loss.png", dpi=600)
                pipeline.stage_iteration_accuracy_fig_dict[stage][iteration].savefig(iteration_dir + "accuracy.png", dpi=600)
                for training_data_name in pipeline.stage_iteration_training_data_dict[stage][iteration]:
                    training_data_dir = AbstractSearch.create_dir_if_not_existent(iteration_dir + "/data/training/")
                    training_data_dir = training_data_dir + training_data_name
                    predictors, responses = pipeline.stage_iteration_training_data_dict[stage][iteration][training_data_name]
                    np.save(training_data_dir + "_predictors.npy", predictors.numpy())
                    np.save(training_data_dir + "_responses.npy", responses.numpy())
                for validation_data_name in pipeline.stage_iteration_validation_data_dict[stage][iteration]:
                    val_data_dir = AbstractSearch.create_dir_if_not_existent(iteration_dir + "/data/validation/")
                    val_data_dir = val_data_dir  + validation_data_name
                    predictors, responses = pipeline.stage_iteration_validation_data_dict[stage][iteration][validation_data_name]
                    np.save(val_data_dir + "_predictors.npy", predictors.numpy())
                    np.save(val_data_dir + "_responses.npy", responses.numpy())
                for test_data_name in pipeline.stage_iteration_test_data_dict[stage][iteration]:
                    test_data_dir = AbstractSearch.create_dir_if_not_existent(iteration_dir + "/data/test/")
                    test_data_dir = test_data_dir + test_data_name
                    predictors, responses = pipeline.stage_iteration_test_data_dict[stage][iteration][test_data_name]
                    np.save(test_data_dir + "_predictors.npy", predictors.numpy())
                    np.save(test_data_dir + "_responses.npy", responses.numpy())
                for layer_name in pipeline.stage_iteration_layer_param_dict[stage][iteration]:
                    weights_dir = AbstractSearch.create_dir_if_not_existent(iteration_dir + "/weights/")
                    w, b = pipeline.stage_iteration_layer_param_dict[stage][iteration][layer_name]
                    checkpoint = tf.train.Checkpoint(w1=w, b1=b)
                    checkpoint.write(weights_dir + "layer_" + layer_name + ".ckpt")
            with open(stage_dir + "epochs.txt", "w") as f:
                f.write(str(pipeline.stage_epoch_dict[stage]))
            loss_dir = AbstractSearch.create_dir_if_not_existent(stage_dir + "loss/")
            with open(loss_dir + "training_loss.txt", "w") as f:
                f.write(str(AbstractSearch.map_list(pipeline.stage_training_loss_list[stage])))
                fig, ax = AbstractSearch.plot_list("Epochs", "Loss", "Training Loss", "", pipeline.stage_training_loss_list[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(loss_dir + "training_loss.jpg", dpi=600)
            with open(loss_dir + "layer_name_training_loss.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_layer_name_training_loss_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Layerwise Training Loss", pipeline.stage_layer_name_training_loss_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(loss_dir + "layer_name_training_loss.jpg", dpi=600)
            with open(loss_dir + "validation_loss.txt", "w") as f:
                f.write(str(AbstractSearch.map_list(pipeline.stage_validation_loss_list[stage])))
                fig, ax = AbstractSearch.plot_list("Epochs", "Loss", "Validation Loss", "", pipeline.stage_validation_loss_list[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(loss_dir + "validation_loss.jpg", dpi=600)
            with open(loss_dir + "layer_name_validation_loss.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_layer_name_validation_loss_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Layerwise Validation Loss",  pipeline.stage_layer_name_validation_loss_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(loss_dir + "layer_name_validation_loss.jpg", dpi=600)
            with open(loss_dir + "data_name_validation_loss.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_data_name_validation_loss_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Validation Loss for Data Sets",  pipeline.stage_data_name_validation_loss_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(loss_dir + "data_name_validation_loss.jpg", dpi=600)

            accuracy_dir = AbstractSearch.create_dir_if_not_existent(stage_dir + "accuracy/")
            with open(accuracy_dir + "training_accuracy.txt", "w") as f:
                f.write(str(AbstractSearch.map_list(pipeline.stage_training_accuracy_list[stage])))
                fig, ax = AbstractSearch.plot_list("Epochs", "Accuracy", "Training Accuracy", "", pipeline.stage_training_accuracy_list[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(accuracy_dir + "training_accuracy.jpg", dpi=600)
            with open(accuracy_dir + "validation_accuracy.txt", "w") as f:
                f.write(str(AbstractSearch.map_list(pipeline.stage_validation_accuracy_list[stage])))
                fig, ax = AbstractSearch.plot_list("Epochs", "Accuracy", "Validation Accuracy", "",  pipeline.stage_validation_accuracy_list[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(accuracy_dir + "validation_accuracy.jpg", dpi=600)
            with open(accuracy_dir + "data_name_validation_accuracy.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_data_name_validation_accuracy_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Accuracy", "Validation Accuracy for Data Sets",  pipeline.stage_data_name_validation_accuracy_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(accuracy_dir + "data_name_validation_accuracy.jpg", dpi=600)
            with open(accuracy_dir + "test_accuracy.txt", "w") as f:
                f.write(str(pipeline.stage_average_test_accuracy[stage]))
            with open(accuracy_dir + "data_name_avg_test_accuracy.txt", "w") as f:
                for data_name in pipeline.stage_test_data_name_accuracy[stage]:
                    f.write(data_name + ": " + str(pipeline.stage_test_data_name_accuracy[stage][data_name]) + "\n")


class BayesianSearch(AbstractSearch):

    def __init__(self, training_parameter_list: List[ParameterSet], validation_parameter_list: List[ParameterSet], test_parameter_list: List[ParameterSet], base_dir: str, max_call_count: int,  env: Environment, sort: Callable[[Environment], List[int]]):
        super().__init__(training_parameter_list, validation_parameter_list, test_parameter_list, base_dir, env, sort)
        self.checkpoint_path = base_dir + "/bayes_opt_checkpoint.pkl"
        self.max_call_count = max_call_count

    def objective_function(self, params):
        configuration = pickle.loads(base64.b64decode(params[0].encode('utf-8')))
        alpha = params[1]
        beta = params[2]
        gamma = params[3]
        epochs = params[4]
        batch_size = params[5]
        optimizer = params[6]
        early_stopping = params[7]
        sample_size = params[8]
        iterations = params[9]
        configuration.loss_funct_list = list(map(self.decode, configuration.loss_funct_list))

        hyperparameter = Hyperparameters(alpha, beta, gamma, epochs, batch_size, optimizer, early_stopping, sample_size, iterations)

        return self.evaluate((configuration, hyperparameter))

    def encode(self, ref):
        if ref == binary_representation_loss:
            return "binary"
        elif ref == categorical_cross_entropy_loss:
            return "categorical"

    def decode(self, ref):
        if ref == "binary":
            return binary_representation_loss
        elif ref == "categorical":
            return categorical_cross_entropy_loss

    def search(self):

        configuration_list = []
        alpha_list = []
        beta_list = []
        gamma_list = []
        epochs_list = []
        batch_size_list = []
        optimizer_list = []
        early_stopping_list = []
        sample_size_list = []
        iterations_list = []

        for conf, params in self._build_search_space():
            conf.loss_funct_list = map(self.encode, conf.loss_funct_list)
            configuration_list.append(base64.b64encode(pickle.dumps(conf)).decode('utf-8'))
            if params.alpha not in alpha_list:
                alpha_list.append(params.alpha)
            if params.beta not in beta_list:
                beta_list.append(params.beta)
            if params.gamma not in gamma_list:
                gamma_list.append(params.gamma)
            if params.epochs not in epochs_list:
                epochs_list.append(params.epochs)
            if params.batch_size not in batch_size_list:
                batch_size_list.append(params.batch_size)
            if params.optimizer not in optimizer_list:
                optimizer_list.append(params.optimizer)
            if params.early_stopping not in early_stopping_list:
                early_stopping_list.append(params.early_stopping)
            if params.sample_size not in sample_size_list:
                sample_size_list.append(params.sample_size)
            if params.iterations not in iterations_list:
                iterations_list.append(params.iterations)

        search_space = [
            Categorical(configuration_list, name="configuration"),
            Categorical(alpha_list, name="alpha"),
            Categorical(beta_list, name="beta"),
            Categorical(gamma_list, name="gamma"),
            Categorical(epochs_list, name="epochs"),
            Categorical(batch_size_list, name="batch_size"),
            Categorical(optimizer_list, name="optimizer"),
            Categorical(early_stopping_list, name="early_stopping"),
            Categorical(sample_size_list, name="sample_size"),
            Categorical(iterations_list, name="iterations"),
        ]

        res = None
        checkpoint_path = self.base_dir + "/bayes_opt_checkpoint.pkl"
        if os.path.exists(checkpoint_path):
            print("Resuming from checkpoint...")
            res = load(checkpoint_path)
            n_calls_remaining = self.max_call_count - len(res.x_iters)
            checkpoint_saver = CheckpointSaver(checkpoint_path, compress=9)
            res = gp_minimize(
                func=self.objective_function,
                dimensions=search_space,
                n_calls=n_calls_remaining,
                x0=res.x_iters,
                y0=res.func_vals,
                callback=[checkpoint_saver]
            )
        else:
            print("Starting new optimization...")
            checkpoint_saver = CheckpointSaver(checkpoint_path, compress=9)
            res = gp_minimize(
                func=self.objective_function,
                dimensions=search_space,
                n_calls=self.max_call_count,
                callback=[checkpoint_saver]
            )
        print("Completed")
        print(f"Best score (objective): {res.fun}")
        for dim, val in zip(self._build_search_space(), res.x):
           print(f"  {dim.name}: {val}")


class HyperoptBayesianSearch(AbstractSearch):

    def __init__(self, training_parameter_list: List[ParameterSet], validation_parameter_list: List[ParameterSet], test_parameter_list: List[ParameterSet], base_dir: str, max_call_count: int,  env: Environment, sort: Callable[[Environment], List[int]]):
        super().__init__(training_parameter_list, validation_parameter_list, test_parameter_list, base_dir, env, sort)
        self.checkpoint_path = self.base_dir + "/bayes_opt_checkpoint.pkl"
        self.max_call_count = max_call_count
        self.trials: Optional[Trials] = None
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as f:
                self.trials = pickle.load(f)
                print("Recovering after " + str(len(self.trials.trials)) + " trial(s).")
        else:
            self.trials = Trials()

    def save_trials(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(self.trials, f)

    def objective_function(self, params):

        configuration = pickle.loads(base64.b64decode(params["configuration"].encode('utf-8')))

        alpha = params["alpha"]
        beta = params["beta"]
        gamma = params["gamma"]
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        optimizer = params["optimizer"]
        early_stopping = params["early_stopping"]
        sample_size = params["sample_size"]
        iterations = params["iterations"]
        configuration.loss_funct_list = list(map(self.decode, configuration.loss_funct_list))

        hyperparameter = Hyperparameters(alpha, beta, gamma, epochs, batch_size, optimizer, early_stopping, sample_size, iterations)

        score = self.evaluate((configuration, hyperparameter))
        self.save_trials()
        return {'loss': -score, 'status': STATUS_OK}

    def encode(self, ref):
        if ref == binary_representation_loss:
            return "binary"
        elif ref == categorical_cross_entropy_loss:
            return "categorical"

    def decode(self, ref):
        if ref == "binary":
            return binary_representation_loss
        elif ref == "categorical":
            return categorical_cross_entropy_loss

    def search(self):

        configuration_list = []
        alpha_list = []
        beta_list = []
        gamma_list = []
        epochs_list = []
        batch_size_list = []
        optimizer_list = []
        early_stopping_list = []
        sample_size_list = []
        iterations_list = []

        for conf, params in self._build_search_space():
            conf.loss_funct_list = map(self.encode, conf.loss_funct_list)
            configuration_list.append(base64.b64encode(pickle.dumps(conf)).decode('utf-8'))
            if params.alpha not in alpha_list:
                alpha_list.append(params.alpha)
            if params.beta not in beta_list:
                beta_list.append(params.beta)
            if params.gamma not in gamma_list:
                gamma_list.append(params.gamma)
            if params.epochs not in epochs_list:
                epochs_list.append(params.epochs)
            if params.batch_size not in batch_size_list:
                batch_size_list.append(params.batch_size)
            if params.optimizer not in optimizer_list:
                optimizer_list.append(params.optimizer)
            if params.early_stopping not in early_stopping_list:
                early_stopping_list.append(params.early_stopping)
            if params.sample_size not in sample_size_list:
                sample_size_list.append(params.sample_size)
            if params.iterations not in iterations_list:
                iterations_list.append(params.iterations)

        hyper_opt_search_space = {
            "configuration" : hp.choice('configuration', configuration_list),
            "alpha": hp.choice('alpha', alpha_list),
            "beta": hp.choice('beta', beta_list),
            "gamma": hp.choice('gamma', gamma_list),
            "epochs": hp.choice('epochs', epochs_list),
            "batch_size": hp.choice('batch_size', batch_size_list),
            "optimizer": hp.choice('optimizer', optimizer_list),
            "early_stopping": hp.choice('early_stopping', early_stopping_list),
            "sample_size": hp.choice('sample_size', sample_size_list),
            "iterations": hp.choice('iterations', iterations_list),
        }

        best = fmin(
            fn=self.objective_function,
            space=hyper_opt_search_space,
            algo=tpe.suggest,
            max_evals=(self.max_call_count - len(self.trials.trials)),
            trials=self.trials
        )

        print("Best parameters:", best)

