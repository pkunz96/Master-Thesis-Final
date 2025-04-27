import os
from typing import Dict, Callable, Optional, List, Tuple

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sklearn.preprocessing import MinMaxScaler

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment
from experiments.old.sampling_old import ParameterSet


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

# Losses


def categorical_cross_entropy_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(out_val, y)


def binary_representation_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    binary_loss = tf.math.log((tf.reduce_sum(tf.square(tf.subtract(out_val, 1)) * tf.square(out_val))) + tf.constant(1.0))
    return binary_loss


def create_contrastive_loss(cluster_labels: List[int], margin: float = 1.0) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:

    def contrastive_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

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

    return contrastive_loss


class Procedure:

    optimizer_dict: Dict[str, tf.keras.optimizers.Optimizer] = {
        "adam": tf.keras.optimizers.Adam,
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
    def create_batch(x_data: tf.Tensor, y_data: tf.Tensor, batch_size: Optional[int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        data_size = tf.shape(x_data)[0]
        if batch_size is None or batch_size > data_size:
            indices = tf.range(data_size)
        else:
            indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=data_size, dtype=tf.int32)
        x_batch = tf.gather(x_data, indices)
        y_batch = tf.gather(y_data, indices)
        return x_batch, y_batch, tf.expand_dims(indices, axis=-1)

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], epochs: int = 100, learning_rate: float = 1.0, batch_size: int = 128, optimizer: str = "adam"):

        # Hyperparameters
        self.layer: Layer = layer
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.optimizer: tf.keras.optimizers.Optimizer = Procedure.optimizer_dict[optimizer](learning_rate=learning_rate)

        # Training Data

        self.training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]] = training_data_dict
        self.validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]] = validation_data_dict
        self.test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]] = test_data_dict

        # Metrics
        self.metrics = Procedure.metrics_list

        # Graphics
        self.accuracy_fig: Optional[Figure] = None
        self.accuracy_ax: Optional[Axes] = None
        self.training_accuracy_line: Optional[Line2D]
        self.validation_accuracy_line: Optional[Line2D]
        self.loss_fig: Optional[Figure] = None
        self.loss_ax: Optional[Axes] = None
        self.epoch_arr: List[int] = []
        self.training_accuracy_list: List[tf.Tensor] = []
        self.validation_accuracy_list: List[tf.Tensor] = []

        self.layer_name_training_loss_dict: Dict[str, List[tf.Tensor]] = dict()
        self.layer_name_validation_loss_dict: Dict[str, List[tf.Tensor]] = dict()

        self.layer_name_training_loss_line_dict: Dict[str, Line2D] = dict()
        self.layer_name_validation_loss_line_dict: Dict[str, Line2D] = dict()

        self.data_name_validation_loss_dict: dict[str, List[tf.Tensor]] = dict()

        self.data_name_validation_loss_line_dict: dict[str, Line2D] = dict()

        self.data_name_accuracy_dict: dict[str, List[tf.Tensor]] = dict()

        self.data_name_accuracy_line_dict: dict[str, Line2D] = dict()

        self.test_data_name_accuracy: Dict[str, tf.Tensor] = dict()
        self.average_test_accuracy: tf.Tensor

    def _calc_accuracy(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        y_pred_classes = tf.argmax(self.layer.predict(x), axis=1)
        y_true_classes = tf.argmax(y, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_classes, y_true_classes), dtype=tf.float32))
        return tf.constant(accuracy)

    def _execute_training_step(self) -> Tuple[tf.Tensor, tf.Tensor]:
        loss = None
        x_data, y_data = self.gen_training_data(self.training_data_dict)
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
        self._save_total_training_loss(loss)
        return x_data, y_data

    def _save_total_training_loss(self, loss: tf.Tensor):
        if self.layer_name_training_loss_dict is not None and "total" in self.layer_name_training_loss_dict:
            self.layer_name_training_loss_dict["total"].append(loss)

    def _init_visualization_context(self):
        plt.ion()
        if "training_accuracy" in self.metrics or "validation_accuracy" in self.metrics:
            self.accuracy_fig, self.accuracy_ax = plt.subplots()
            self.accuracy_ax.set_xlabel('Epochs')
            self.accuracy_ax.set_ylabel('Accuracy')
            self.accuracy_ax.set_title('Live Accuracy Curve')
            self.accuracy_ax.set_ylim(bottom=0)
            if "training_accuracy" in self.metrics:
                self.training_accuracy_line: Optional[Line2D] = self.accuracy_ax.plot([], [], label='Training Accuracy', marker='o')[0]
            if "validation_accuracy" in self.metrics:
                self.validation_accuracy_line: Optional[Line2D] = self.accuracy_ax.plot([], [], label='Validation Accuracy', marker='o')[0]
                for data_name in self.validation_data_dict:
                    if data_name == "total":
                        continue
                    self.data_name_accuracy_dict[data_name] = []
                    self.data_name_accuracy_line_dict[data_name] = self.accuracy_ax.plot([], [], label='Validation Accuracy for ' + data_name, marker='o')[0]
            self.accuracy_ax.legend()
            self.accuracy_ax.grid(True)
        if "training_loss" in self.metrics or "validation_loss" in self.metrics:
            self.loss_fig, self.loss_ax = plt.subplots()
            self.loss_ax.set_xlabel('Epochs')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.set_title('Live Loss Curve')
            if "training_loss" in self.metrics:
                self.layer_name_training_loss_dict["total"] = []
                self.layer_name_training_loss_line_dict["total"] = self.loss_ax.plot([], [], label='Training Loss', marker='o')[0]
                cur_layer = self.layer
                while cur_layer is not None:
                    if cur_layer.loss_function is not None and cur_layer.successor is not None:
                        self.layer_name_training_loss_dict[cur_layer.layer_name] = []
                        self.layer_name_training_loss_line_dict[cur_layer.layer_name] = self.loss_ax.plot([], [], label='Training Loss @ ' + cur_layer.layer_name, marker='o')[0]
                    cur_layer = cur_layer.successor
            if "validation_loss" in self.metrics:
                self.layer_name_validation_loss_dict["total"] = []
                self.layer_name_validation_loss_line_dict["total"] = self.loss_ax.plot([], [], label='Validation Loss', marker='o')[0]
                cur_layer = self.layer
                while cur_layer is not None:
                    if cur_layer.loss_function is not None and cur_layer.successor is not None:
                        self.layer_name_validation_loss_dict[cur_layer.layer_name] = []
                        self.layer_name_validation_loss_line_dict[cur_layer.layer_name] = self.loss_ax.plot([], [], label='Validation Loss @ ' + cur_layer.layer_name, marker='o')[0]
                    cur_layer = cur_layer.successor
                for data_name in self.validation_data_dict:
                    if data_name == "total":
                        continue
                    self.data_name_validation_loss_dict[data_name] = []
                    self.data_name_validation_loss_line_dict[data_name] = self.loss_ax.plot([], [], label='Validation Loss for ' + data_name, marker='o')[0]
            self.loss_ax.legend()
            self.loss_ax.grid(True)
        plt.show()

    @staticmethod
    def gen_training_data(training_data_dict) -> Tuple[tf.Tensor, tf.Tensor]:
        training_data_list = []
        for name in training_data_dict:
            training_data_list.append(training_data_dict[name])
        x_data, y_data = AbstractSearch.stack_training_data(training_data_list)
        return x_data, y_data

    def _evaluate_model(self, epoch: int, x_training_data: tf.Tensor, y_training_data: tf.Tensor) -> "Procedure":
        self.epoch_arr.append(epoch)
        if "training_loss" in self.metrics:
            self.layer_name_training_loss_line_dict["total"].set_xdata(self.epoch_arr)
            self.layer_name_training_loss_line_dict["total"].set_ydata(self.layer_name_training_loss_dict["total"])
            cur_layer = self.layer
            while cur_layer is not None:
                if cur_layer.layer_name in self.layer_name_training_loss_dict and cur_layer.successor is not None:
                    self.layer_name_training_loss_dict[cur_layer.layer_name].append(cur_layer.cur_loss)
                    self.layer_name_training_loss_line_dict[cur_layer.layer_name].set_xdata(self.epoch_arr)
                    self.layer_name_training_loss_line_dict[cur_layer.layer_name].set_ydata(self.layer_name_training_loss_dict[cur_layer.layer_name])
                cur_layer = cur_layer.successor
        if "validation_loss" in self.metrics:
            x_val, y_val, _ = self.validation_data_dict["total"]
            x_val_sample, y_val_sample, indices = Procedure.create_batch(x_val, y_val, 64)
            loss, prediction = self.layer.run(x_val_sample, x_val_sample, indices, y_val_sample)
            self.layer_name_validation_loss_dict["total"].append(loss)
            self.layer_name_validation_loss_line_dict["total"].set_xdata(self.epoch_arr)
            self.layer_name_validation_loss_line_dict["total"].set_ydata(self.layer_name_validation_loss_dict["total"])
            cur_layer = self.layer
            while cur_layer is not None:
                if cur_layer.layer_name in self.layer_name_validation_loss_dict and cur_layer.successor is not None:
                    self.layer_name_validation_loss_dict[cur_layer.layer_name].append(cur_layer.cur_loss)
                    self.layer_name_validation_loss_line_dict[cur_layer.layer_name].set_xdata(self.epoch_arr)
                    self.layer_name_validation_loss_line_dict[cur_layer.layer_name].set_ydata(self.layer_name_validation_loss_dict[cur_layer.layer_name])
                cur_layer = cur_layer.successor
            for data_name in self.validation_data_dict:
                if data_name == "total":
                    continue
                d_x_data, d_y_data, _ = self.validation_data_dict[data_name]
                d_x_val_sample, d_y_val_sample, indices = Procedure.create_batch(d_x_data, d_y_data, 64)
                loss, prediction = self.layer.run(d_x_val_sample, d_x_val_sample, indices, d_y_val_sample)
                self.data_name_validation_loss_dict[data_name].append(loss)
                self.data_name_validation_loss_line_dict[data_name].set_xdata(self.epoch_arr)
                self.data_name_validation_loss_line_dict[data_name].set_ydata(self.data_name_validation_loss_dict[data_name])
        if "training_accuracy" in self.metrics:
            training_accuracy = self._calc_accuracy(x_training_data, y_training_data)
            self.training_accuracy_list.append(training_accuracy)
            self.training_accuracy_line.set_xdata(self.epoch_arr)
            self.training_accuracy_line.set_ydata(self.training_accuracy_list)
        if "validation_accuracy" in self.metrics:
            x_val, y_val, _ = self.validation_data_dict["total"]
            validation_accuracy = self._calc_accuracy(x_val, y_val)
            self.validation_accuracy_list.append(validation_accuracy)
            self.validation_accuracy_line.set_xdata(self.epoch_arr)
            self.validation_accuracy_line.set_ydata(self.validation_accuracy_list)
            for data_name in self.validation_data_dict:
                if data_name == "total":
                    continue
                d_x_data, d_y_data, _ = self.validation_data_dict[data_name]
                data_name_accuracy = self._calc_accuracy(d_x_data, d_y_data)
                self.data_name_accuracy_dict[data_name].append(data_name_accuracy)
                self.data_name_accuracy_line_dict[data_name].set_xdata(self.epoch_arr)
                self.data_name_accuracy_line_dict[data_name].set_ydata(self.data_name_accuracy_dict[data_name])
        if self.loss_ax is not None:
            self.loss_ax.relim()
            self.loss_ax.autoscale_view()
        if self.accuracy_ax is not None:
            self.accuracy_ax.relim()
            self.accuracy_ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)
        return self

    def _test_model(self) -> None:
        test_data_list: List[Tuple[tf.Tensor, tf.Tensor]] = []
        for data_name in self.test_data_dict:
            x_data, y_data, _= self.test_data_dict[data_name]
            test_data_list.append((x_data, y_data))
            self.test_data_name_accuracy[data_name] = self._calc_accuracy(x_data, y_data)
        x_data, y_data = AbstractSearch.stack_training_data(test_data_list)
        self.average_test_accuracy = self._calc_accuracy(x_data, y_data)

    def train(self) -> Layer:
        self.clear()
        for epoch in range(self.epochs):
            x_data, y_data = self._execute_training_step()
            self._evaluate_model(epoch, x_data, y_data)
        self._test_model()
        plt.ioff()
        return self.layer

    def disable_metric(self, metric: str) -> "Procedure":
        new_metrics = []
        for cur_metric in self.metrics:
            if cur_metric != metric:
                new_metrics.append(cur_metric)
        self.metrics = new_metrics

    def clear(self):
        self.accuracy_fig: Optional[Figure] = None
        self.accuracy_ax: Optional[Axes] = None
        self.training_accuracy_line: Optional[Line2D]
        self.validation_accuracy_line: Optional[Line2D]
        self.loss_fig: Optional[Figure] = None
        self.loss_ax: Optional[Axes] = None
        self.epoch_arr: List[int] = []
        self.training_accuracy_list: List[tf.Tensor] = []
        self.validation_accuracy_list: List[tf.Tensor] = []
        self.layer_name_training_loss_dict: Dict[str, List[tf.Tensor]] = dict()
        self.layer_name_validation_loss_dict: Dict[str, List[tf.Tensor]] = dict()
        self.layer_name_training_loss_line_dict: Dict[str, Line2D] = dict()
        self.layer_name_validation_loss_line_dict: Dict[str, Line2D] = dict()
        self.data_name_validation_loss_dict: dict[str, List[tf.Tensor]] = dict()
        self.data_name_validation_loss_line_dict: dict[str, Line2D] = dict()
        self.data_name_accuracy_dict: dict[str, List[tf.Tensor]] = dict()
        self.data_name_accuracy_line_dict: dict[str, Line2D] = dict()
        self._init_visualization_context()


class MLDGProcedure(Procedure):

    def __init__(self, layer: Layer, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], epochs: int = 100, alpha: float = 1.0, batch_size: int = 128, optimizer: str = "adam", gamma=.2, beta=.2):
        super().__init__(layer, training_data_dict, validation_data_dict, test_data_dict, epochs, alpha, batch_size, optimizer)
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

        self._save_total_training_loss(loss)

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
            training_data_list = []
            meta_training_data_list = []
            for name in self.training_data_dict:
                rand = int(np.random.randint(0, 2, size=1))
                if rand == 0:
                    training_data_list.append(self.training_data_dict[name])
                else:
                    meta_training_data_list.append(self.training_data_dict[name])
        x_training_data, y_training_data = AbstractSearch.stack_training_data(training_data_list)
        x_meta_training_data, y_meta_training_data = AbstractSearch.stack_training_data(meta_training_data_list)
        return x_training_data, y_training_data, x_meta_training_data, y_meta_training_data


class Pipeline:

    @staticmethod
    def _average_tensors(tensors: List[List[tf.Tensor]]):
        result: List[tf.Tensor] = []
        length: int = len(tensors[0])

        for length_index in range(0, length):
            sum_tensor: tf.Tensor = tf.constant(0.0)
            for sample_index in range(len(tensors)):
                sum_tensor = sum_tensor + tensors[sample_index][length_index]
            sum_tensor = sum_tensor / len(tensors)
            result.append(sum_tensor)
        return result

    @staticmethod
    def _average_stage_iteration_dict(stage_iteration_dict: Dict[int, Dict[int, List[tf.Tensor]]]) -> Dict[int, List[tf.Tensor]]:
        stage_dict: Dict[int, List[tf.Tensor]] = dict()
        for stage in stage_iteration_dict:
            value_list = []
            for iteration in stage_iteration_dict[stage]:
                value_list.append(stage_iteration_dict[stage][iteration])
            stage_dict[stage] = Pipeline._average_tensors(value_list)
        return stage_dict

    @staticmethod
    def _average_stage_iteration_name_dict(stage_iteration_name_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]]) -> Dict[int, Dict[str, List[tf.Tensor]]]:
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

        self.stage_iteration_epoch_dict: Dict[int, Dict[int, int]] = dict()
        self.stage_iteration_training_accuracy_dict: Dict[int, Dict[int, List[tf.Tensor]]] = dict()
        self.stage_iteration_validation_accuracy_dict: Dict[int, Dict[int, List[tf.Tensor]]] = dict()

        self.stage_iteration_layer_name_training_loss_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]] = dict()
        self.stage_iteration_layer_name_validation_loss_dict: Dict[int, Dict[int, Dict[str, List[tf.Tensor]]]] = dict()

        self.stage_iteration_data_name_validation_loss_dict: Dict[int, Dict[int, Dict[str, Dict[str, List[tf.Tensor]]]]] = dict()
        self.stage_iteration_data_name_accuracy_dict: Dict[int, Dict[int, Dict[str, Dict[str, List[tf.Tensor]]]]] = dict()

        self.stage_iteration_loss_fig_dict: Dict[int, Dict[int, Figure]] = dict()
        self.stage_iteration_accuracy_fig_dict: Dict[int, Dict[int, Figure]] = dict()

        self.stage_iteration_training_data_dict: Dict[int, Dict[int, Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]]] = dict()
        self.stage_iteration_validation_data_dict: Dict[int, Dict[int, Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]]] = dict()

        self.stage_test_data_name_accuracy_dict: Dict[int, Dict[str, tf.Tensor]] = dict()
        self.stage_avg_test_accuracy_dict: Dict[int,  tf.Tensor] = dict()

        for cur_stage in range(len(subsequent_procedure_builder_list) + 1):
            self.stage_iteration_epoch_dict[cur_stage] = dict()
            self.stage_iteration_training_accuracy_dict[cur_stage] = dict()
            self.stage_iteration_validation_accuracy_dict[cur_stage] = dict()
            self.stage_iteration_layer_name_training_loss_dict[cur_stage] = dict()
            self.stage_iteration_layer_name_validation_loss_dict[cur_stage] = dict()
            self.stage_iteration_data_name_validation_loss_dict[cur_stage] = dict()
            self.stage_iteration_data_name_accuracy_dict[cur_stage] = dict()
            self.stage_iteration_loss_fig_dict[cur_stage] = dict()
            self.stage_iteration_accuracy_fig_dict[cur_stage] = dict()

            self.stage_iteration_training_data_dict[cur_stage] = dict()
            self.stage_iteration_validation_data_dict[cur_stage] = dict()

            for cur_iteration in range(iterations):
                self.stage_iteration_epoch_dict[cur_stage][cur_iteration] = 0
                self.stage_iteration_training_accuracy_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_validation_accuracy_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_layer_name_training_loss_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_layer_name_validation_loss_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_data_name_validation_loss_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_data_name_accuracy_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_loss_fig_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_accuracy_fig_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_training_data_dict[cur_stage][cur_iteration] = dict()
                self.stage_iteration_validation_data_dict[cur_stage][cur_iteration] = dict()

        self.stage_epoch_dict = None
        self.stage_training_accuracy_dict = None
        self.stage_validation_accuracy_dict = None
        self.stage_layer_name_training_loss_dict = None
        self.stage_layer_name_validation_loss_dict = None
        self.stage_data_name_validation_loss_dict = None
        self.stage_data_name_accuracy_dict = None

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
        self.stage_iteration_training_accuracy_dict[stage][iteration] = procedure.training_accuracy_list
        self.stage_iteration_validation_accuracy_dict[stage][iteration] = procedure.validation_accuracy_list
        self.stage_iteration_layer_name_training_loss_dict[stage][iteration] = procedure.layer_name_training_loss_dict
        self.stage_iteration_layer_name_validation_loss_dict[stage][iteration] = procedure.layer_name_validation_loss_dict
        self.stage_iteration_data_name_validation_loss_dict[stage][iteration] = procedure.data_name_validation_loss_dict
        self.stage_iteration_data_name_accuracy_dict[stage][iteration] = procedure.data_name_accuracy_dict
        self.stage_iteration_loss_fig_dict[stage][iteration] = procedure.loss_fig
        self.stage_iteration_accuracy_fig_dict[stage][iteration] = procedure.accuracy_fig
        self.stage_iteration_training_data_dict[stage][iteration] = procedure.training_data_dict
        self.stage_iteration_validation_data_dict[stage][iteration] = procedure.validation_data_dict
        self.stage_test_data_name_accuracy_dict[stage] = procedure.test_data_name_accuracy
        self.stage_avg_test_accuracy_dict[stage] = procedure.average_test_accuracy

    def _average_stats(self):
        self.stage_epoch_dict = dict()
        for stage in self.stage_iteration_epoch_dict:
            epoch_count: int = 0
            for iteration in self.stage_iteration_epoch_dict[stage]:
                cur_count = self.stage_iteration_epoch_dict[stage][iteration]
                epoch_count = epoch_count + cur_count
            self.stage_epoch_dict[stage] = epoch_count / len(self.stage_iteration_epoch_dict[stage])
        self.stage_training_accuracy_dict = dict()
        self.stage_training_accuracy_dict = Pipeline._average_stage_iteration_dict(self.stage_iteration_training_accuracy_dict)
        self.stage_validation_accuracy_dict = Pipeline._average_stage_iteration_dict(self.stage_iteration_validation_accuracy_dict)
        self.stage_layer_name_training_loss_dict = Pipeline._average_stage_iteration_name_dict(self.stage_iteration_layer_name_training_loss_dict)
        self.stage_layer_name_validation_loss_dict = Pipeline._average_stage_iteration_name_dict(self.stage_iteration_layer_name_validation_loss_dict)
        self.stage_data_name_validation_loss_dict = Pipeline._average_stage_iteration_name_dict(self.stage_iteration_data_name_validation_loss_dict)
        self.stage_data_name_accuracy_dict = Pipeline._average_stage_iteration_name_dict(self.stage_iteration_data_name_accuracy_dict)


class Configuration:

    def __init__(self, neuron_count_list: List[int], activation_func_list: List[str], loss_func_list: List[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]], forwarding_list: List[bool], binary_layer_list: List[bool]):
        self.neuron_count_list: List[int] = neuron_count_list
        self.activation_func_list: List[str] = activation_func_list
        self.loss_funct_list: List[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = loss_func_list
        self.forwarding_list: List[bool] = forwarding_list
        self.binary_layer_list: List[bool] = binary_layer_list

    def build(self) -> Layer:
        pass


class Hyperparameters:

    def __init__(self, alpha, beta, gamma, epochs, batch_size, optimizer, dropout, early_stopping, sample_size, iterations):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.dropout = dropout
        self.early_stopping = early_stopping
        self.sample_size = sample_size
        self.iterations = iterations


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
    def _build_model(class_count: int, configuration: Configuration) -> Layer:
        model = []
        index = 0

        # TODO

        #cur_in = class_count
        cur_in = 9
        for neuron_count, activation_func, loss_func, forwarding, is_binary in zip(configuration.neuron_count_list, configuration.activation_func_list, configuration.loss_funct_list, configuration.forwarding_list, configuration.binary_layer_list):
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
    def create_training_data(param_set: sampling.ParameterSet, sample_size: int) -> Tuple[tf.Tensor, tf.Tensor, int]:
        # Sampling
        x_np_data, y_np_data = sampling.MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment).sample(sample_size)
        # Scaling
        x_np_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_np_data)
        # Conversion to TensorFlow Tensors
        x_tf_data, y_tf_data = tf.cast(tf.convert_to_tensor(x_np_data), dtype=tf.float32), tf.cast(
            tf.convert_to_tensor(y_np_data[1:, :]), dtype=tf.float32)
        # Swapping first and last column so that the state is passed through the first layers.
        x_tf_data = tf.gather(x_tf_data, list(range(x_tf_data.shape[1] - 1, -1, -1)), axis=1)
        # One-Hot-Encoding
        y_tf_data_reshaped = tf.reshape(y_tf_data, shape=(y_tf_data.shape[0],))
        unique_values, unique_encoding = tf.unique(y_tf_data_reshaped)
        y_tf_one_hot_data = tf.one_hot(unique_encoding, depth=unique_values.shape[0])
        # Returns the training data - x_tf_data and y_tf_one_hot_data - and the class count.
        return x_tf_data, y_tf_one_hot_data, unique_values.shape[0]

    @staticmethod
    def generate_data_dict(hyperparameters: Hyperparameters, parameter_set_list: List[ParameterSet], validation_data: bool = False) -> Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]:
        data_dict = dict()
        has_total = False
        for param_set in parameter_set_list:
            param_set_str = "p_mu=" + str(param_set.problem_mu) + " p_sigma=" + str(
                param_set.problem_sigma) + " s_mu" + str(param_set.problem_size_mu) + " s_sigma=" + str(
                param_set.problem_size_sigma)
            if validation_data and not has_total:
                param_set_str = "total"
                has_total = True
            data_dict[param_set_str] = AbstractSearch.create_training_data(param_set, hyperparameters.sample_size)
        return data_dict

    @staticmethod
    def _build_search_space() -> List[Tuple[Configuration, Hyperparameters]]:
        return [(Configuration([2, 10, 2],  ["relu", "relu", "relu"], [None, binary_representation_loss, categorical_cross_entropy_loss], [True, False, False], [False, False, False]), Hyperparameters(0.3, 0.3, 0.3, 10, 64, "adam", 0.5, 100, sample_size=10*3, iterations=2))]

    def __init__(self, training_parameter_list: List[ParameterSet], validation_parameter_list: List[ParameterSet], test_parameter_list: List[ParameterSet],  base_dir: str):
        self.search_space: List[Tuple[Configuration, Hyperparameters]] = AbstractSearch._build_search_space()
        self.training_parameter_list = training_parameter_list
        self.validation_parameter_list = validation_parameter_list
        self.test_parameter_list = test_parameter_list
        self.base_dir = base_dir

    def search(self):
        configuration = self._next_configuration()
        count = 0
        while configuration is not None and count < 1:
            pipeline: Pipeline = self._build_pipeline(configuration)
            pipeline.run()
            self.save_result(configuration, pipeline)
            configuration = self._next_configuration()
            count = count + 1

    def _next_configuration(self) -> Optional[Tuple[Configuration, Hyperparameters]]:
        pass

    def _build_pipeline(self, configuration: Tuple[Configuration, Hyperparameters]) -> Pipeline:
        conf, hyperparameters = configuration
        training_data = AbstractSearch.generate_data_dict(hyperparameters, self.training_parameter_list, False)
        validation_data = AbstractSearch.generate_data_dict(hyperparameters, self.validation_parameter_list, True)
        test_data = AbstractSearch.generate_data_dict(hyperparameters, self.test_parameter_list, False)
        model: Layer = AbstractSearch._build_model(training_data[next(iter(training_data))][2], conf)
        initial_procedure_builder: Callable[[], Procedure] = self._create_initial_procedure_builder(model, hyperparameters, training_data, validation_data, test_data)
        subsequent_procedure_builders: List[Callable[[Procedure], Procedure]] = self._create_subsequent_procedure_builders(hyperparameters, training_data, validation_data, test_data)
        return Pipeline(initial_procedure_builder, subsequent_procedure_builders, iterations=hyperparameters.iterations)

    def _create_initial_procedure_builder(self, model: Layer, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]) -> Callable[[], Procedure]:
        pass

    def _create_subsequent_procedure_builders(self, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]) -> List[Callable[[Procedure], Procedure]]:
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

    def save_result(self, configuration: Tuple[Configuration, Hyperparameters], pipeline: Pipeline) -> None:
        cur_dir: str = AbstractSearch.create_dir_if_not_existent(self.base_dir + "./")
        with open(cur_dir + "hyperparameters.txt", "w") as f:
            f.write("iterations: " + str(configuration[1].iterations))
            f.write("sample_size: " + str(configuration[1].sample_size))
            f.write("alpha: " + str(configuration[1].alpha))
            f.write("beta: " + str(configuration[1].beta))
            f.write("gamma: " + str(configuration[1].gamma))
            f.write("epochs: " + str(configuration[1].epochs))
            f.write("batch_size: " + str(configuration[1].batch_size))
            f.write("optimizer: " + str(configuration[1].optimizer))
            f.write("dropout rate: " + str(configuration[1].dropout))
            f.write("early_stopping: " + str(configuration[1].early_stopping))
        with open(cur_dir + "configuration.txt", "w") as f:
            f.write("neuron_count_list: " + str(configuration[0].neuron_count_list))
            f.write("activation_func_list: " + str(configuration[0].activation_func_list))
            f.write("forwarding_list: " + str(configuration[0].forwarding_list))
            f.write("loss_funct_list: " + str(configuration[0].loss_funct_list))
            f.write("binary_layer_list: " + str(configuration[0].binary_layer_list))
        for stage in range(0, len(pipeline.subsequent_procedure_builder_list) + 1):
            stage_dir: str = AbstractSearch.create_dir_if_not_existent(cur_dir + "stage_" + str(stage) + "/")
            for iteration in range(0, pipeline.iterations):
                iteration_dir = stage_dir + "iteration_" + str(iteration) + "/"
                iteration_dir = AbstractSearch.create_dir_if_not_existent(iteration_dir)
                with open(iteration_dir + "training_accuracy.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list(pipeline.stage_iteration_training_accuracy_dict[stage][iteration])))
                with open(iteration_dir + "validation_accuracy.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list(pipeline.stage_iteration_validation_accuracy_dict[stage][iteration])))
                with open(iteration_dir + "layer_name_training_loss.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list_dict(pipeline.stage_iteration_layer_name_training_loss_dict[stage][iteration])))
                with open(iteration_dir + "layer_name_validation_loss.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list_dict(pipeline.stage_iteration_layer_name_validation_loss_dict[stage][iteration])))
                with open(iteration_dir + "data_name_validation_loss.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list_dict(pipeline.stage_iteration_data_name_validation_loss_dict[stage][iteration])))
                with open(iteration_dir + "data_name_accuracy.txt", "w") as f:
                    f.write(str(AbstractSearch.map_list_dict(pipeline.stage_iteration_data_name_accuracy_dict[stage][iteration])))
                pipeline.stage_iteration_loss_fig_dict[stage][iteration].savefig(iteration_dir + "loss.png")
                pipeline.stage_iteration_accuracy_fig_dict[stage][iteration].savefig(iteration_dir + "accuracy.png")
                for training_data_name in pipeline.stage_iteration_training_data_dict[stage][iteration]:
                    training_data_dir = iteration_dir + "train_" + training_data_name
                    predictors, responses, _ = pipeline.stage_iteration_training_data_dict[stage][iteration][training_data_name]
                    np.save(training_data_dir + "predictors.npy", predictors.numpy())
                    np.save(training_data_dir + "responses.npy", responses.numpy())
                for validation_data_name in pipeline.stage_iteration_validation_data_dict[stage][iteration]:
                    val_data_dir = iteration_dir + "val_" + validation_data_name
                    predictors, responses, _ = pipeline.stage_iteration_validation_data_dict[stage][iteration][validation_data_name]
                    np.save(val_data_dir + "predictors.npy", predictors.numpy())
                    np.save(val_data_dir + "responses.npy", responses.numpy())
            with open(stage_dir + "training_accuracy.txt", "w") as f:
                f.write(str(AbstractSearch.map_list(pipeline.stage_training_accuracy_dict[stage])))
                fig, ax = AbstractSearch.plot_list("Epochs", "Accuracy", "Training Accuracy", "", pipeline.stage_training_accuracy_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(stage_dir + "training_accuracy.jpg")
            with open(stage_dir + "validation_accuracy.txt", "w") as f:
                f.write(str(AbstractSearch.map_list(pipeline.stage_validation_accuracy_dict[stage])))
                fig, ax = AbstractSearch.plot_list("Epochs", "Accuracy", "Validation Accuracy", "", pipeline.stage_validation_accuracy_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(stage_dir + "validation_accuracy.jpg")
            with open(stage_dir + "validation_loss_at_layers.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_layer_name_training_loss_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Training Loss @ Layers", pipeline.stage_layer_name_training_loss_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(stage_dir + "validation_loss_at_layers.jpg")
            with open(stage_dir + "validation_loss_at_layers.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_layer_name_validation_loss_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Validation Loss @ Layers", pipeline.stage_layer_name_validation_loss_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(stage_dir + "validation_loss_at_layers.jpg")
            with open(stage_dir + "validation_loss_at_data_sets.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_data_name_validation_loss_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Validation Loss @ Data Sets", pipeline.stage_data_name_validation_loss_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(stage_dir + "validation_loss_at_data_sets.jpg")
            with open(stage_dir + "accuracy_at_data_sets.txt", "w") as f:
                f.write(str(AbstractSearch.map_list_dict(pipeline.stage_data_name_accuracy_dict[stage])))
                fig, ax = AbstractSearch.plot_dict_list("Epochs", "Loss", "Accuracy @ Data Sets", pipeline.stage_data_name_accuracy_dict[stage])
                ax.set_ylim(bottom=0)
                fig.savefig(stage_dir + "accuracy_at_data_sets.jpg")
            with open(stage_dir + "test_accuracy.txt", "w") as f:
                f.write(str(pipeline.stage_test_data_name_accuracy_dict[stage]))
            with open(stage_dir + "avg_test_accuracy.txt", "w") as f:
                f.write(str(pipeline.stage_avg_test_accuracy_dict[stage]))


class BayesianSearch(AbstractSearch):

    def __init__(self,training_parameter_list: List[ParameterSet], validation_parameter_list: List[ParameterSet], test_parameter_list: List[ParameterSet], base_dir: str):
        super().__init__(training_parameter_list, validation_parameter_list, test_parameter_list, base_dir)

    def _next_configuration(self) -> Optional[Tuple[Configuration, Hyperparameters]]:
        return self.search_space[0]


