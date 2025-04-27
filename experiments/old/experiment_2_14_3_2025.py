from typing import Dict, Callable, Optional, List, Tuple

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans


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

    def __init__(self, name: str,  input_dim: int, output_dim: int, loss_function: Optional[Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]], loss_weight: float = 1.0, activation_function_name: str = "relu"):

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
        self.activation_function: Callable[[tf.Tensor], tf.Tensor] = Layer.activation_function_dict[activation_function_name]
        self.activation_function_name: str = activation_function_name

        # Loss
        self.loss_function: Optional[Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = loss_function
        self.loss_weight = tf.constant(loss_weight)
        self.cur_loss = None

        # Weight Initialization
        self.w1: tf.Variable = tf.Variable(tf.random.normal([input_dim, output_dim]))
        self.b1: tf.Variable = tf.Variable(tf.random.normal([1, output_dim]))
        print(self.w1)


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
        linear = in_val @ self.w1 + self.b1
       # print("linear " + str(linear))
        activated = self.activation_function(linear)
       # print("activated " + str(activated))
        return activated

    def calc_loss(self, out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor):
        loss: tf.Tensor = tf.constant(0.0)
        if self.loss_function is not None:
            loss = self.loss_function(out_val, x, x_index, y)
            if self.loss_weight is not None:
                loss = tf.constant(self.loss_weight) * loss
        return loss


# Losses

def categorical_cross_entropy_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(out_val, y)


def binary_representation_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    binary_loss = tf.math.log((tf.reduce_sum(tf.square(tf.subtract(out_val, 1)) * tf.square(out_val))) + tf.constant(1.0))
    return binary_loss


def create_contrastive_loss(labels2, margin: float = 1.0) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:

    def contrastive_loss(out_val: tf.Tensor, x: tf.Tensor, x_index: tf.Tensor, y: tf.Tensor) -> tf.Tensor:

        batch_size = tf.shape(out_val)[0]
        indices = tf.where(tf.range(batch_size)[:, None] < tf.range(batch_size))

        out_val_1 = tf.gather(out_val, indices[:, 0])  # First element of each pair
        out_val_2 = tf.gather(out_val, indices[:, 1])

        distances = tf.sqrt(tf.reduce_sum((out_val_1 - out_val_2) ** 2, axis=1) + tf.constant(10**-2))

        def get_label(index_pair):
            i_0 = x_index[index_pair[0]][0]
            i_1 = x_index[index_pair[1]][0]
            return int(labels2[i_0] == labels2[i_1])

        with ThreadPoolExecutor() as executor:
           labels = list(executor.map(lambda index_pair: get_label(index_pair), indices))

        c_margin = tf.constant(margin, dtype=tf.float32)

        loss_similar = tf.cast(labels, tf.float32) * (distances ** 2)
        loss_dissimilar = (1 - tf.cast(labels, tf.float32)) * tf.square(tf.maximum(0.0, c_margin - distances))

        loss = tf.reduce_mean(loss_similar + loss_dissimilar)
        return loss

    return contrastive_loss


# Model Creation and Training

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


def create_batch(x_data: tf.Tensor, y_data: tf.Tensor, batch_size: Optional[int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if batch_size is None or batch_size > x_data.shape[0]:
        return x_data, y_data, tf.transpose(tf.constant([list(range(0, x_data.shape[0]))]))
    index_list = np.random.randint(0, x_data.shape[0], size=128)
    return tf.stack([x_data[index] for index in index_list]), tf.stack([y_data[index] for index in index_list]), tf.transpose(tf.constant([index_list]))


def training_step(first_layer: Layer, x_data: tf.Tensor, y_data: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer, batch_size: int = 5) -> Tuple[Layer, tf.Tensor]:
    loss = None
    x_sample, y_sample, x_index = create_batch(x_data, y_data, batch_size)
    with tf.GradientTape(persistent=False) as tape:
        layer_params = []
        current_layer = first_layer
        while current_layer is not None:
            tape.watch(current_layer.w1)
            tape.watch(current_layer.b1)
            layer_params.append((current_layer.w1, current_layer.b1))
            current_layer = current_layer.successor
        loss, prediction = first_layer.run(x_sample, x_sample, x_index, y_sample)
    dw_list = [param[0] for param in layer_params]
    db_list = [param[1] for param in layer_params]
    gradients = tape.gradient(loss, dw_list + db_list)
    optimizer.apply_gradients(zip(gradients, dw_list + db_list))
    return first_layer, loss


def create_vis_context(first_layer: Layer) -> Tuple[Dict[str, List[tf.Tensor]], Figure, Axes, Dict[str, Line2D]]:
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Live Loss Curve')
    loss_dict: Dict[str, List[tf.Tensor]] = dict()
    loss_dict["total"] = []
    line_dict: Dict[str, Line2D] = dict()
    line_dict["total"] = ax.plot([], [], label='Total Loss', marker='o')[0]
    cur_layer = first_layer
    while cur_layer is not None:
        if cur_layer.loss_function is not None and cur_layer.successor is not None:
            loss_dict[cur_layer.layer_name] = []
            line_dict[cur_layer.layer_name] = ax.plot([], [], label='Loss @ ' + cur_layer.layer_name, marker='o')[0]
        cur_layer = cur_layer.successor
    ax.legend()
    ax.grid(True)
    plt.show()
    return loss_dict, fig, ax, line_dict


def train(first_layer: Layer, x_data: tf.Tensor, y_data: tf.Tensor, epochs: int = 100, learning_rate: float = 1.0, batch_size: int = 100) -> Layer:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_dict, fig, ax, line_dict = create_vis_context(first_layer)
    epoch_arr = []
    for epoch in range(epochs):
        layer, loss = training_step(first_layer, x_data, y_data, optimizer, batch_size)
        epoch_arr.append(epoch)
        loss_dict["total"].append(loss)
        line_dict["total"].set_xdata(epoch_arr)
        line_dict["total"].set_ydata(loss_dict["total"])
        cur_layer = layer
        while cur_layer is not None:
            if cur_layer.layer_name in loss_dict and cur_layer.successor is not None:
                loss_dict[cur_layer.layer_name].append(cur_layer.cur_loss)
                line_dict[cur_layer.layer_name].set_xdata(epoch_arr)
                line_dict[cur_layer.layer_name].set_ydata(loss_dict[cur_layer.layer_name])
            cur_layer = cur_layer.successor
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)
        y_pred_classes = np.argmax(layer.predict(x_data), axis=1)  # Klasse mit der höchsten Wahrscheinlichkeit auswählen
        y_true_classes = np.argmax(y_data, axis=1)

        accuracy = np.mean(y_pred_classes == y_true_classes)
        print(accuracy)
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}\n\n")
    plt.clf()
    return first_layer







#

# Model Creation

extractor: List[Layer] = [
    Layer(name="in", input_dim=2, output_dim=2, loss_function=categorical_cross_entropy_loss, loss_weight=1.0, activation_function_name="softmax"),
    #Layer(name="latent", input_dim=2, output_dim=2, loss_function=categorical_cross_entropy_loss, loss_weight=1.0, activation_function_name="softmax"),
]

predictor: List[Layer] = [
    #Layer(name="out", input_dim=10, output_dim=2, loss_function=categorical_cross_entropy_loss, loss_weight=1.0, activation_function_name="softmax")
]

model: List[Layer] = extractor + predictor

# Data Creation - Tmp

sample_size: int = 4
predictor_count: int = 2

x_data = tf.random.uniform(shape=(sample_size, predictor_count), minval=0, maxval=1)
y_data = tf.one_hot(tf.random.uniform(shape=(sample_size,), minval=0, maxval=2, dtype=tf.int32), depth=2)


# Pretraining

first_layer: Layer = train(compile_model(model), x_data, y_data, epochs=1000, learning_rate=1.0)

extractor[1].successor = None
#for row in extractor[0].predict(x_data):
 #   print(row)

# Clustering

extractor_model_first_layer: Layer = compile_model(extractor)

y_fine_pred = extractor_model_first_layer.predict(x_data)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(y_fine_pred)
labels = kmeans.labels_

# Fine-Tuning

contrastive_loss_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = create_contrastive_loss(labels, 2.0)

last_extractor_layer: Layer = extractor[len(extractor) - 1]
last_extractor_layer.loss_function = contrastive_loss_function
last_extractor_layer.loss_weight = 1.0
last_extractor_layer.successor = None


model = extractor + predictor

pretraining_model_first_layer: Layer = train(compile_model(model), x_data, y_data, epochs=200, learning_rate=0.1, batch_size=12)
















