from typing import List, Tuple, Dict, Callable

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from numpy.typing import NDArray
from sklearn.cluster import KMeans


class LossLayer(layers.Layer):

    def __init__(self, loss_function: Callable[[tf.Tensor], tf.Tensor], lambda_value: float):
        super().__init__(trainable=False, name="loss_layer")
        self.loss_function = loss_function
        self.lambda_value = lambda_value

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # In case self.loss_function returns non-scalar value, its reduced through reduce_sum.

        # The following line is solely there for testing purposes.
        # self.add_loss(self.loss_function(inputs))

        self.add_loss(self.lambda_value * tf.reduce_sum(self.loss_function(inputs)))
        return inputs


class BinaryExtractorPredictorModelBuilder:

    @staticmethod
    def assure_is_defined(value):
        if value is None:
            raise ValueError("The passed value is None, even though a defined value is expected")
        return value

    def __init__(self):
        self.one_d_input_size: int = -1
        self.representation_loss_function: Callable[[tf.Tensor], tf.Tensor]
        self.layer_arr: List[layers.Layer] = []
        self.extractor_layer_arr: List[layers.Layer] = []
        self.predictor_layer_arr: List[layers.Layer] = []
        self.extractor_initialized = False

    def add_one_d_input_size(self, one_d_input_size: int) -> "BinaryExtractorPredictorModelBuilder":
        self.one_d_input_size = BinaryExtractorPredictorModelBuilder.assure_is_defined(one_d_input_size)
        return self

    def add_representation_loss_function(self, representation_loss_function: Callable[[tf.Tensor], tf.Tensor]) -> "BinaryExtractorPredictorModelBuilder":
        self.representation_loss_function = BinaryExtractorPredictorModelBuilder.assure_is_defined(representation_loss_function)
        return self

    def add_dense_extractor_layer(self, neuron_count: int, activation_function: str) -> "BinaryExtractorPredictorModelBuilder":
        if self.extractor_initialized:
            raise RuntimeError("Cannot add another extractor layer once a predictor has been added.")
        layer: keras.Layer
        if len(self.layer_arr) == 0:
            shape: Tuple[int] = (BinaryExtractorPredictorModelBuilder.assure_is_defined(self.one_d_input_size),)
            layer = layers.Dense(BinaryExtractorPredictorModelBuilder.assure_is_defined(neuron_count), activation=BinaryExtractorPredictorModelBuilder.assure_is_defined(activation_function), input_shape=shape)
        else:
            layer = layers.Dense(BinaryExtractorPredictorModelBuilder.assure_is_defined(neuron_count), activation=BinaryExtractorPredictorModelBuilder.assure_is_defined(activation_function))
        self.layer_arr.append(layer)
        self.extractor_layer_arr.append(layer)
        return self

    def add_dense_predictor_layer(self, neuron_count: int, activation_function: str) -> "BinaryExtractorPredictorModelBuilder":
        if len(self.layer_arr) == 0:
            raise RuntimeError("Cannot add predictor layer without preceding extraction layer.")
        elif not self.extractor_initialized:
            # TODO Parameterize lambda
            self.layer_arr.append(LossLayer(self.representation_loss_function, 1))
        self.extractor_initialized = True
        layer = layers.Dense(BinaryExtractorPredictorModelBuilder.assure_is_defined(neuron_count), activation=BinaryExtractorPredictorModelBuilder.assure_is_defined(activation_function))
        self.layer_arr.append(layer)
        self.predictor_layer_arr.append(layer)
        return self

    def build(self) -> keras.Model:
        return keras.models.Sequential(self.layer_arr)

    def build_tunable_model(self, model: keras.Model, contrastive_loss: Callable[[tf.Tensor], tf.Tensor]) -> keras.Model:
        ext_layers = [layer for layer in model.layers if layer in self.extractor_layer_arr]
        pre_layers = [layer for layer in model.layers if layer in self.predictor_layer_arr]
        contrastive_loss_layer: LossLayer = LossLayer(contrastive_loss, 1)
        return keras.Sequential(ext_layers + [contrastive_loss_layer] + pre_layers)

    def get_extractor_model(self, model: keras.Model) -> keras.Model:
        if not self.extractor_layer_arr:
            raise ValueError("No extractor layers found in the model.")
        return keras.models.Sequential([layer for layer in model.layers if layer in self.extractor_layer_arr])

    def get_predictor_model(self, model: keras.Model) -> keras.Model:
        if not self.predictor_layer_arr:
            raise ValueError("No predictor layers found in the model.")
        return keras.models.Sequential([layer for layer in model.layers if layer in self.predictor_layer_arr])



class TrainingProcedure:

    def __init__(self,  optimizer: str, epochs: int, class_loss: str, metrics: List[str], rep_loss_function: Callable[[tf.Tensor], tf.Tensor]):
        # Hyperparameter
        self.optimizer: str = optimizer
        self.epochs: int = epochs
        self.class_loss: str = class_loss
        self.metrics: List[str] = metrics
        self.rep_loss_function: Callable[[tf.Tensor], tf.Tensor] = rep_loss_function
        # Training and Test Data
        self.x_pre: NDArray = np.empty()
        self.y_pre: NDArray = np.empty()
        self.x_fine: NDArray = np.empty()
        self.y_fine: NDArray  = np.empty()
        self.x_fine_label_dict: Dict[Tuple[float],  int] = dict()
        # Data Structures
        self.model: keras.Model = keras.Sequential([])
        self.builder: BinaryExtractorPredictorModelBuilder = BinaryExtractorPredictorModelBuilder()
        #

    def _sample(self):
        #TODO
        pass

    def _standardize(self):
        #TODO
        pass

    def _create_architecture(self):
        self.builder.add_one_d_input_size(self.x_pre.shape[1])\
            .add_representation_loss_function(self.rep_loss_function)\
            .add_dense_extractor_layer(64, 'relu')\
            .add_dense_extractor_layer(128, 'relu')\
            .add_dense_predictor_layer(64, 'relu') \
            .add_dense_predictor_layer(10, 'softmax')
        self.model = self.builder\
            .build()
        self.model.compile(optimizer=self.optimizer, loss=self.class_loss, metrics=self.metrics)

    def _run_pre_protocol(self):
        self._sample()
        self._standardize()
        # Make sure _create_architecture is called after sampling so that the implementation can align input layer dims.
        self._create_architecture()

    def _pretraining(self):
        self.model.fit(self.x_pre, self.y_pre)

    def _clustering(self):
        extractor: keras.Model = self.builder.get_extractor_model(self.model)
        y_fine_pred = extractor.predict(self.x_fine)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(y_fine_pred)
        labels = kmeans.labels_
        for index in range(0, len(labels)):
            x_value_tpl = tuple(self.x_fine[index])
            self.x_fine_label_dict[x_value_tpl] = labels[index]

    def _label_x_fine_values(self, x_val_0: NDArray, x_val_1: NDArray) -> int:
        return int(self.x_fine_label_dict[tuple(x_val_0)] == self.x_fine_label_dict[tuple(x_val_1)])

    def _fine_tuning(self):

        pass

    def run(self) -> keras.Model:
        self._run_pre_protocol()
        self._pretraining()
        self._clustering()
        self._fine_tuning()
        return self.model



experiment: TrainingProcedure = TrainingProcedure('adam', 150, 'categorical_crossentropy', ['accuracy'], lambda x: x)


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
