from typing import Callable, List, Tuple, Dict, Optional
from kneed import KneeLocator

import tensorflow as tf
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from sampling import ParameterSet

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment

from nn.multi_domain_nn_forwarding import Layer, Procedure, categorical_cross_entropy_loss, binary_representation_loss, \
    create_contrastive_loss, BinaryLayer, AbstractSearch, MLDGProcedure, BayesianSearch, Hyperparameters


class BinaryEmbeddingSearch(BayesianSearch):

    base_dir = "./test"

    training_parameter_list: List[ParameterSet] = [ParameterSet(10,3,3,3)]

    validation_parameter_list: List[ParameterSet] = [ParameterSet(10,3,3,3), ParameterSet(12,0,3,3)]

    test_parameter_list: List[ParameterSet] = [ParameterSet(10,3,3,3), ParameterSet(12,0,3,3)]

    def __init__(self):
        super().__init__(BinaryEmbeddingSearch.training_parameter_list, BinaryEmbeddingSearch.validation_parameter_list, BinaryEmbeddingSearch.test_parameter_list, BinaryEmbeddingSearch.base_dir)

    def _create_initial_procedure_builder(self, model: Layer, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]) -> Callable[[], Procedure]:

        def pretraining():
            return Procedure(model, training_data_dict, validation_data_dict, test_data_dict, epochs=hyperparameters.epochs, learning_rate=hyperparameters.alpha, batch_size=hyperparameters.batch_size, optimizer= hyperparameters.optimizer)

        return pretraining

    def _create_subsequent_procedure_builders(self, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor, int]]) -> List[Callable[[Procedure], Procedure]]:

        def fine_tuning(procedure: Procedure):
            first_layer: Layer = procedure.layer

            last_extractor_layer: Optional[Layer] = None

            cur_layer: Layer = first_layer
            while cur_layer is not None:
                if cur_layer.forward:
                    last_extractor_layer = cur_layer
                cur_layer = cur_layer.successor

            if last_extractor_layer is not None:
                predictor_first_layer: Layer = last_extractor_layer.successor
                last_extractor_layer.successor = None

                x_data, y_data = Procedure.gen_training_data(training_data_dict)

                predictions = first_layer.predict(x_data)

                inertia = []
                labels = []
                k_range = list(range(1, 10))

                optimal_k = None

                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(predictions)
                    inertia.append(kmeans.inertia_)
                    labels.append(kmeans.labels_)

                kneedle = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
                optimal_k = kneedle.elbow

                contrastive_loss_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = create_contrastive_loss(labels[optimal_k - 1], 1.0)

                last_extractor_layer.loss_function = contrastive_loss_function
                last_extractor_layer.successor = predictor_first_layer

                return Procedure(first_layer, training_data_dict, validation_data_dict, test_data_dict, epochs=hyperparameters.epochs, learning_rate=hyperparameters.alpha, batch_size=hyperparameters.batch_size, optimizer=hyperparameters.optimizer)

        return [fine_tuning]


BinaryEmbeddingSearch().search()