from itertools import product
from math import ceil
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
from kneed import KneeLocator

import tensorflow as tf
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Categorical

from sampling import ParameterSet

from algorithms.top_down_merge_sort import top_down_merge_sort, gen_top_down_merge_sort_environment

from nn.multi_domain_nn_forwarding import Layer, Procedure, categorical_cross_entropy_loss, \
    binary_representation_loss, \
    create_contrastive_loss, BinaryLayer, AbstractSearch, BayesianSearch, Hyperparameters, Configuration, MLDGProcedure, \
    HyperoptBayesianSearch

from sklearn.metrics import silhouette_score

class DIPEInsertionSortSearch(HyperoptBayesianSearch):

    bayesian_sample_count = 200

    sample_size = 8096

    suffix = "top_down_merge_sort"

    base_dir = "./dipe_" + suffix + "_" + str(sample_size)

    sorting_algorithm = top_down_merge_sort

    gen_env = gen_top_down_merge_sort_environment

    def __init__(self):
        super().__init__(DIPEInsertionSortSearch.training_parameter_list, DIPEInsertionSortSearch.validation_parameter_list, DIPEInsertionSortSearch.test_parameter_list, DIPEInsertionSortSearch.base_dir, DIPEInsertionSortSearch.bayesian_sample_count, gen_insertion_sort_environment, straight_insertion_sort)

    def _build_search_space(self) -> List[Tuple[Configuration, Hyperparameters]]:
        search_space = []

        binary_embedding_dim: int = 3  # Two predicates plus the state.

        for first_extractor_neuron_count_exp in range(7, 11):
            for predictor_neuron_count_exp in range(7, 11):
                for hidden_extractor_layer_count in range(1, 3):
                    first_hidden_extractor_layer_size = 2**first_extractor_neuron_count_exp
                    extract_neuron_count_arr = [first_hidden_extractor_layer_size - x * ceil((first_hidden_extractor_layer_size - binary_embedding_dim) / hidden_extractor_layer_count) for x in range(0, hidden_extractor_layer_count + 1)]
                    extract_neuron_count_arr[len(extract_neuron_count_arr) - 1] = binary_embedding_dim
                    predictor_neuron_count_arr = [2**predictor_neuron_count_exp for x in range(2)]
                    neuron_count_arr = [1] + extract_neuron_count_arr + predictor_neuron_count_arr + [1]
                    loss_func_arr = []
                    forwarding_arr = []
                    binary_arr = []
                    for layer_index in range(len(neuron_count_arr)):
                        if layer_index == len(extract_neuron_count_arr):
                            loss_func_arr.append(binary_representation_loss)
                        elif layer_index == len(neuron_count_arr) - 1:
                            loss_func_arr.append(categorical_cross_entropy_loss)
                        else:
                            loss_func_arr.append(None)
                        if layer_index < len(extract_neuron_count_arr):
                            forwarding_arr.append(True)
                        else:
                            forwarding_arr.append(False)
                        binary_arr.append(False)
                    layer_count = len(neuron_count_arr) - 1
                    for activation_func_list in list(product(["relu", "sigmoid"], repeat=layer_count)):
                        activation_func_list = activation_func_list + ("softmax",)
                        for learning_rate in [0.01, 0.001]:
                            for beta in [0.01, 0.001]:
                                for gamma in [0.01, 0.001]:
                                    search_space.append((Configuration(neuron_count_arr, activation_func_list, loss_func_arr, forwarding_arr, binary_arr), Hyperparameters(learning_rate, beta, gamma, 10, 64, "adam", 10, sample_size=8096, iterations=5)))
        return search_space

    def _create_initial_procedure_builder(self, model: Layer, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> Callable[[], Procedure]:
        def pretraining():
            last_extractor_layer: Optional[Layer] = None
            cur_layer: Layer = model
            while cur_layer is not None:
                if cur_layer.loss_function is None:
                    cur_layer = cur_layer.successor
                else:
                    last_extractor_layer = cur_layer
                    break
            if cur_layer is not None:
                cur_layer.loss_function = binary_representation_loss
                cur_layer.loss_weight = tf.constant(1.0)
            return MLDGProcedure(model, training_data_dict, validation_data_dict, test_data_dict, epochs=hyperparameters.epochs, batch_size=hyperparameters.batch_size, optimizer=hyperparameters.optimizer, early_stopping=hyperparameters.early_stopping,  alpha = hyperparameters.alpha, beta = hyperparameters.beta, gamma = hyperparameters.gamma)
        return pretraining

    def _create_subsequent_procedure_builders(self, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> List[Callable[[Procedure], Procedure]]:

        def fine_tuning(procedure: Procedure):
            first_layer: Layer = procedure.layer

            last_extractor_layer: Optional[Layer] = None

            cur_layer: Layer = first_layer
            while cur_layer is not None:
                if cur_layer.loss_function is None:
                    cur_layer = cur_layer.successor
                else:
                    last_extractor_layer = cur_layer
                    break

            if last_extractor_layer is not None:
                predictor_first_layer: Layer = last_extractor_layer.successor
                last_extractor_layer.successor = None
                last_extractor_layer.loss_weight = tf.constant(1.0)

                x_data, y_data = Procedure.merge_training_data_dict(training_data_dict)

                predictions = first_layer.predict(x_data)

                inertia = []
                labels = []
                models = []
                k_range = list(range(2, 10))

                optimal_k = None
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(predictions)
                    inertia.append(kmeans.inertia_)
                    labels.append(kmeans.labels_)
                    models.append(kmeans)

                kneedle = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
                optimal_k = kneedle.elbow

                if optimal_k is None:
                    silhouette_scores = []
                    for model in models:
                        score = -1.0
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(predictions, model.labels_)
                        silhouette_scores.append(score)
                    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]

                contrastive_loss_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = create_contrastive_loss(labels[optimal_k - 1], 1.5)

                last_extractor_layer.loss_function = contrastive_loss_function
                last_extractor_layer.successor = predictor_first_layer

                return Procedure(first_layer, training_data_dict, validation_data_dict, test_data_dict, epochs=hyperparameters.epochs, learning_rate=hyperparameters.alpha, batch_size=hyperparameters.batch_size, optimizer=hyperparameters.optimizer, cache_validation_data=False)

        return [fine_tuning]



DIPEInsertionSortSearch().search()