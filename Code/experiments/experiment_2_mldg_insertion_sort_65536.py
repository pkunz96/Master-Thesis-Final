from itertools import product
from typing import Callable, List, Tuple, Dict, Optional
from kneed import KneeLocator

import tensorflow as tf
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Categorical

from sampling import ParameterSet

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment

from nn.multi_domain_nn_forwarding import Layer, Procedure, categorical_cross_entropy_loss, \
    binary_representation_loss, \
    create_contrastive_loss, BinaryLayer, AbstractSearch, BayesianSearch, Hyperparameters, Configuration, MLDGProcedure, \
    HyperoptBayesianSearch


class MLDGInsertionSortSearch(HyperoptBayesianSearch):

    base_dir = "./mldg_insertion_sort_65536"

    bayesian_sample_count = 100

    training_parameter_list: List[ParameterSet] = [ParameterSet(10,5,10,0), ParameterSet(15,5,10,0), ParameterSet(15,10,10,0)]

    validation_parameter_list: List[ParameterSet] = [ParameterSet(10,5,10,0), ParameterSet(15,5,10,0), ParameterSet(15,10,10,0)]

    test_parameter_list: List[ParameterSet] = [ParameterSet(10,5,10,0), ParameterSet(100, 20, 10, 0)]

    def __init__(self):
        super().__init__(MLDGInsertionSortSearch.training_parameter_list, MLDGInsertionSortSearch.validation_parameter_list, MLDGInsertionSortSearch.test_parameter_list, MLDGInsertionSortSearch.base_dir, MLDGInsertionSortSearch.bayesian_sample_count, gen_insertion_sort_environment, straight_insertion_sort)

    def _build_search_space(self) -> List[Tuple[Configuration, Hyperparameters]]:
        search_space = []
        # 1 to 3 hidden layers
        for hidden_layer_counter in range(1, 4):
            for activation_func_list in list(product(["relu", "sigmoid"], repeat=hidden_layer_counter + 1)):
                activation_func_list = activation_func_list + ("softmax",)
                #128 to 1024 neuron per hidden layer
                for neuron_count_pow_base_2 in range(7, 11):
                    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
                        for beta in [0.1, 0.01, 0.001, 0.0001]:
                            for gamma in [0.1, 0.01, 0.001, 0.0001]:
                                neuron_count_arr = []
                                for count in range(0, hidden_layer_counter):
                                    neuron_count_arr += [2**neuron_count_pow_base_2]
                                neuron_count_arr = [1] + neuron_count_arr + [1]
                                loss_func_arr = []
                                forwarding_arr = []
                                binary_arr = []
                                for count in range(0, hidden_layer_counter + 2):
                                    if count == hidden_layer_counter + 1:
                                        loss_func_arr.append(categorical_cross_entropy_loss)
                                    else:
                                        loss_func_arr.append(None)
                                    forwarding_arr.append(False)
                                    binary_arr.append(False)
                                search_space.append((Configuration(neuron_count_arr, activation_func_list,loss_func_arr, forwarding_arr, binary_arr), Hyperparameters(learning_rate, beta, gamma, 1000, 64, "adam", 100, sample_size=65536, iterations=5)))
        print("Search Space Size: " + str(len(search_space)))
        return search_space

    def _create_initial_procedure_builder(self, model: Layer, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> Callable[[], Procedure]:
        def pretraining():
            return MLDGProcedure(model, training_data_dict, validation_data_dict, test_data_dict, epochs=hyperparameters.epochs , batch_size=hyperparameters.batch_size, optimizer=hyperparameters.optimizer, early_stopping=hyperparameters.early_stopping,  alpha = hyperparameters.alpha, beta = hyperparameters.beta, gamma = hyperparameters.gamma)
        return pretraining

    def _create_subsequent_procedure_builders(self, hyperparameters: Hyperparameters, training_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], validation_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]], test_data_dict: Dict[str, Tuple[tf.Tensor, tf.Tensor]]) -> List[Callable[[Procedure], Procedure]]:
        return []


MLDGInsertionSortSearch().search()