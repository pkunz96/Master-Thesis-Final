from typing import List, Tuple

from sklearn.preprocessing import MinMaxScaler

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment
from experiments import sampling
from experiments.sampling import ParameterSet
from nn.algo_learning_dg_framework import AbstractSearch, MLDGProcedure, Procedure, Layer, \
    binary_representation_loss, categorical_cross_entropy_loss, FishProcedure
import tensorflow as tf


def create_training_data(param_set: sampling.ParameterSet, sample_size: int) -> Tuple[tf.Tensor, tf.Tensor, int]:
    # Sampling
    mcs:  sampling.MonteCarloSampling =  sampling.MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment)
    mcs.init()
    x_np_data, y_np_data = mcs.sample(sample_size)
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
    return x_tf_data, y_tf_one_hot_data, unique_values.shape[0]


# Data Creation

sample_size: int = 2**4

x_training_data_0, y_training_data_0, class_count = create_training_data(sampling.ParameterSet(20, 5, 0, 100, suffix="quick_sort"), sample_size)
x_training_data_1, y_training_data_1, class_count = create_training_data(sampling.ParameterSet(20, 5, 50, 100, suffix="quick_sort"), sample_size)
x_training_data_2, y_training_data_2, class_count = create_training_data(sampling.ParameterSet(20, 5, 100, 100, suffix="quick_sort"), sample_size)
x_training_data_3, y_training_data_3, class_count = create_training_data(sampling.ParameterSet(20, 5, 150, 100, suffix="quick_sort"), sample_size)


"""
    training_parameter_list: List[ParameterSet] = [
        ParameterSet(20, 5, 0, 100, suffix=suffix),   # mu = 0 and sigma = 100
        ParameterSet(20, 5, 50, 100, suffix=suffix),  # mu = 50 and sigma = 100
        ParameterSet(20, 5, 100, 100, suffix=suffix), # mu = 100 and sigma = 100
        ParameterSet(20, 5, 150, 100, suffix=suffix), # mu = 150 and sigma = 100
        ParameterSet(20, 5, 0, 50, suffix=suffix),  # mu = 0 and sigma = 50
        ParameterSet(20, 5, 0, 150, suffix=suffix),   # mu = 0 and sigma = 150
    ]

    validation_parameter_list: List[ParameterSet] = [
        ParameterSet(20, 5, 0, 100, suffix=suffix),   # mu = 0 and sigma = 100
        ParameterSet(20, 5, 50, 100, suffix=suffix),  # mu = 50 and sigma = 100
        ParameterSet(20, 5, 100, 100, suffix=suffix), # mu = 100 and sigma = 100
        ParameterSet(20, 5, 150, 100, suffix=suffix), # mu = 150 and sigma = 100
        ParameterSet(20, 5, 0, 50, suffix=suffix),   # mu = 0 and sigma = 50
        ParameterSet(20, 5, 0, 150, suffix=suffix),   # mu = 0 and sigma = 150 
    ]
"""

x_validation_data_0, y_validation_data_0, class_count = create_training_data(sampling.ParameterSet(20, 5, 0, 50, suffix="quick_sort"), sample_size)
x_validation_data_1, y_validation_data_1, class_count = create_training_data(sampling.ParameterSet(20, 5, 0, 150,  suffix="quick_sort"), sample_size)

extractor: List[Layer] = [
    Layer(name="extractor_in", input_dim=x_training_data_0.shape[1], output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="extractor_1", input_dim=128, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="extractor_out", input_dim=128, output_dim=256, forward=False, loss_function=None, loss_weight=1.0,activation_function_name="sigmoid")
]

predictor: List[Layer] = [
    Layer(name="predictor_in", input_dim=256, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="predictor_in", input_dim=128, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="predictor_out", input_dim=128, output_dim=class_count, forward=False, loss_function=categorical_cross_entropy_loss, loss_weight=1.0, activation_function_name="softmax")
]

model: List[Layer] = extractor + predictor

training_data_dict = {
    "training0": (x_training_data_0, y_training_data_0),
    #"training1": (x_training_data_1, y_training_data_1),
    #"training2": (x_training_data_2, y_training_data_2),
   # "training3": (x_training_data_3, y_training_data_3),

}

validation_data_dict = {
    "validation0" : (x_validation_data_0, y_validation_data_0),
    #"validation1": (x_validation_data_1, y_validation_data_1),
}

procedure: Procedure = Procedure(Procedure.compile_model(model), training_data_dict, validation_data_dict, validation_data_dict, 1000, 0.0001, 128, "adam", 1000)
procedure.train()