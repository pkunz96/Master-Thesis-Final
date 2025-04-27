from math import ceil
import random
from typing import Callable, List, Dict, Tuple
from kneed import KneeLocator

import tensorflow as tf
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import sampling

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment

from nn.multi_domain_nn_forwarding import Layer, Procedure, categorical_cross_entropy_loss, binary_representation_loss, \
    create_contrastive_loss, BinaryLayer, AbstractSearch, MLDGProcedure


def create_training_data(param_set: sampling.ParameterSet, sample_size: int) -> Tuple[tf.Tensor, tf.Tensor, int]:
    # Sampling
    x_np_data, y_np_data = sampling.MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment).sample(sample_size)
    # Scaling
    x_np_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_np_data)
    # Conversion to TensorFlow Tensors
    x_tf_data, y_tf_data = tf.cast(tf.convert_to_tensor(x_np_data), dtype=tf.float32), tf.cast(tf.convert_to_tensor(y_np_data[1:,:]), dtype=tf.float32)
    # Swapping first and last column so that the state is passed through the first layers.
    x_tf_data = tf.gather(x_tf_data, list(range(x_tf_data.shape[1] - 1, -1, -1)), axis=1)
    # One-Hot-Encoding
    y_tf_data_reshaped = tf.reshape(y_tf_data, shape=(y_tf_data.shape[0],))
    unique_values, unique_encoding = tf.unique(y_tf_data_reshaped)
    y_tf_one_hot_data = tf.one_hot(unique_encoding, depth=unique_values.shape[0])
    # Returns the training data - x_tf_data and y_tf_one_hot_data - and the class count.
    return x_tf_data, y_tf_one_hot_data, unique_values.shape[0]


# Data Creation

sample_size: int = 2**10


x_training_data_0, y_training_data_0, class_count = create_training_data(sampling.ParameterSet(10, 3, 5, 0), sample_size)
x_training_data_1, y_training_data_1, class_count = create_training_data(sampling.ParameterSet(5, 1, 5, 0), sample_size)
#x_training_data_2, y_training_data_2, class_count = create_training_data(sampling.ParameterSet(15, 5, 5, 0), sample_size)
#x_training_data_3, y_training_data_3, class_count = create_training_data(sampling.ParameterSet(0, 5, 5, 0), sample_size)
#x_training_data_4, y_training_data_4, class_count = create_training_data(sampling.ParameterSet(25, 5, 5, 0), sample_size)


x_validation_data_0, y_validation_data_0, class_count = create_training_data(sampling.ParameterSet(10, 3, 5, 0), sample_size)
x_validation_data_1, y_validation_data_1, class_count = create_training_data(sampling.ParameterSet(20, 7, 5, 0), sample_size)
x_validation_data_2, y_validation_data_2, class_count = create_training_data(sampling.ParameterSet(4, 1, 5, 0), sample_size)

extractor: List[Layer] = [
    Layer(name="extractor_in", input_dim=x_training_data_0.shape[1], output_dim=128, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="extractor_1", input_dim=128, output_dim=128, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="extractor_out", input_dim=128, output_dim=8, forward=True, loss_function=None, loss_weight=1.0,activation_function_name="relu")
]

predictor: List[Layer] = [
    Layer(name="predictor_in", input_dim=8, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="relu"),
    Layer(name="predictor_in", input_dim=128, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="relu"),

    Layer(name="predictor_out", input_dim=128, output_dim=class_count, forward=False, loss_function=categorical_cross_entropy_loss, loss_weight=1.0, activation_function_name="softmax")
]

model: List[Layer] = extractor + predictor


training_data_dict = {
    "total": (x_training_data_0, y_training_data_0),
    "training1": (x_training_data_1, y_training_data_1),
    #"training2": (x_training_data_2, y_training_data_2),
    #"training3": (x_training_data_3, y_training_data_3),
    #"training4": (x_training_data_4, y_training_data_4),
}


validation_data_dict = {
    "total" : (x_validation_data_0, y_validation_data_0),
    "validation1": (x_validation_data_1, y_validation_data_1),
    "validation2": (x_validation_data_2, y_validation_data_2),

}


procedure: MLDGProcedure = MLDGProcedure(Procedure.compile_model(model), training_data_dict, validation_data_dict, 2000, 0.0001, 128, "adam", gamma=.2, beta=.2)
procedure.train()


procedure: Procedure = Procedure(Procedure.compile_model(model), training_data_dict, validation_data_dict, 2000, 0.0001, 128, "adam")
procedure.train()


#procedure: Procedure = Procedure(Procedure.compile_model(model), x_data, y_data,  x_test, y_test, validation_data_dict=val_data_dict, epochs=1000, learning_rate=0.0001, batch_size=256, optimizer="adam")
#procedure: Procedure = Procedure(Procedure.compile_model(model), x_training, y_training, x_test, y_test, epochs=300, learning_rate=0.0001, batch_size=256, optimizer="adam")


#procedure.disable_metric("training_accuracy")
#procedure.disable_metric("validation_accuracy")
#procedure.disable_metric("training_loss")
#procedure.disable_metric("validation_loss")


first_layer: Layer = procedure.train()

extractor[len(extractor) - 1].successor = None
predictions = first_layer.predict(x_data)

for row in predictions.numpy():
    print([float(val) for val in row])
# Clustering

inertia = []
labels = []
k_range = list(range(1, 100))


for k in k_range:
    y_fine_pred = extractor[0].predict(x_data)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(y_fine_pred)
    inertia.append(kmeans.inertia_)
    labels.append(kmeans.labels_)

kneedle = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
optimal_k = kneedle.elbow

label_list = labels[optimal_k - 1]
cluster_sample_dict = dict()

for index in range(0, len(label_list)):
    label = label_list[index]
    if label not in cluster_sample_dict:
        cluster_sample_dict[label] = [(x_data[index], y_data[index])]
    else:
        cluster_sample_dict[label].append((x_data[index], y_data[index]))

cluster_freq_dic = dict()

for cluster in cluster_sample_dict:
    cluster_freq_dic[cluster] = dict()
    cluster_data_points = cluster_sample_dict[cluster]
    for data_point in cluster_data_points:
        predictor = data_point[0]
        rounded_predictor = tuple(map(float, tf.round(predictor).numpy()))
        if rounded_predictor not in cluster_freq_dic[cluster]:
            cluster_freq_dic[cluster][rounded_predictor] = 1
        else:
            cluster_freq_dic[cluster][rounded_predictor] = cluster_freq_dic[cluster][rounded_predictor] + 1
plt.clf()

for cluster in cluster_freq_dic:
    predictor_freq_dict = cluster_freq_dic[cluster]
    x_labels = list(map(lambda x: str(x), predictor_freq_dict.keys()))
    y_freq = predictor_freq_dict.values()

    fig, ax = plt.subplots()
    ax.set_xlabel('Embeddings')
    ax.set_ylabel('Frequency')
    ax.set_title('Embedding Frequency @ Cluster ' + str(cluster))
    ax.plot(x_labels, y_freq, label='Training Accuracy', marker='o')[0]
    ax.set_xticklabels(x_labels, rotation=45)

    plt.draw()
    plt.pause(0.1)


# Fine-Tuning

contrastive_loss_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = create_contrastive_loss(labels[optimal_k-1], 2.0)

predictor[len(predictor) - 1].loss_weight=0.7

last_extractor_layer: Layer = extractor[len(extractor) - 1]
last_extractor_layer.loss_function = contrastive_loss_function
last_extractor_layer.loss_weight = 1.0
last_extractor_layer.successor = None


model = extractor + predictor

procedure: Procedure = Procedure(Procedure.compile_model(model), x_data, y_data, x_test, y_test, epochs=1000, learning_rate=0.0001, batch_size=64, optimizer="adam")
procedure.train()




# ToDo: Plotting a scatter plot to visualize clusters
# ToDo: Defining a custom layer able to compare inputs using <





