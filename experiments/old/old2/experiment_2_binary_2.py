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


def training_data_generation():
    x,y,z = create_training_data(sampling.ParameterSet(5, 3, 5, 0), 2**10)
    return x, y

def validation_data_generation():
    return dict()

#search = AbstractSearch(10, ["relu"], [128], ["relu"], [128], None, 2, 2, 2, 2, binary_representation_loss, categorical_cross_entropy_loss, [training_data_generation], {"test" : validation_data_generation}, "./test")
#search.search()


"""
    def __init__(self,
                 input_dim: int,
                 extractor_activation_function_list: List[str],
                 extractor_neuron_count_list: List[int],
                 predictor_activation_function_list: List[str],
                 predictor_neuron_count_list: List[int],
                 binary_layer_index: int,
                 min_extractor_layer_count: int,
                 max_extractor_layer_count: int,
                 min_predictor_layer_count: int,
                 max_predictor_layer_count: int,
                 extractor_loss_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
                 predictor_loss_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
                 training_data_generation_procedure_list: List[Callable[[], Tuple[tf.Tensor, tf.Tensor]]],
                 validation_data_generation_procedure_list: Dict[str, Callable[[], Tuple[tf.Tensor, tf.Tensor]]],
                 base_path: str
        ):

"""



# Data Creation


sample_size: int = 2**4


"""

x_data, y_data, class_count = create_training_data(sampling.ParameterSet(10, 3, 5, 0), sample_size)

count = 0
while count < 0:
    cur_x_test, cur_y_test, _ = create_training_data(sampling.ParameterSet(random.randint(1, 20), 6, 5, 0), sample_size)
    x_data = tf.concat([cur_x_test, x_data], axis=0)
    y_data = tf.concat([cur_y_test, y_data], axis=0)
    count = count + 1

x_test, y_test, _ = create_training_data(sampling.ParameterSet(5, 3, 5, 0), sample_size)
"""


#    def __init__(self, problem_mu: int, problem_sigma: int, problem_size_mu: int, problem_size_sigma: int):


# Model Creation

#out_count = unique_values.shape[0]

#binary = BinaryLayer("binary", 128, 256)
x_data, additional_y_val_data, class_count = create_training_data(sampling.ParameterSet(10, 3, 5, 0), 2)

extractor: List[Layer] = [
    Layer(name="extractor_in", input_dim=x_data.shape[1], output_dim=256, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="extractor_1", input_dim=256, output_dim=256, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),

    # binary,
  #  Layer(name="extractor_in", input_dim=8002, output_dim=256, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
  #  Layer(name="extractor_1", input_dim=8002, output_dim=256, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
    Layer(name="extractor_2", input_dim=256, output_dim=128, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="relu"),
    #Layer(name="extractor_1", input_dim=64, output_dim=32, forward=True, loss_function=None, loss_weight=1.0, activation_function_name="sigmoid"),
   # Layer(name="extractor_in", input_dim=256, output_dim=256, forward=True, loss_function=None, loss_weight=1.0,  activation_function_name="sigmoid"),

    Layer(name="extractor_out", input_dim=128, output_dim=8, forward=True, loss_function=None, loss_weight=.4,activation_function_name="relu")
]

predictor: List[Layer] = [
    Layer(name="predictor_in", input_dim=8, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="relu"),
    Layer(name="predictor_1", input_dim=128, output_dim=128, forward=False, loss_function=None, loss_weight=1.0, activation_function_name="relu"),
    Layer(name="predictor_out", input_dim=128, output_dim=class_count, forward=False, loss_function=categorical_cross_entropy_loss, loss_weight=1.0, activation_function_name="softmax")
]

model: List[Layer] = extractor + predictor


# Pretraining
additional_x_val_data, additional_y_val_data, _ = create_training_data(sampling.ParameterSet(10, 3, 5, 0), sample_size)



training_data_dict = {"total" : (additional_x_val_data, additional_y_val_data), "other0" : (additional_x_val_data, additional_y_val_data), "other1" : (additional_x_val_data, additional_y_val_data), "other2" : (additional_x_val_data, additional_y_val_data)}

validation_data_dict = {"total" : (additional_x_val_data, additional_y_val_data)}




procedure: MLDGProcedure = MLDGProcedure(Procedure.compile_model(model), training_data_dict, validation_data_dict, 2000, 0.001, 128, "adam", gamma=.2, beta=.2)
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





