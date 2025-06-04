import ast
import csv
import os
import re
from functools import reduce

from h5py.h5t import NORM_NONE
from tensorflow import split


def resolve_path(base_dir: str, filename: str):
    if not base_dir.endswith("/"):
        base_dir = base_dir + "/"
    return base_dir + filename


def find_stage_dirs(base_dir: str):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pattern = re.compile(r'^stage_\d+$')
    return list(map(lambda x: resolve_path(base_dir, x), filter(lambda x: pattern.match(x), subdirs)))


def read_configuration(dir: str):
    configuration = tuple()
    conf_path = resolve_path(dir, "configuration.txt")
    with open(conf_path, 'r') as file:
        for line in file:
            value = line.split(":")[1].strip()
            configuration += (value,)
    return configuration


def read_hyperparameter(dir: str):
    hyperparamters = tuple()
    conf_path = resolve_path(dir, "hyperparameters.txt")
    with open(conf_path, 'r') as file:
        for line in file:
            value = line.split(":")[1].strip()
            hyperparamters += (value,)
    return hyperparamters


def fetch_avg_epoch_count(dir: str):
    epochs = None
    epochs_path = resolve_path(dir, "epochs.txt")
    with open(epochs_path, 'r') as file:
        for line in file:
            epochs =  float(line)
            break
    return epochs


def parse_float_list(line: str):
    line = line.replace("[","").replace("]", "")
    return list(map(lambda x: float(x.strip()), line.split(",")))


def fetch_name_value_list_dict(file_path: str):
    name_value_list_dict = dict()
    file_content_str = ""
    with open(file_path, 'r') as file:
        for line in file:
            file_content_str += line
    file_content_str = file_content_str.replace("{", "").replace("}", "")
    name_value_list_str = file_content_str.split("],")
    for name_value_list_tuple in name_value_list_str:
        if not name_value_list_tuple.endswith("]"):
            name_value_list_tuple += "]"
        name_value_list_arr =  name_value_list_tuple.split(":")
        name = name_value_list_arr[0]
        value_list_arr = name_value_list_arr[1]
        value_arr = parse_float_list(value_list_arr)
        name_value_list_dict[name] = value_arr
    return name_value_list_dict

def fetch_avg_training_accuracy(dir: str):
    value_list = []
    training_accuracy_path = resolve_path(dir, "accuracy/training_accuracy.txt")
    with open(training_accuracy_path, 'r') as file:
        for line in file:
            value_list = parse_float_list(line)
            break
    return value_list


def fetch_avg_validation_accuracy(dir: str):
    value_list = []
    training_accuracy_path = resolve_path(dir, "accuracy/validation_accuracy.txt")
    with open(training_accuracy_path, 'r') as file:
        for line in file:
            value_list = parse_float_list(line)
            break
    return value_list


def fetch_avg_test_accuracy(dir: str):
    avg_test_accuracy = 0.0
    training_accuracy_path = resolve_path(dir, "accuracy/test_accuracy.txt")
    with open(training_accuracy_path, 'r') as file:
        for line in file:
            line = line.replace("tf.Tensor(", "").replace(", shape=(), dtype=float32)", "").strip()
            avg_test_accuracy = float(line)
            break
    return avg_test_accuracy


def fetch_data_name_avg_test_accuracy(dir: str):
    data_name_avg_test_accuracy_dict = dict()
    training_accuracy_path = resolve_path(dir, "accuracy/data_name_avg_test_accuracy.txt")
    with open(training_accuracy_path, 'r') as file:
        for line in file:
            line_arr = line.split(":")
            name = line_arr[0].strip()
            tensor_str = line_arr[1].strip().replace("tf.Tensor(", "").replace(", shape=(), dtype=float32)", "").strip()
            data_name_avg_test_accuracy_dict[name] = float(tensor_str)
    return data_name_avg_test_accuracy_dict


def fetch_data_name_avg_val_accuracy(dir: str):
    training_accuracy_path = resolve_path(dir, "accuracy/data_name_validation_accuracy.txt")
    return fetch_name_value_list_dict(training_accuracy_path)


def fetch_avg_validation_loss(dir: str):
    value_list = []
    training_accuracy_path = resolve_path(dir, "loss/validation_loss.txt")
    with open(training_accuracy_path, 'r') as file:
        for line in file:
            value_list = parse_float_list(line)
            break
    return value_list


def fetch_avg_training_loss(dir: str):
    value_list = []
    training_accuracy_path = resolve_path(dir, "loss/training_loss.txt")
    with open(training_accuracy_path, 'r') as file:
        for line in file:
            value_list = parse_float_list(line)
            break
    return value_list

def fetch_data_name_avg_val_loss(dir: str):
    training_accuracy_path = resolve_path(dir, "loss/data_name_validation_loss.txt")
    return fetch_name_value_list_dict(training_accuracy_path)


def create_stage_report(dir: str):
    epochs = fetch_avg_epoch_count(dir)
    avg_training_accuracy_list = fetch_avg_training_accuracy(dir)
    avg_validation_accuracy_list = fetch_avg_validation_accuracy(dir)
    avg_test_accuracy = fetch_avg_test_accuracy(dir)
    data_name_test_accuracy = fetch_data_name_avg_test_accuracy(dir)
    data_name_val_accuracy = fetch_data_name_avg_val_accuracy(dir)
    avg_validation_loss = fetch_avg_validation_loss(dir)
    avg_training_loss = fetch_avg_training_loss(dir)
    data_name_avg_val_loss = fetch_data_name_avg_val_loss(dir)
    return epochs, avg_training_accuracy_list, avg_validation_accuracy_list, avg_test_accuracy, data_name_test_accuracy, data_name_val_accuracy, avg_validation_loss, avg_training_loss, data_name_avg_val_loss

def calc_slope(arr):
    sample_size = min([5, len(arr)])
    initial_sample = arr[0:sample_size]
    terminal_sample = arr[len(arr) - (sample_size):]
    initial_avg = sum(initial_sample) / sample_size
    terminal_avg = sum(terminal_sample) / sample_size
    return (terminal_avg - initial_avg) / len(arr)

def calc_slope_per_list(value_dict):
    result_dict = {}
    for name in value_dict:
        result_dict[name] = calc_slope(value_dict[name])
    return result_dict

def find_min_entry(value_dict):
    name = None
    value = None
    for cur_name in value_dict:
        if value is None or value > value_dict[cur_name]:
            name = cur_name
            value = value_dict[cur_name]
    return name, value

def find_max_entry(value_dict):
    name = None
    value = None
    for cur_name in value_dict:
        if value is None or value < value_dict[cur_name]:
            name = cur_name
            value = value_dict[cur_name]
    return name, value

directory_list = ["/media/pkunz/Expansion/experiment_2/dipe_insertion_sort_16384/05-14 02_17_[10, 256, 129, 3, 64, 64, 2]_('sigmoid', 'relu', 'relu', 'relu', 'relu', 'sigmoid', 'softmax')_[None, None, None, binary_representation_loss, None, None, categorical_cross_entropy_loss]"]

def find_result_directories(base_path):
    dirs_with_configuration = []
    base_depth = base_path.rstrip(os.path.sep).count(os.path.sep)

    for root, dirs, files in os.walk(base_path):
        current_depth = root.count(os.path.sep) - base_depth
        if current_depth > 2:
            dirs[:] = []  # Prevent deeper traversal
            continue
        if 'configuration.txt' in files:
            dirs_with_configuration.append(root)

    return dirs_with_configuration


base_dir = input("Base:")

if not base_dir.endswith("/"):
    base_dir += "/"

directory_list = find_result_directories(base_dir)


with open(base_dir + "results.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    field_arr = [
        "group_name",
        "neuron_count_list",
        "activation_func_list",
        "forwarding_func_list",
        "loss_func_list",
        "alpha",
        "beta",
        "gamma",
        "iterations",
        "sample_size",
        "epochs",
        "batch_size",
        "optimizer",
        "early_stopping"
    ]
    stage_field_arr = [
        "avg_epoch_count",
        "avg_training_accuracy_slope",
        "avg_validation_accuracy_slope",
        "avg_test_accuracy",
        "min_data_set_test_accuracy_name",
        "min_data_set_test_accuracy_value",
        "max_data_set_test_accuracy_name",
        "max_data_set_test_accuracy_value",
        "min_data_set_validation_accuracy_slope_name",
        "min_data_set_validation_accuracy_slop_value",
        "max_data_set_validation_accuracy_slope_name",
        "max_data_set_validation_accuracy_slop_value",
        "avg_training_loss_slope",
        "avg_validation_loss_slope",
        "min_data_set_validation_loss_slope_name",
        "min_data_set_validation_loss_slop_value",
        "max_data_set_validation_loss_slope_name",
        "max_data_set_validation_loss_slop_value",
    ]

    max_stage_count = 1

    for directory in directory_list:
        stage_list = find_stage_dirs(directory)
        max_stage_count = max([len(stage_list), max_stage_count])

    field_arr += (stage_field_arr*max_stage_count)

    writer.writerow(field_arr)

    for directory in directory_list:
        try:
            row = []
            group_name = directory.replace(base_dir, "").split("/")[0]
            row.append(group_name)
            configuration = read_configuration(directory)
            row.append(configuration[0])
            row.append(configuration[1])
            row.append(configuration[2])
            row.append(configuration[3])
            hyperparameters = read_hyperparameter(directory)
            row.append(hyperparameters[0])
            row.append(hyperparameters[1])
            row.append(hyperparameters[2])
            row.append(hyperparameters[3])
            row.append(hyperparameters[4])
            row.append(hyperparameters[5])
            row.append(hyperparameters[6])
            row.append(hyperparameters[7])
            row.append(hyperparameters[8])
            for stage_dir in find_stage_dirs(directory):
                avg_epoch, avg_training_accuracy_list, avg_validation_accuracy_list, avg_test_accuracy, data_name_test_accuracy, data_name_val_accuracy, avg_validation_loss, avg_training_loss, data_name_avg_val_loss = create_stage_report(stage_dir)
                row.append(avg_epoch)
                row.append(calc_slope(avg_training_accuracy_list))
                row.append(calc_slope(avg_validation_accuracy_list))
                row.append(avg_test_accuracy)
                min_data_set_test_accuracy_name, min_data_set_test_accuracy_value = find_min_entry(data_name_test_accuracy)
                row.append(min_data_set_test_accuracy_name)
                row.append(min_data_set_test_accuracy_value)
                max_data_set_test_accuracy_name, max_data_set_test_accuracy_value = find_max_entry(data_name_test_accuracy)
                row.append(max_data_set_test_accuracy_name)
                row.append(max_data_set_test_accuracy_value)
                min_data_set_validation_accuracy_name, min_data_set_validation_accuracy_value = find_min_entry(calc_slope_per_list(data_name_val_accuracy))
                row.append(min_data_set_validation_accuracy_name)
                row.append(min_data_set_validation_accuracy_value)
                max_data_set_validation_accuracy_name, max_data_set_validation_accuracy_value = find_max_entry(calc_slope_per_list(data_name_val_accuracy))
                row.append(max_data_set_validation_accuracy_name)
                row.append(max_data_set_validation_accuracy_value)
                row.append(calc_slope(avg_training_loss))
                row.append(calc_slope(avg_validation_loss))
                min_data_set_validation_loss_slope_name, min_data_set_validation_loss_slop_value = find_min_entry(calc_slope_per_list(data_name_avg_val_loss))
                row.append(min_data_set_validation_loss_slope_name)
                row.append(min_data_set_validation_loss_slop_value)
                max_data_set_validation_loss_slope_name, max_data_set_validation_loss_slop_value = find_max_entry(calc_slope_per_list(data_name_avg_val_loss))
                row.append(max_data_set_validation_loss_slope_name)
                row.append(max_data_set_validation_loss_slop_value)
            writer.writerow(row)
        except Exception:
            print("Could not evaluate " + directory)






