from typing import List, Dict, Tuple

import psutil
import os
import time
import threading

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from algorithms.straight_insertion_sort import gen_insertion_sort_environment, straight_insertion_sort
from experiments.sampling import ParameterSet, MonteCarloSampling

def map_sample_to_density_kde_safe(
    predictor_arr: np.ndarray,
    response_arr: np.ndarray,
    decimals: int = 0
) -> Tuple[Dict[Tuple[int], float], Dict[Tuple[int], float]]:
    joint_arr = np.hstack((predictor_arr, response_arr))

    def full_rank_subset(arr: np.ndarray) -> np.ndarray:
        # Select maximal linearly independent subset of features
        u, s, vh = np.linalg.svd(arr - arr.mean(axis=0), full_matrices=False)
        rank = np.sum(s > 1e-10)
        return arr @ vh[:rank].T  # project to full-rank subspace

    try:
        joint_kde = gaussian_kde(joint_arr.T)
        joint_pdf = joint_kde(joint_arr.T)
    except np.linalg.LinAlgError:
        reduced_joint = full_rank_subset(joint_arr)
        joint_kde = gaussian_kde(reduced_joint.T)
        joint_pdf = joint_kde(reduced_joint.T)

    try:
        predictor_kde = gaussian_kde(predictor_arr.T)
        predictor_pdf = predictor_kde(predictor_arr.T)
    except np.linalg.LinAlgError:
        reduced_predictor = full_rank_subset(predictor_arr)
        predictor_kde = gaussian_kde(reduced_predictor.T)
        predictor_pdf = predictor_kde(reduced_predictor.T)

    # Normalize to get probability mass estimates
    joint_pdf /= joint_pdf.sum()
    predictor_pdf /= predictor_pdf.sum()

    def arr_to_dict(arr, pdf) -> Dict[Tuple[int], float]:
        d = {}
        for row, prob in zip(arr, pdf):
            key = tuple(np.round(row, decimals=decimals).astype(int))
            d[key] = d.get(key, 0.0) + float(prob)
        return d

    joint_dict = arr_to_dict(joint_arr, joint_pdf)
    predictor_dict = arr_to_dict(predictor_arr, predictor_pdf)

    return joint_dict, predictor_dict

def calc_jensen_shannon_divergence(dens_0: Dict[Tuple[int...], float], dens_1: Dict[Tuple[int], float]) -> float:
    joint_key_set = set(dens_0.keys()) | set(dens_1.keys())
    dens_0_mass_vector = []
    dens_1_mass_vector = []
    for key in joint_key_set:
        if key in dens_0:
            dens_0_mass_vector.append(dens_0[key])
        else:
            dens_0_mass_vector.append(0.0)
        if key in dens_1:
            dens_1_mass_vector.append(dens_1[key])
        else:
            dens_1_mass_vector.append(0.0)
    return jensenshannon(np.array(dens_0_mass_vector, dtype=np.float64), np.array(dens_1_mass_vector, dtype=np.float64), base=2)


def create_parameter_sets(min_mu: int, max_mu: int, mu_step_size: int, min_sigma: int, max_sigma: int, sigma_step_size: int) -> List[ParameterSet]:
    param_set_list = []
    for mu in range(min_mu,  max_mu + 1, mu_step_size):
        for sigma in range(min_sigma, max_sigma + 1, sigma_step_size):
            param_set_list.append(ParameterSet(mu, sigma, 5, 0))
    return param_set_list

def determine_sample_size(param_set_list, epsilon) -> int:
    sample_size = 8
    found = False
    while not found:
        sample_size *= 2
        found = True
        for param_set in param_set_list:
            for index in range(5):
                mcs_0: MonteCarloSampling = MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment)
                mcs_0.init()
                mcs_1: MonteCarloSampling = MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment)
                mcs_1.init()
                pred_arr_0, resp_arr_0 = mcs_0.sample(sample_size)
                pred_arr_1, resp_arr_1 = mcs_1.sample(sample_size)
                outer_joint_density, outer_predictor_density = map_sample_to_density_kde_safe(pred_arr_0, resp_arr_0)
                inner_joint_density, inner_predictor_density = map_sample_to_density_kde_safe(pred_arr_1, resp_arr_1)
                distance = calc_jensen_shannon_divergence(outer_joint_density, inner_joint_density)
                if distance >= epsilon:
                    found = False
                    break
            if not found:
                break
    return sample_size


def run(min_mu: int, max_mu: int, mu_step_size: int, min_sigma: int, max_sigma: int, sigma_step_size) -> Tuple[NDArray, List[str]]:

    param_set_list = []

    for mu in range(min_mu,  max_mu + 1, mu_step_size):
        for sigma in range(min_sigma, max_sigma + 1, sigma_step_size):
            param_set_list.append(ParameterSet(mu, sigma, 5, 0)) # Tuples of length 5.

    distance_map_data = []
    y_label = ["μ=" + str(param_set.problem_mu) + "_σ=" + str(param_set.problem_sigma) for param_set in param_set_list]
    x_label = list(y_label)

    sample_size = determine_sample_size(param_set_list, 0.2)

    for out_param_index in range(len(param_set_list)):
        distance_map_data.append([])
        outer_param_set = param_set_list[out_param_index]
        for inner_param_index in range(len(param_set_list)):
            if inner_param_index < out_param_index:
                distance_map_data[out_param_index].append(np.nan)
            else:
                inner_param_set = param_set_list[inner_param_index]
                mcs_out: MonteCarloSampling = MonteCarloSampling(outer_param_set, straight_insertion_sort, gen_insertion_sort_environment)
                mcs_in: MonteCarloSampling = MonteCarloSampling(inner_param_set, straight_insertion_sort, gen_insertion_sort_environment)
                mcs_out.init()
                mcs_in.init()
                outer_predictor_arr, outer_response_arr = mcs_out.sample(sample_size)
                inner_predictor_arr, inner_response_arr = mcs_in.sample(sample_size)
                outer_joint_density, outer_predictor_density = map_sample_to_density_kde_safe(outer_predictor_arr, outer_response_arr)
                inner_joint_density, inner_predictor_density = map_sample_to_density_kde_safe(inner_predictor_arr, inner_response_arr)
                distance = calc_jensen_shannon_divergence(outer_joint_density, inner_joint_density)
                distance_map_data[out_param_index].append(distance)

    distance_array = np.array(distance_map_data)
#    distance_array = np.where(distance_array == -1.0, np.nan, distance_array)

    # Create the heatmap

    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_array, xticklabels=x_label, yticklabels=y_label, annot=False, cmap='viridis', mask=np.isnan(distance_array), cbar_kws={'label': 'Jensen-Shannon Divergence'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def get_memory_usage():
    # Get the current process's memory usage in bytes
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss  # Resident Set Size (RSS), which is the non-swapped physical memory the process has used

def print_memory_profile(interval=1):
    while True:
        memory_usage = get_memory_usage()
        print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")  # Convert bytes to MB
        time.sleep(interval)

def some_other_task():
    # Example of another task running concurrently
    while True:
        print("Performing some other task...")
        time.sleep(5)

def start_memory_thread(interval=2):
    memory_thread = threading.Thread(target=print_memory_profile, args=(interval,))
    memory_thread.daemon = True  # Daemon thread will exit when the main program exits
    memory_thread.start()


start_memory_thread()
run(0, 50, 50, 0, 50, 50)


