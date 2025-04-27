import os
import time
from typing import List, Dict, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from algorithms.straight_insertion_sort import gen_insertion_sort_environment, straight_insertion_sort
from experiments.sampling import ParameterSet, MonteCarloSampling


def map_sample_to_density_kde_safe( predictor_arr:  NDArray, response_arr: NDArray, decimals: int = 0) -> Tuple[Dict[Tuple[int], float], Dict[Tuple[int], float]]:

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


def calc_jensen_shannon_divergence(dens_0: Dict[Tuple[int], float], dens_1: Dict[Tuple[int], float]) -> float:
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


def estimate_sample_size(param_set_list, epsilon) -> int:
    sample_size = 8
    found = False
    while not found:
        sample_size *= 2
        found = True
        for param_set in param_set_list:
            for index in range(5):
                mcs_0: MonteCarloSampling = MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment)
                mcs_0.init(caching=True)
                mcs_1: MonteCarloSampling = MonteCarloSampling(param_set, straight_insertion_sort, gen_insertion_sort_environment)
                mcs_1.init(caching=True)
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


def measure_distance(param_set_list: List[ParameterSet], sample_size: int) -> NDArray:
    distance_map_data = []
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
                mcs_out.init(caching=False)
                mcs_in.init(caching=False)
                outer_predictor_arr, outer_response_arr = mcs_out.sample(sample_size)
                inner_predictor_arr, inner_response_arr = mcs_in.sample(sample_size)
                outer_joint_density, outer_predictor_density = map_sample_to_density_kde_safe(outer_predictor_arr, outer_response_arr)
                inner_joint_density, inner_predictor_density = map_sample_to_density_kde_safe(inner_predictor_arr, inner_response_arr)
                distance = calc_jensen_shannon_divergence(outer_joint_density, inner_joint_density)
                distance_map_data[out_param_index].append(distance)
    return np.array(distance_map_data)


def save_and_visualize(param_set_list: List[ParameterSet], distance_matrix: NDArray, sample_size: int, epsilon: float) -> None:
    labels = ["μ=" + str(param_set.problem_mu) + "_σ=" + str(param_set.problem_sigma) for param_set in param_set_list]

    directory = "data_experiment_1"
    os.makedirs(directory, exist_ok=True)
    filename = directory + "/experiment_1_" + str(time.time())

    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, xticklabels=labels, yticklabels=labels, annot=False, cmap='viridis', mask=np.isnan(distance_matrix), cbar_kws={'label': 'Jensen-Shannon Divergence (ε=' + str(epsilon) + ', sample_size=' + str(sample_size) + ')'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')

    df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    df.to_csv(filename + ".csv")

    plt.show()

print(1)

epsilon = 0.01

print(2)
parameter_set_list = create_parameter_sets(0, 10, 10, 0, 10, 10)

print(3)
estimated_sample_size = estimate_sample_size(parameter_set_list, epsilon)

print(4)
distance_matrix = measure_distance(parameter_set_list, estimated_sample_size)

print(5)
save_and_visualize(parameter_set_list, distance_matrix, estimated_sample_size, epsilon)