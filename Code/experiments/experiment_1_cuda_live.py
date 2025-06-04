import os
import time
from typing import List, Dict, Tuple, Callable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import jensenshannon

from algorithms.straight_insertion_sort import gen_insertion_sort_environment, straight_insertion_sort
from experiments.sampling import ParameterSet, MonteCarloSampling
import cupy
from cuml import KernelDensity


def silverman_bandwidth(np_arr: NDArray) -> float:
    n, d = np_arr.shape
    averaged_standard_deviation = np.mean( np.std(np_arr, axis=0, ddof=1))
    h = ((4 / (d + 2)) ** (1 / (d + 4))) * n ** (-1 / (d + 4)) * averaged_standard_deviation
    return h


def project_svd(cuda_arr: cupy.ndarray) -> cupy.ndarray:
    print("Enter SVD")
    cur_time = time.time()
    centered = cuda_arr - cupy.mean(cuda_arr, axis=0)
    u, s, vh = cupy.linalg.svd(centered, full_matrices=False)
    rank = cupy.sum(s > 1e-10)
    reduced = centered @ vh[:rank].T
    print("Exit SVD after " + str(time.time() - cur_time))
    return reduced


def kde_batched(kde, data, batch_size):
    results = []
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        scores = kde.score_samples(batch)
        results.append(scores)
        cupy.get_default_memory_pool().free_all_blocks()  # Optional inside loop
    return cupy.concatenate(results)


def estimate_density(np_arr: NDArray, decimals: int = 0, batch_size: int = 512) -> Dict[Tuple[int], float]:
    cuda_arr = cupy.asarray(np_arr).astype('float32')
    try:
        kde = KernelDensity(bandwidth=silverman_bandwidth(np_arr))
        kde.fit(cuda_arr)
        log_pdf_cp = kde_batched(kde, cuda_arr, batch_size)
        pdf_cp = cupy.exp(log_pdf_cp)
    except Exception:
        reduced_arr = project_svd(cuda_arr).astype('float32')
        kde = KernelDensity(bandwidth=silverman_bandwidth(reduced_arr))
        kde.fit(reduced_arr)
        log_pdf_cp = kde_batched(kde, reduced_arr, batch_size)
        pdf_cp = cupy.exp(log_pdf_cp)
    pdf_cp /= cupy.sum(pdf_cp)
    np_arr = cupy.asnumpy(cuda_arr)
    pdf_cpu = cupy.asnumpy(pdf_cp)
    d: Dict[Tuple[int], float] = {}
    for row, prob in zip(np_arr, pdf_cpu):
        key = tuple(np.round(row, decimals=decimals).astype(int))
        d[key] = d.get(key, 0.0) + float(prob)
    cupy.get_default_memory_pool().free_all_blocks()
    return d

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
    return jensenshannon(
        np.array(dens_0_mass_vector, dtype=np.float64),
        np.array(dens_1_mass_vector, dtype=np.float64), base=2
    )


def create_parameter_sets(
        min_mu: int,
        max_mu: int,
        mu_step_size: int,
        min_sigma: int,
        max_sigma: int,
        sigma_step_size: int
) -> List[ParameterSet]:
    param_set_list = []
    for mu in range(min_mu,  max_mu + 1, mu_step_size):
        for sigma in range(min_sigma, max_sigma + 1, sigma_step_size):
            param_set_list.append(ParameterSet(mu, sigma, 5, 0))
    return param_set_list


def sample(
        parameter_set: ParameterSet,
        sample_size: int,
        algorithm,
        environment,
        error_estimate_count: int
) -> Tuple[Dict[Tuple[int], float], float]:
    density: Dict[Tuple[int], float] = {}
    error: float = 0.0
    for index in range(error_estimate_count):
        mcs_0: MonteCarloSampling = MonteCarloSampling(parameter_set, algorithm, environment)
        mcs_0.init(caching=False, sampling_size=sample_size)
        mcs_1: MonteCarloSampling = MonteCarloSampling(parameter_set, algorithm, environment)
        mcs_1.init(caching=False, sampling_size=sample_size)
        pred_arr_0, resp_arr_0 = mcs_0.sample(sample_size)
        pred_arr_1, resp_arr_1 = mcs_1.sample(sample_size)
        pred_density_0 = estimate_density(pred_arr_0)
        pred_density_1 = estimate_density(pred_arr_1)
        error += calc_jensen_shannon_divergence(pred_density_0, pred_density_1)
        density = pred_density_0
    error /= error_estimate_count
    return density, error


def measure_distance(
        param_set_list: List[ParameterSet],
        algorithm,
        environment,
        error_estimate_count: int = 1,
        min_sample_size_exp: int = 8,
        max_sample_size_exp: int = 20,
        on_completed: Callable[[List[ParameterSet], int, float, NDArray], None] = None
) -> Dict[int, Tuple[float, NDArray]]:
    measurements_dict: Dict[int, Tuple[float, NDArray]] = {}
    for sample_size_exp in range(min_sample_size_exp, max_sample_size_exp + 1):
        sample_size = 2**sample_size_exp
        error = 0.0
        distance_map = []
        print("Sample Size: " + str(sample_size))
        for out_param_index in range(len(param_set_list)):
            distance_map.append([])
            outer_density, outer_density_error\
                = sample(param_set_list[out_param_index],
                         sample_size, algorithm, environment, error_estimate_count)
            for inner_param_index in range(len(param_set_list)):
                if inner_param_index < out_param_index:
                    distance_map[out_param_index].append(np.nan)
                else:
                    inner_density, inner_density_error \
                        = sample(param_set_list[inner_param_index],
                                 sample_size, algorithm, environment, error_estimate_count)
                    distance = calc_jensen_shannon_divergence(outer_density, inner_density)
                    error = max([outer_density_error, inner_density_error, error])
                    distance_map[out_param_index].append(distance)
            print("Sample Size: " + str((out_param_index + 1) / len(param_set_list)) + " %")
        np_distance_map = np.array(distance_map)
        measurements_dict[sample_size] = (error, np_distance_map)
        if on_completed is not None:
            on_completed(param_set_list, sample_size, error, np_distance_map)
    return measurements_dict


def create_on_completed(base_dir: str):

    if not base_dir.endswith("/"):
        base_dir += "/"
    base_dir += (str(time.time()) + "/")
    os.makedirs(base_dir, exist_ok=True)

    def on_completed(
            parameter_set_list: List[ParameterSet],
            sample_size: int,
            error: float,
            distance_map: NDArray
    ) -> None:
        labels = ["μ=" + str(param_set.problem_mu) + "_σ=" + str(param_set.problem_sigma)
                  for param_set in parameter_set_list]
        filename = base_dir + "experiment_1_live_sample_size_" + str(sample_size) +"_error_" + str(error)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            distance_map,
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
            cmap='viridis',
            mask=np.isnan(distance_map),
            cbar_kws={'label': 'Jensen-Shannon Divergence (ε=' +
                               str(round(error, 5)) + ', sample_size='
                                    + str(sample_size) + ')'})
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
        df = pd.DataFrame(distance_map, index=labels, columns=labels)
        df.to_csv(filename + ".csv")
    return on_completed


parameter_set_list = create_parameter_sets(0, 4, 2, 1, 2, 1)

measure_distance(
        parameter_set_list,
        straight_insertion_sort,
        gen_insertion_sort_environment,
        #min_sample_size_exp=16,
        min_sample_size_exp=20,
        max_sample_size_exp=30,
        on_completed = create_on_completed("data_experiment_1")
)