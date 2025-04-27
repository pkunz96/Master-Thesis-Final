from typing import List, Dict, Tuple

import psutil
import os
import time
import threading

import numpy as np
from numpy.typing import NDArray

from scipy.spatial.distance import jensenshannon

from algorithms.straight_insertion_sort import gen_insertion_sort_environment, straight_insertion_sort
from experiments.old.sampling_old import ParameterSet, MonteCarloSampling

"""
This file implements the experiment described in section 4.2.3
"""

#  ParameterSet(problem_mu: int, problem_sigma: int, problem_size_mu: int, problem_size_sigma: int)
param_set_list: List[ParameterSet] = [ParameterSet(0, 2, 2, 0), ParameterSet(0, 2, 2, 0)]

sample_size: int = 10**int(input("Enter sample size exponent: "))
print(sample_size)

def map_sample_to_density(predictor_arr: NDArray[int], response_arr: NDArray[int]) -> Tuple[Dict[Tuple[int], float], Dict[Tuple[int], float]]:
    joint_density: Dict[Tuple[int], float] = dict()
    predictor_density: Dict[Tuple[int], float] = dict()

    joint_arr: NDArray[int] = np.hstack((predictor_arr, response_arr[1:]))

    unique_row, row_count = np.unique(joint_arr, axis=0, return_counts=True)
    for row, count in zip(unique_row, row_count):
        joint_density[tuple(row)] = float(count) / joint_arr.shape[0]

    unique_row, row_count = np.unique(predictor_arr, axis=0, return_counts=True)
    for row, count in zip(unique_row, row_count):
        predictor_density[tuple(row)] = float(count) / predictor_arr.shape[0]

    return joint_density, predictor_density


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


joint_density_distance_map: Dict[Tuple[ParameterSet, ParameterSet], float] = dict()
predictor_density_distance_map: Dict[Tuple[ParameterSet, ParameterSet], float] = dict()

def run():
    for outer_index in range(0, len(param_set_list)):
        for inner_index in range(outer_index + 1, len(param_set_list)):
            outer_param_set = param_set_list[outer_index]
            inner_param_set = param_set_list[inner_index]
            outer_predictor_arr, outer_response_arr = MonteCarloSampling(outer_param_set, straight_insertion_sort, gen_insertion_sort_environment).sample(2*sample_size)
            inner_predictor_arr, inner_response_arr = MonteCarloSampling(inner_param_set, straight_insertion_sort, gen_insertion_sort_environment).sample(2*sample_size)
            outer_joint_density, outer_predictor_density = map_sample_to_density(outer_predictor_arr, outer_response_arr)
            inner_joint_density, inner_predictor_density = map_sample_to_density(inner_predictor_arr, inner_response_arr)
            predictor_density_distance_map[(outer_param_set, inner_param_set)] = calc_jensen_shannon_divergence(outer_predictor_density, inner_predictor_density)
            joint_density_distance_map[(outer_param_set, inner_param_set)] = calc_jensen_shannon_divergence(outer_joint_density, inner_joint_density)


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
run();

print(joint_density_distance_map)
print(predictor_density_distance_map)