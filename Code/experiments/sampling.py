import os
import queue
import random
import threading
from datetime import datetime
from math import ceil
from typing import Callable, List, Tuple, Dict, Optional
from numpy.typing import NDArray
import numpy as np
import time

from algorithms.environment_oop import Environment
from algorithms.straight_insertion_sort import gen_insertion_sort_environment, straight_insertion_sort

# Monte-Carlo sampling as outlined in chapter 4.2.3


class ParameterSet:

    def __init__(self, problem_mu: int, problem_sigma: int, problem_size_mu: int, problem_size_sigma: int, suffix: str = ""):
        self.problem_mu = problem_mu
        self.problem_sigma = problem_sigma
        self.problem_size_mu = problem_size_mu
        self.problem_size_sigma = problem_size_sigma
        self.suffix = suffix

    def as_str(self) -> str:
        return str(self.problem_mu) + "_" + str(self.problem_sigma) + "_" + str(self.problem_size_mu) + "_" + str(self.problem_size_sigma) + "_" + self.suffix


class MonteCarloSampling:

    @staticmethod
    def _generate_state_successor_dict(raw_training_record: NDArray[int], computation_state_successor_dict: Dict[int, List[int]]) -> Dict[int, List[int]]:
        for row_index in range(0, len(raw_training_record) - 1):
            cur_state = raw_training_record[row_index][-1]
            suc_state = raw_training_record[row_index + 1][-1]
            if cur_state not in computation_state_successor_dict:
                computation_state_successor_dict[cur_state] = []
            successor_list = computation_state_successor_dict[cur_state]
            if suc_state not in successor_list:
                successor_list.append(suc_state)
        return computation_state_successor_dict

    @staticmethod
    def _extract_sample(sample: NDArray[int], computation_state_successor_dict: Dict[int, List[int]]) -> Tuple[Optional[NDArray[int]], Optional[NDArray[int]]]:
        predictor_list = []
        response_list = []
        for row_index in range(0, sample.shape[0] - 1):
            state = sample[row_index][-1]
            if state in computation_state_successor_dict and len(computation_state_successor_dict[state]) > 1:
                predictor_list.append(sample[row_index])
                response_list.append(sample[row_index+1][-1])
        max_index = len(predictor_list) - 1
        if max_index < 0:
            return None, None
        index = random.randint(0, max_index)
        return np.array(predictor_list[index], dtype=np.int16), np.array(response_list[index], dtype=np.int16)

    @staticmethod
    def ensure_experiments_dir():
        current_dir = os.getcwd()
        dir_name = os.path.basename(current_dir)
        if dir_name == 'experiments':
            print(f"Already in 'experiments' directory: {current_dir}")
            return
        # Search upwards for an 'experiments' directory
        path = current_dir
        while path != os.path.dirname(path):  # Until reaching root
            potential_path = os.path.join(path, 'experiments')
            if os.path.isdir(potential_path):
                os.chdir(potential_path)
                print(f"Changed working directory to: {potential_path}")
                return
            path = os.path.dirname(path)
        raise FileNotFoundError("'experiments' directory not found in current or parent directories.")

    @staticmethod
    def _load_data(parameter_set: ParameterSet) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        MonteCarloSampling.ensure_experiments_dir()
        predictor_file_name = "./data_sampling/" + parameter_set.as_str() + "_pred.npy"
        response_file_name = "./data_sampling/" + parameter_set.as_str() + "_response.npy"
        if os.path.exists(predictor_file_name) and os.path.exists(response_file_name):
            return np.load(predictor_file_name), np.load(response_file_name)
        return None, None

    def __init__(self, parameter_set: ParameterSet, sort: Callable[[Environment], List[int]], create_environment: Callable[[List[int]], Environment]):
        self.parameter_set: ParameterSet = parameter_set
        self.sort: Callable[[Environment], List[int]] = sort
        self.create_environment: Callable[[List[int]], Environment] = create_environment
        self.problem_sample_dict: Dict[Tuple[int], NDArray] = dict()
        self.computation_state_successor_dict = dict()
        self.predictor_arr: Optional[NDArray[int]] = None
        self.response_arr: Optional[NDArray[int]] = None

    def _generate_problem(self) -> List[int]:
        problem_size: int = np.random.normal(self.parameter_set.problem_size_mu, self.parameter_set.problem_size_sigma, 1)
        return [ceil(value) for value in
                np.random.normal(self.parameter_set.problem_mu, self.parameter_set.problem_sigma, max([ceil(problem_size), 2]))]

    def _sample_execution(self) -> NDArray[int]:
        env: Environment = self.create_environment(self._generate_problem())
        self.sort(env)
        return np.array(env.pass_record, dtype=np.int16)

    def _init_state_suc_dict(self):
        return self.computation_state_successor_dict

    def init(self, caching: bool = True, sampling_size: int = 10**6):
        directory = "./data_sampling/"
        os.makedirs(directory, exist_ok=True)
        predictor_arr, response_arr = None, None
        if caching is True:
            predictor_arr, response_arr = MonteCarloSampling._load_data(self.parameter_set)
        if predictor_arr is None:
            for i in range(0, 30):
                MonteCarloSampling._generate_state_successor_dict(self._sample_execution(), self.computation_state_successor_dict)
            computation_state_successor_dict = self._init_state_suc_dict()
            sampled_count = 0
            start_time = time.time()
            while predictor_arr is None or sampled_count < sampling_size:
                thread_count = min([5, sampling_size - sampled_count])
                result_queue = queue.Queue()
                thread_list = []
                for i in range(0, thread_count):
                    thread = threading.Thread(target=lambda q: q.put(self._sample_execution()), args=(result_queue,))
                    thread_list.append(thread)
                for thread in thread_list:
                    thread.start()
                for thread in thread_list:
                    thread.join()
                while not result_queue.empty():
                    predictors, response = MonteCarloSampling._extract_sample(result_queue.get(), computation_state_successor_dict)
                    if predictors is None or response is None:
                        continue
                    elif predictor_arr is None:
                        predictor_arr = np.array([predictors])
                        response_arr = np.array([[response]])
                    else:
                        predictor_arr = np.vstack([predictor_arr, predictors])
                        response_arr = np.vstack([response_arr, response])
                    sampled_count += len(predictors)
                cur_time = time.time()
                samples_per_time = sampled_count / (cur_time - start_time)
                #print("------ Round ------")
                #print(self.parameter_set.as_str() + ": " + str(round(sampled_count / sampling_size, 8)*100) + " % (" + str(sampled_count) + "/" + str(sampling_size) + ")" )
                #print("Samples per time: " + str(samples_per_time) + "/s")
                #print("Time remaining: " + str(((sampling_size- sampled_count) / samples_per_time) / (24 * 3600)) + " days")
                #print("\n\n")
            np.save(directory + self.parameter_set.as_str() + "_pred.npy", predictor_arr)
            np.save(directory + self.parameter_set.as_str() + "_response.npy", response_arr)
        self.predictor_arr = predictor_arr
        self.response_arr = response_arr

    def sample(self, sample_size: int) -> Tuple[NDArray, NDArray]:
        rows = [random.randint(0, len(self.predictor_arr) - 1) for _ in range(sample_size)]
        # Resampling
        predictors, responses = self.predictor_arr[rows], self.response_arr[rows]
        return predictors, responses
