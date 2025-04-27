import queue
import random
import threading
from math import ceil
from typing import Callable, List, Tuple, Dict, Optional
from numpy.typing import NDArray
import numpy as np
import time

from algorithms.environment_oop import Environment
from algorithms.straight_insertion_sort import gen_insertion_sort_environment, straight_insertion_sort

# Monte-Carlo sampling as outlined in chapter 4.2.3

class ParameterSet:

    def __init__(self, problem_mu: int, problem_sigma: int, problem_size_mu: int, problem_size_sigma: int):
        self.problem_mu = problem_mu
        self.problem_sigma = problem_sigma
        self.problem_size_mu = problem_size_mu
        self.problem_size_sigma = problem_size_sigma


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
                predictor_list.append(sample[row_index][:-1])
                response_list.append(sample[row_index+1][-1])
        max_index = len(predictor_list) - 1
        if max_index < 0:
            return None, None
        index = random.randint(0, max_index)
        return np.array(predictor_list[index], dtype=np.int16), np.array(response_list[index], dtype=np.int16)

    def __init__(self, parameter_set: ParameterSet, sort: Callable[[Environment], List[int]], create_environment: Callable[[List[int]], Environment]):
        self.parameter_set: ParameterSet = parameter_set
        self.sort: Callable[[Environment], List[int]] = sort
        self.create_environment: Callable[[List[int]], Environment] = create_environment
        self.problem_sample_dict: Dict[Tuple[int], NDArray] = dict()
        self.computation_state_successor_dict = dict()
        for i in range(0, 1000):
            MonteCarloSampling._generate_state_successor_dict(self._sample_execution(),  self.computation_state_successor_dict)

    def _generate_problem(self) -> List[int]:
        problem_size: int = np.random.normal(self.parameter_set.problem_size_mu, self.parameter_set.problem_size_sigma, 1)
        return [ceil(value) for value in
                np.random.normal(self.parameter_set.problem_mu, self.parameter_set.problem_sigma, max([ceil(problem_size), 2]))]

    """
    def _sample_execution(self) -> NDArray[int]:
        problem: List[int] = self._generate_problem()
        problem_tuple = tuple(problem)
        if problem_tuple in self.problem_sample_dict:
            return self.problem_sample_dict[problem_tuple]
        else:
            env: Environment = self.create_environment(self._generate_problem())
            self.sort(env)
            run = np.array(env.pass_record,  dtype=np.int16)
            self.problem_sample_dict[problem_tuple] = run
            return run
"""
    def _sample_execution(self) -> NDArray[int]:
        env: Environment = self.create_environment(self._generate_problem())
        self.sort(env)
        return np.array(env.pass_record, dtype=np.int16)

    def _init_state_suc_dict(self):
        #computation_state_successor_dict = dict()
        #for i in range(0, 1000):
            #computation_state_successor_dict = MonteCarloSampling._generate_state_successor_dict(self._sample_execution())
        return self.computation_state_successor_dict

    def sample(self, sample_size: int) -> Tuple[NDArray, NDArray]:
        start_timestamp: int = int(time.time())
        print("Sampling started at " + str(start_timestamp) + ". Sample size: " + str(sample_size) + ".")

        predictor_arr: NDArray[int] = np.empty((0,9), dtype=np.int16)
        # Original Version: response_arr: NDArray[int] = np.empty((1,), dtype=np.int16)
        response_arr: NDArray[int] = np.empty((), dtype=np.int16)
        computation_state_successor_dict = self._init_state_suc_dict()
        while predictor_arr.shape[0] < sample_size:
            thread_count = min([400, sample_size - predictor_arr.shape[0]])
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
                predictors, responses = MonteCarloSampling._extract_sample(result_queue.get(), computation_state_successor_dict)
                if predictors is None or responses is None:
                    continue
                predictor_arr = np.vstack([predictor_arr, predictors])
                response_arr = np.vstack([response_arr, responses])
        end_timestamp: int = int(time.time())
        print("Sampling ended at " + str(end_timestamp) + ". Duration: " + str(end_timestamp - start_timestamp))

    #        cur_timestamp: int = int(time.time())
        # cur_progress: float = predictor_arr.shape[0]*100 / sample_size;

        # projected_time_demand_min = round((((100 / cur_progress) - 1) * (cur_timestamp - start_timestamp)) / 60, 2)
        # projected_time_demand_h = round(projected_time_demand_min / 60, 2)
        # print("Progress: " + str(cur_progress) + " %. Remaining: " + str(projected_time_demand_min) + " min or " + str(projected_time_demand_h) + " h")
        return predictor_arr, response_arr

#sampling = MonteCarloSampling(ParameterSet(10,5,5,5), straight_insertion_sort, gen_insertion_sort_environment)

#while True:
#    sampling.sample(1024)

