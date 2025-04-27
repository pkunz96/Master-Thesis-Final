from algorithms.heap_sort import heap_sort, gen_heap_sort_environment
from algorithms.quick_sort import quick_sort, gen_quick_sort_environment
from experiments import sampling
from experiments.sampling import ParameterSet

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment


alg = quick_sort
env = gen_quick_sort_environment
suffix = "quick_sort"

mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 0, 100, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 50, 100, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 100, 100, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 150, 100, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 0, 50, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 0, 150, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 50, 50, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 50, 150, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 100, 150, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 100, 50, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 150, 50, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 150, 150, suffix=suffix), alg, env)
mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 200, 200, suffix=suffix), alg, env)
mcs.init()
