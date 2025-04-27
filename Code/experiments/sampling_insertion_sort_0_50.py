from experiments import sampling
from experiments.sampling import ParameterSet

from algorithms.straight_insertion_sort import straight_insertion_sort, gen_insertion_sort_environment


alg = straight_insertion_sort
env = gen_insertion_sort_environment
suffix = "insertion_sort"

#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 0, 100, suffix=suffix), alg, env)
#mcs.init()
#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 50, 100, suffix=suffix), alg, env)
#mcs.init()
#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 100, 100, suffix=suffix), alg, env)
#mcs.init()


#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 150, 100, suffix=suffix), alg, env)
#mcs.init()
mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 0, 50, suffix=suffix), alg, env)
mcs.init()
#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 0, 150, suffix=suffix), alg, env)
#mcs.init()


#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 50, 50, suffix=suffix), alg, env)
#mcs.init()

#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 50, 150, suffix=suffix), alg, env)
#mcs.init()

#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 100, 150, suffix=suffix), alg, env)
#mcs.init()

#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 100, 50, suffix=suffix), alg, env)
#mcs.init()

#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 150, 50, suffix=suffix), alg, env)
#mcs.init()
#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 150, 150, suffix=suffix), alg, env)
#mcs.init()

#Running
#mcs: sampling.MonteCarloSampling = sampling.MonteCarloSampling(ParameterSet(20, 5, 200, 200, suffix=suffix), alg, env)
#mcs.init()
