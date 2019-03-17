"""optimization_TS.py

CS7641-HW2: Optimization
Performs ABAGAILs travelling sales person problem, https://github.com/pushkar/ABAGAIL


Mike Tong

created: FEB 2019
"""

import sys
import os
import time
from time import clock
from itertools import product
from array import array

sys.path.append('./ABAGAIL/ABAGAIL.jar')
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays


# Random number generator */
random = Random()

N = 100
maxIters = 10000
numTrials= 1


points = [[0 for x in range(2)] for x in range(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

outfile = './assets/results/optimization/TP_@ALG@_@N@_LOG.csv'
early_stop_patience = 100

fill = [N] * N
ranges = array('i', fill)
ef = TravelingSalesmanSortEvaluationFunction(points);
odd = DiscreteUniformDistribution(ranges);
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# MIMIC
for t in range(numTrials):
    for samples in [200, 300, 500, 1000]:
        for keep in [10, 20, 40, 70, 100]:
            for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
                fname = outfile.replace('@ALG@','MIMIC{}_{}_{}'.format(samples,keep,m)).replace('@N@',str(t+1))
                df = DiscreteDependencyTree(m, ranges);
                with open(fname,'w') as f:
                    f.write('iterations,fitness,time,fevals\n')
                ef = TravelingSalesmanSortEvaluationFunction(points);
                pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
                mimic = MIMIC(samples, keep, pop)
                fit = FixedIterationTrainer(mimic, 10)
                times =[0]
                early_stop = []
                for i in range(0,maxIters,10):
                    start = clock()
                    fit.train()
                    elapsed = time.clock()-start
                    times.append(times[-1]+elapsed)
                    score = ef.value(mimic.getOptimal())
                    st = '{},{},{}\n'.format(i,score,times[-1])
                    print st
                    with open(fname,'a') as f:
                        f.write(st)

                    early_stop.append(score)
                    if len(early_stop) == early_stop_patience:
                        if round(score,3) == round(sum(early_stop[:-1])/(len(early_stop)-1), 3):
                            print("Early stopping at iteration {}".format(i))
                            break
                        else:
                            early_stop.pop(0)

# RHC
for t in range(numTrials):
    fname = outfile.replace('@ALG@','RHC').replace('@N@',str(t+1))
    with open(fname,'w') as f:
        f.write('iterations,fitness,time,fevals\n')
    ef = TravelingSalesmanRouteEvaluationFunction(points)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times =[0]
    for i in range(0,maxIters,10):
        start = clock()
        fit.train()
        elapsed = time.clock()-start
        times.append(times[-1]+elapsed)
        score = ef.value(rhc.getOptimal())
        st = '{},{},{}\n'.format(i,score,times[-1])
        print st
        with open(fname,'a') as f:
            f.write(st)

        # early_stop.append(score)
        # if len(early_stop) == early_stop_patience:
        #     if round(score, 3) == round(sum(early_stop[:-1])/(len(early_stop)-1), 3):
        #         print("Early stopping at iteration {}".format(i))
        #         break
        #     else:
        #         early_stop.pop(0)

# SA
for t in range(numTrials):
    for T in [1e1, 1e3, 1e5, 1e7, 1e9, 1e11, 1e13]:
        for CE in [0.20, 0.40, 0.60, 0.80, 0.90, 0.99]:
            fname = outfile.replace('@ALG@','SA{}'.format(CE)).replace('@N@',str(t+1))
            with open(fname,'w') as f:
                f.write('iterations,fitness,time,fevals\n')
            ef = TravelingSalesmanRouteEvaluationFunction(points)
            hcp = GenericHillClimbingProblem(ef, odd, nf)
            sa = SimulatedAnnealing(1E10, CE, hcp)
            fit = FixedIterationTrainer(sa, 10)
            times =[0]
            for i in range(0,maxIters,10):
                start = clock()
                fit.train()
                elapsed = time.clock()-start
                times.append(times[-1]+elapsed)
                score = ef.value(sa.getOptimal())
                st = '{},{},{}\n'.format(i,score,times[-1])
                print st
                with open(fname,'a') as f:
                    f.write(st)

                # early_stop.append(score)
                # if len(early_stop) == early_stop_patience:
                #     if round(score, 3) == round(sum(early_stop[:-1])/(len(early_stop)-1), 3):
                #         print("Early stopping at iteration {}".format(i))
                #         break
                #     else:
                #         early_stop.pop(0)

# GA
for t in range(numTrials):
    for pop in [100, 200, 300]:  # population
        for mate in [20, 30, 40]:  # population to mate
            for mutate in [20, 30, 40]:  # population to mutate
                fname = outfile.replace('@ALG@','GA{}_{}_{}'.format(pop,mate,mutate)).replace('@N@',str(t+1))
                with open(fname,'w') as f:
                    f.write('iterations,fitness,time,fevals\n')
                ef = TravelingSalesmanRouteEvaluationFunction(points)
                gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
                ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
                fit = FixedIterationTrainer(ga, 10)
                times =[0]
                early_stop = []
                for i in range(0,maxIters,10):
                    start = clock()
                    fit.train()
                    elapsed = time.clock()-start
                    times.append(times[-1]+elapsed)
                    score = ef.value(ga.getOptimal())
                    st = '{},{},{}\n'.format(i,score,times[-1])
                    print st
                    with open(fname,'a') as f:
                        f.write(st)

                    early_stop.append(score)
                    if len(early_stop) == early_stop_patience:
                        if round(score, 3) == round(sum(early_stop[:-1])/(len(early_stop)-1), 3):
                            print("Early stopping at iteration {}".format(i))
                            break
                        else:
                            early_stop.pop(0)
