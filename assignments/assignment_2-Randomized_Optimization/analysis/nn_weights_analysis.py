"""nn_weights_analysis.py

CS7641-HW2: Optimization
Performs ABAGAILs Random Hill Climb optimization
Code referenced from https://github.com/mitian223/CS7641/blob/master/HW2/ANN_rhc.py


Mike Tong

created: FEB 2019
"""

import os
import sys
sys.path.append('./ABAGAIL/ABAGAIL.jar')

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.MIMIC as MIMIC
from func.nn.activation import LogisticSigmoid, HyperbolicTangentSigmoid
from ancilary import errorOnDataSet, init_instances, train


# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 19
HIDDEN_LAYER1 = 12
HIDDEN_LAYER2 = 12
OUTPUT_LAYER = 1
bp_iters = 2000
rhc_iters = 15000
sa_iters = 10000
ga_iters = 10000

bp_iterations  = 10
rhc_iterations = 10

data_path = './assets/data/cleaned_student_data.csv'
bp_path = './assets/results/nn_weight/BP/BP_LOG.csv'
rhc_path = './assets/results/nn_weight/RHC/RHC_LOG.csv'
sa_path = './assets/results/nn_weight/SA/SA_LOG.csv'
ga_path = './assets/results/nn_weight/GA/GA_LOG.csv'


def Backpropogation(out_path, train_inst, test_inst, repeats, training_iterations):
    for i in range(repeats):
        out_path_ = out_path.replace("BP_", 'BP_{}'.format(str(i).zfill(3)))
        with open(out_path_, 'w') as f:
            f.write('{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','elapsed'))
        factory = BackPropagationNetworkFactory()
        measure = SumOfSquaresError()
        data_set = DataSet(train_inst)
        # acti = LogisticSigmoid()
        acti = HyperbolicTangentSigmoid()
        rule = RPROPUpdateRule()
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER],acti)
        train(
            BatchBackPropagationTrainer(data_set,classification_network,measure,rule),
            classification_network, 'Backprop', train_inst, test_inst, measure, training_iterations, out_path_
        )

def Random_hill_climb(out_path, train_inst, test_inst, repeats, training_iterations):
    """Run this experiment"""
    for i in range(repeats):
        out_path_ = out_path.replace("RHC_", 'RHC_{}'.format(str(i).zfill(3)))
        with open(out_path_, 'w') as f:
            f.write('{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','elapsed'))
        factory = BackPropagationNetworkFactory()
        measure = SumOfSquaresError()
        data_set = DataSet(train_inst)
        # acti = LogisticSigmoid()
        acti = HyperbolicTangentSigmoid()
        rule = RPROPUpdateRule()
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER],acti)
        nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
        oa = RandomizedHillClimbing(nnop)
        train(oa, classification_network, 'RHC', train_inst, test_inst, measure, training_iterations, out_path_)

def Simulated_annealing(out_path, train_inst, test_inst, T, CE, training_iterations):
    """Run this experiment"""
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_inst)
    # acti = LogisticSigmoid()
    acti = HyperbolicTangentSigmoid()
    rule = RPROPUpdateRule()

    oa_name = "SA_T{}_CE{}".format(int(T), str(CE).split('.')[-1])
    with open(out_path.replace('SA_', oa_name),'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','elapsed'))

    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER],acti)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = SimulatedAnnealing(T, CE, nnop)
    train(oa, classification_network, oa_name, train_inst, test_inst, measure, training_iterations, out_path.replace('SA_',oa_name))

def Genetic_algorithm(out_path, train_inst, test_inst, P, mate, mutate, training_iterations):
    """Run this experiment"""
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_inst)
    # acti = LogisticSigmoid()
    acti = HyperbolicTangentSigmoid()
    rule = RPROPUpdateRule()

    oa_name = "GA_P{}_mate{}_mut{}".format(P, mate, mutate)
    with open(out_path.replace('GA_', oa_name),'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','elapsed'))

    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], acti)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = StandardGeneticAlgorithm(P, mate, mutate, nnop)
    train(oa, classification_network, oa_name, train_inst, test_inst, measure, training_iterations, out_path.replace('GA_', oa_name))


if __name__ == "__main__":
    train_inst, test_inst = init_instances(data_path)
    Backpropogation(bp_path, train_inst, test_inst, bp_iterations, bp_iters)
    Random_hill_climb(rhc_path, train_inst, test_inst, rhc_iterations, rhc_iters)
    #
    for T in [1e1, 1e3, 1e5, 1e7, 1e9, 1e11, 1e13]:
        for CE in [0.20, 0.40, 0.60, 0.80, 0.90, 0.99]:
            Simulated_annealing(sa_path, train_inst, test_inst, T, CE, sa_iters)

    for p in [100, 200, 300]:
        for mate in [20, 30, 40]:
            for mutate in [20, 30, 40]:
                Genetic_algorithm(ga_path, train_inst, test_inst, p, mate, mutate, ga_iters)
