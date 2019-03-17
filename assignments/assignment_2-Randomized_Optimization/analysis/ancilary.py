"""ancillary.py

CS7641-HW2: Optimization
Support functions for neural net weight analysis
Code referenced from ABAGAIL, https://github.com/pushkar/ABAGAIL


Mike Tong

created: FEB 2019
"""

import csv
import sys
sys.path.append('./ABAGAIL/ABAGAIL.jar')
import time
from random import sample as Sample
from shared import Instance


def init_instances(data, test_size=0.25):
    """Read in a CSV and returns a list of ABAGAIL Instances
    :param data: Path to DataSet

    :return: list of Instance objects"""
    instances = []

    # Read in the CSV file
    with open(data, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            if row[0].isdigit():
                instance = Instance([float(value) for value in row[:-1]])
                instance.setLabel(Instance(float(row[-1])))
                instances.append(instance)

    data_inds = range(len(instances))
    n_test = int(len(instances) * test_size)

    i_test = Sample(data_inds, n_test)
    i_train = [i for i in data_inds if i not in i_test]

    train = [instances[i] for i in i_train]
    test = [instances[i] for i in i_test]

    return train, test

def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc


def train(oa, network, oaName, training_ints, testing_ints, measure, training_iterations, out_path):
    """Train a given network on a set of instances.
    """
    print("\nError results for %s\n---------------------------".format(oaName))
    times = [0]
    for iteration in xrange(training_iterations):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
    	times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
    	    MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_tst,acc_trg,acc_tst,times[-1])
            print txt
            with open(out_path,'a+') as f:
                f.write(txt)
