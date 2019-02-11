# python3
# coding: utf-8

"""
CS7641 - Assignment 1 support functions

Mike Tong


Created: JAN 2019
"""

from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve


def prep_data_for_clf(df, target, test_size=0.2, random_state=7308):
    """Performs train test split more succinctly
    """
    X = df.drop(columns=(target), axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prep_housing(df, verbose=True):
    """Modifies the housing dataset for classification
    """
    df_c = df.copy()
    df_c = df_c[df_c['PRICE'] != 0]

    thirds = df_c.shape[0]/3 # locations to split data evenly into thirds
    df_first_split = df_c['PRICE'].sort_values().iloc[int(thirds)]
    df_second_split = df_c['PRICE'].sort_values().iloc[int(thirds * 2)]

    df_c['class'] = 'err'
    if verbose:
            print("Group 1 is lte to {}".format(
                int(df_first_split)))
            print("Group 2 is gt {} and lte to {}".format(
                int(df_first_split), int(df_second_split)))
            print("Group 3 is gt {}".format(
                int(df_second_split)))

            print("Adding `class` column via Group designations, this may take a while")

    for i,v in df_c.iterrows():
        if v['PRICE'] <= df_first_split:
            df_c.loc[i, 'class'] = 'Group 1'
        if v['PRICE'] > df_first_split and v['PRICE'] <= df_second_split:
            df_c.loc[i, 'class'] = 'Group 2'
        if v['PRICE'] > df_second_split:
            df_c.loc[i,'class'] = 'Group 3'

    return df_c.drop(columns=['PRICE'], axis=1)

def prep_student(df):
    df_c = df.copy()

    df_c['gender'] = 0
    i = df_c[df_c['female'] == 1].index
    df_c.loc[i, 'gender'] = 1

    return df_c.drop(columns=['male', 'female'], axis=1)

def measure_execution_time(clf, X, y, iterations=10):
    """Measures the wall time for train and test.
    """
    training_times = []
    testing_times = []

    for _ in range(iterations):
        st = time()
        clf.fit(X, y)
        et = time()
        training_times.append(et-st)

        st = time()
        clf.predict(X)
        et = time()
        testing_times.append(et-st)

    return training_times, testing_times

def plot_learning_curve(estimator, title, dataset, target, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(0.05, 0.95, 10), scale=False):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    X = dataset.drop(columns=[target], axis=1)
    y = dataset[target]

    if scale:
        sclr = StandardScaler()
        sclr.fit(X.astype(float))

        X = sclr.transform(X.astype(float))

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training size precentage")
    plt.ylabel("Score")
    train_sizes_, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xticks(train_sizes)
    plt.legend(loc=4)#"best")
    plt.show()
