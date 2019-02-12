# python3
# coding: utf-8

"""
CS7641 - Assignment 1 for performing cleaning and classification

Mike Tong


Created: JAN 2019
"""
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from dt_analysis import DecisionTreeC_Analysis
from xgb_analysis import XGBC_Analysis
from mlpc_analysis import MLPC_Analysis
from svm_analysis import SVMC_Analysis
from knn_analysis import KNNC_Analysis

from ancillary import prep_housing, prep_student

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='student')
    parser.add_argument('--student', default="./cleaned_student_data.csv")
    parser.add_argument('--housing', default="./cleaned_housing_data.csv")
    parser.add_argument('--sample_housing', default=30000,
        help="Sampling a portion of the housing data.")
    parser.add_argument('--save_data', default=False,
        help="True or False to save the data as CSVs")
    parser.add_argument('--classifier', default='all',
        help="Either 'DT', 'XGB', 'SVM', 'MLP', 'KNN' or 'all'.")
    parser.add_argument('--prep', default=False,
        help="Only use if data is not prepped after cleaned")
    args = parser.parse_args()

    datasets = []
    targets = []

    if args.dataset == 'all' or args.dataset == 'student':
        df_student = pd.read_csv(args.student)
        if args.prep:
            df_student = prep_student(df_student)

        df_student.index.name = 'Student Dataset'
        datasets.append(df_student)
        targets.append('gender')

    if args.dataset == 'all' or args.dataset == 'housing':
        df_housing = pd.read_csv(args.housing)
        if args.prep:
            df_housing = prep_housing(df_housing)

        df_housing.index.name = 'Housing Dataset'
        df_housing = df_housing.sample(args.sample_housing, random_state=7308)
        datasets.append(df_housing)
        targets.append('class')

    for dataset, target in zip(datasets, targets):
        if args.classifier=='all' or args.classifier=='DT':
            analysis = DecisionTreeC_Analysis(dataset=dataset, target=target, save=args.save_data)
            analysis.general_analysis()
            analysis.depth_analysis(range_=np.linspace(2, 28, 18, dtype=int))
            analysis.max_node_analysis(range_=np.linspace(2, 1000, 20, dtype=int))
            analysis.min_samples_analysis(range_=range(1,31))

        if args.classifier=='all' or args.classifier=='XGB':
            analysis = XGBC_Analysis(dataset=dataset, target=target, save=args.save_data)
            analysis.general_analysis()
            analysis.depth_analysis(range_=range(2,20))
            analysis.n_estimator_analysis(np.linspace(10, 2500, 20, dtype=int))
            analysis.lr_analysis()

        if args.classifier=='all' or args.classifier=='SVM':
            analysis = SVMC_Analysis(dataset=dataset, target=target, save=args.save_data)
            analysis.general_analysis()
            analysis.kernel_analysis()
            analysis.penalty_analysis()

        if args.classifier=='all' or args.classifier=='MLP':
            analysis = MLPC_Analysis(dataset=dataset, target=target, save=args.save_data)
            analysis.general_analysis()
            analysis.max_iteration_analysis()
            analysis.hidden_layer_analysis()
            analysis.activation_analysis()

        if args.classifier=='all' or args.classifier=='KNN':
            analysis = KNNC_Analysis(dataset=dataset, target=target, save=args.save_data)
            analysis.general_analysis()
            analysis.n_neighbors_analysis()


if __name__ == '__main__':
    main()
