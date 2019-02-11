# python3
# coding: utf-8

"""
CS7641 - Assignment 1 eXtreme Gradient Boost Analysis

Mike Tong


Created: JAN 2019
"""


from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from ancillary import measure_execution_time, prep_data_for_clf, plot_learning_curve


class XGBC_Analysis(object):
    def __init__(self, dataset, target, save=False, random_state=7308):
        self.data = dataset
        self.target = target
        self.save = save
        self.random = random_state

    def general_analysis(self):
        print("\n######")
        print("eXtreme Gradient Boosted Decision Tree Classifier:")
        print("Default baseline values")

        clf = XGBClassifier(n_jobs=-1)
        plot_learning_curve(clf, '{} XGB Learning Curve'.format(
            self.data.index.name), self.data, self.target, cv=5)

        print("~~~~~~")
        print("Execution time metrics")
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target, random_state=self.random)

        training_time, testing_time = measure_execution_time(clf,
            self.data.drop(columns=[self.target], axis=1),
            self.data[self.target], iterations=5)
        print("Training time input dim of {} : {:.4f} (+/- {:.4f})".format(
            X_train.shape, np.mean(training_time), np.std(training_time))
            )
        print("Testing time input dim of {}: {:.4f} (+/- {:.4f})".format(
            X_test.shape, np.mean(testing_time), np.std(testing_time))
            )

        print("\n~~~~~~")
        scores = cross_val_score(clf,
            pd.concat([X_train, X_test]),
            pd.concat([y_train, y_test]),
            cv=10, n_jobs=-1)
        print("10 Fold Cross Validation Accuracy: {:.4f} (+/- {:.4f})".format(
            scores.mean(), scores.std() * 2))

        clf.fit(X_train, y_train)
        preds_train = clf.predict(X_train)
        preds_test = clf.predict(X_test)

        print("Training Accuracy:",
            accuracy_score(y_true=y_train, y_pred=preds_train))
        print("Training F1:",
            f1_score(y_true=y_train, y_pred=preds_train, average='weighted'))
        print("Testing Accuracy:",
		      accuracy_score(y_true=y_test, y_pred=preds_test))
        print("Testing F1:",
            f1_score(y_true=y_test, y_pred=preds_test, average='weighted'))
        print('~~~~~~\n')

    def depth_analysis(self, range_=range(2,20)):
        print("\n######")
        print("Testing different max tree depths.")
        metrics = defaultdict(list)
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target,
            random_state=self.random)

        for d in range_:
            clf = XGBClassifier(max_depth=d, random_state=self.random, n_jobs=-1)
            clf.fit(X_train, y_train)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_acc'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train))
            metrics['test_acc'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test))

        self.plot_metric(metrics, range_, xlabel='Max Depth')

        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv(
            "./results/XGB/{}_XGB_depth_analysis.csv".format(
                self.data.index.name)
            )

        return metrics


    def n_estimator_analysis(self, range_=np.linspace(10, 1000, 10).astype(int)):
        print("\n######")
        print("Testing different amounts of trees.")
        metrics = defaultdict(list)
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target,
            random_state=self.random)

        for n in range_:
            clf = XGBClassifier(n_estimators=int(n),
                random_state=self.random, n_jobs=-1)

            clf.fit(X_train, y_train)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_acc'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train))
            metrics['test_acc'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test))

        self.plot_metric(metrics, range_, xlabel='Number of Trees')

        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv(
            "./results/XGB/{}_XGB_n_estimator.csv".format(
                self.data.index.name)
            )

        return metrics


    def lr_analysis(self, range_=np.linspace(0.01, 1.0, 23)):
        print("\n######")
        print("Testing different learning rates.")
        metrics = defaultdict(list)
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target,
            random_state=self.random)

        for lr in range_:
            clf = XGBClassifier(learning_rate=lr,
                random_state=self.random,
                n_jobs=-1)

            clf.fit(X_train, y_train)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_acc'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train))
            metrics['test_acc'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test))

        self.plot_metric(metrics, range_, xlabel='Learning Rate')

        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv(
            "./results/XGB/{}_XGB_lr_analysis.csv".format(
                self.data.index.name)
            )

        return metrics


    def plot_metric(self, metrics, range_, xlabel):
        plt.gcf().set_size_inches(8, 5)
        n_items = len(metrics[list(metrics.keys())[0]])

        #  for col in accs:
        for col in metrics:
            plt.plot(range_, metrics[col], 'o-')

        plt.legend(list(metrics.keys()), ncol=1, loc=4)
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy Score (Weighted)')
        plt.title('Accuracy of Train and Test for {}'.format(
            self.data.index.name))

        plt.xticks(range_, rotation=45)
        plt.grid()
        plt.show()
