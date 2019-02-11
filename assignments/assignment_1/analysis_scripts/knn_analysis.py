# python3
# coding: utf-8

"""
CS7641 - Assignment 1 K-Nearest Neighbors Analysis

Mike Tong


Created: JAN 2019
"""


from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from ancillary import measure_execution_time, prep_data_for_clf, plot_learning_curve

class KNNC_Analysis(object):
    def __init__(self, dataset, target, save=False, random_state=7308):
        self.data = dataset
        self.target = target
        self.save = save
        self.random = random_state

    def general_analysis(self):
        print("\n######")
        print("KNN Classifier:")
        print("Default Baseline values (5 neighbors)")

        clf = KNeighborsClassifier(n_jobs=-1)
        plot_learning_curve(clf, '{} KNN Learning Curve (uniform)'.format(
            self.data.index.name), self.data, self.target, cv=5, scale=True)

        clf = KNeighborsClassifier(weights='distance', n_jobs=-1)
        plot_learning_curve(clf, '{} KNN Learning Curve (distance)'.format(
            self.data.index.name), self.data, self.target, cv=5, scale=True)

        print("\n~~~~~~")
        print("Execution time metrics")
        X_train, X_test, y_train, y_test = prep_data_for_clf(self.data,
            self.target, random_state=self.random)

        sclr = StandardScaler()
        sclr.fit(X_train.astype('float'))
        X_train_std = sclr.transform(X_train.astype('float'))
        X_test_std = sclr.transform(X_test.astype('float'))

        training_time, testing_time = measure_execution_time(clf,
            pd.concat([pd.DataFrame(X_train_std), pd.DataFrame(X_test_std)]), pd.concat([y_train, y_test])
            )
        print("Training time input dim of {} : {:.4f} (+/- {:.4f})".format(
            X_train.shape, np.mean(training_time), np.std(training_time))
            )
        print("Testing time input dim of {}: {:.4f} (+/- {:.4f})".format(
            X_test.shape, np.mean(testing_time), np.std(testing_time))
            )

        for w in ['uniform', 'distance']:
            print("\n~~~~~~")
            print('{} weights:'.format(w.capitalize()))
            clf = KNeighborsClassifier(weights=w, n_jobs=-1)
            scores = cross_val_score(clf,
                pd.concat([pd.DataFrame(X_train_std), pd.DataFrame(X_test_std)]), pd.concat([y_train, y_test]),
                cv=10, n_jobs=-1)
            print("10 Fold Cross Validation Accuracy: {:.4f} (+/- {:.4f})".format(
                scores.mean(), scores.std() * 2))

            clf.fit(X_train_std, y_train)
            preds_train = clf.predict(X_train_std)
            preds_test = clf.predict(X_test_std)
            print("Training Accuracy:",
                accuracy_score(y_true=y_train, y_pred=preds_train))
            print("Training F1:",
                f1_score(y_true=y_train, y_pred=preds_train, average='weighted'))
            print("Testing Accuracy:",
                accuracy_score(y_true=y_test, y_pred=preds_test))
            print("Testing F1:",
                f1_score(y_true=y_test, y_pred=preds_test, average='weighted'))

        print("~~~~~~\n")

    def n_neighbors_analysis(self, range_=range(1,50)):
        print("\n######")
        print("Testing different neighbor values")
        metrics = defaultdict(list)
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target, random_state=self.random)

        sclr = StandardScaler()
        sclr.fit(X_train.astype('float'))
        X_train_std = sclr.transform(X_train.astype('float'))
        X_test_std = sclr.transform(X_test.astype('float'))

        for n in range_:
            clf = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
            clf.fit(X_train_std, y_train)
            metrics['train_acc_uniform'].append(
                clf.score(X_train_std, y_train)
                )
            metrics['test_acc_uniform'].append(
                clf.score(X_test_std, y_test)
                )

            clf = KNeighborsClassifier(n_neighbors=n, weights="distance", n_jobs=-1)
            clf.fit(X_train_std, y_train)
            metrics['train_acc_distance'].append(
                clf.score(X_train_std, y_train)
                )
            metrics['test_acc_distance'].append(
                clf.score(X_test_std, y_test)
                )

        for m in metrics.values():
            plt.plot(range_, m, 'o-')

        plt.legend([i for i in metrics], ncol=1, loc='best')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy scores (weighted)')
        plt.title('Accuracy scores of Train and Test for {}'.format(
            self.data.index.name))
        plt.show()

        optimal_weight = np.argmax(np.max(list(metrics.values()), axis=1))
        self.optimal_weight = [i.split('_')[-1] for i in metrics][optimal_weight]
        self.optimal_n = np.argmax(list(metrics.values())[optimal_weight])

        print("Better weight metric is:", self.optimal_weight)
        print("Updated Learning Curves:")
        clf = KNeighborsClassifier(n_neighbors=self.optimal_n, weights=self.optimal_weight, n_jobs=-1)
        plot_learning_curve(clf, '{} KNN Learning Curve (distance)'.format(
            self.data.index.name), self.data, self.target, cv=5)

        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv("./results/KNN/{}_KNN_neighbor_analysis.csv".format(
                self.data.index.name))

        return metrics
