# python3
# coding: utf-8

"""
CS7641 - Assignment 1 Support Vector Machine Analysis

Mike Tong


Created: JAN 2019
"""


from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from ancillary import measure_execution_time, prep_data_for_clf, plot_learning_curve


class SVMC_Analysis(object):
    def __init__(self, dataset, target, save=False, random_state=7308):
        self.data = dataset
        self.target = target
        self.save = save
        self.random = random_state

    def general_analysis(self):
        print("\n######")
        print("Support Vector Machine Classifier:")
        print("Default baseline values")

        clf = SVC(gamma='scale')
        plot_learning_curve(clf, '{} SVM Learning Curve'.format(
            self.data.index.name), self.data, self.target, cv=5, scale=True)

        print("\n~~~~~~")
        print("Execution time metrics")
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target, random_state=self.random)

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

        print("\n~~~~~~")
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
        print('~~~~~~\n')

    def kernel_analysis(self):
        print('\n######')
        print("Testing Different Kernel Functions with 10 Fold X-Val")
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target, random_state=self.random)

        sclr = StandardScaler()
        sclr.fit(X_train.astype('float'))
        X_train_std = sclr.transform(X_train.astype('float'))
        X_test_std = sclr.transform(X_test.astype('float'))

        accuracy = []
        stdev = []
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']

        for k in kernels:
            clf = SVC(kernel=k, gamma='scale')
            scores = cross_val_score(
                clf, pd.concat([pd.DataFrame(X_train_std), pd.DataFrame(X_test_std)]), pd.concat([y_train, y_test]),
                    cv=10, n_jobs=-1)

            accuracy.append(scores.mean())
            stdev.append(scores.std() * 2)

        results = pd.DataFrame(index=kernels,
            data=np.array([accuracy, stdev]).T,
            columns=['acc', 'std'])

        if self.save:
            results.to_csv("./results/SVM/{}_SVM_kernel_analysis.csv".format(
                self.data.index.name))

        return results

    def penalty_analysis(self, range_=[-7, -5, -3, -2, -1, 2, 3, 5, 7, 9]):
        print('\n######')
        print("Testing Different Penalty Values with 10 Fold X-Val")
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data, self.target, random_state=self.random)

        sclr = StandardScaler()
        sclr.fit(X_train.astype('float'))
        X_train_std = sclr.transform(X_train.astype('float'))
        X_test_std = sclr.transform(X_test.astype('float'))

        accuracy = []
        stdev = []
        for k in range_:

            clf = SVC(C=2**k, kernel='linear', gamma='scale')
            scores = cross_val_score(clf,
                pd.concat([pd.DataFrame(X_train_std), pd.DataFrame(X_test_std)]), pd.concat([y_train, y_test]),
                cv=10, n_jobs=-1)

            accuracy.append(scores.mean())
            stdev.append(scores.std() * 2)

        results = pd.DataFrame(index=[2**i for i in range_],
            data=np.array([accuracy, stdev]).T, columns=['acc', 'std'])
        if self.save:
            results.to_csv("./results/SVM/{}_SVM_penalty_analysis.csv".format(
                self.data.index.name)
                )

        return results
