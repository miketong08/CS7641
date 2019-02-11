# python3
# coding: utf-8

"""
CS7641 - Assignment 1 Decision Tree Analysis

Mike Tong


Created: JAN 2019
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from ancillary import measure_execution_time, prep_data_for_clf, plot_learning_curve

class DecisionTreeC_Analysis(object):
    def __init__(self, dataset, target, save=False, random_state=7308):
        self.data = dataset
        self.target = target
        self.random_state = random_state
        self.save = save

    def general_analysis(self):
        print("\n######")
        print("Decision Tree Classifier:")
        print("Default Baseline values (no max depth or max leaf nodes)\n")

        clf = DecisionTreeClassifier(random_state=self.random_state)
        plot_learning_curve(clf, '{} Decision Tree Learning Curve'.format(self.data.index.name), self.data, self.target, cv=5)

        print("\n~~~~~~")
        print("Execution time metrics")
        X_train, X_test, y_train, y_test = prep_data_for_clf(self.data, self.target, random_state=self.random_state)

        training_time, testing_time = measure_execution_time(clf,
            self.data.drop(columns=[self.target], axis=1), self.data[self.target])
        print("Training time input dim of {} : {:.4f} (+/- {:.4f})".format(
            X_train.shape, np.mean(training_time), np.std(training_time))
            )
        print("Testing time input dim of {}: {:.4f} (+/- {:.4f})".format(
            X_test.shape, np.mean(testing_time), np.std(testing_time))
            )

        print("\n~~~~~~")
        print('Split on Gini Importance:')
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

        print('\n~~~~~~')
        print('Split on Entropy Gain:')
        clf = DecisionTreeClassifier(criterion='entropy', random_state=7308)
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
        print("~~~~~~\n")


    def depth_analysis(self, range_=range(2,20)):
        print("\n######")
        print("Testing different tree depths")
        metrics = defaultdict(list)  # keep track of test/train accuracies
        X_train, X_test, y_train, y_test = prep_data_for_clf(self.data,
            self.target, random_state=self.random_state)

        for d in range_:
            clf = DecisionTreeClassifier(max_depth=d, random_state=self.random_state)
            clf.fit(X_train, y_train)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_gini'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train)
                )
            metrics['test_gini'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test)
                )

            clf = DecisionTreeClassifier(criterion='entropy',
                max_depth=d,
                random_state=self.random_state)
            clf.fit(X_train, y_train)

            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_entropy'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train)
                )
            metrics['test_entropy'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test)
                )

        self.plot_metric(metrics, range_,
            xlabel='Max Depth',
            title='Accuracy vs Maximum Tree Depth for the {}'.format(
                self.data.index.name)
                )

        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv(
        	"./results/DT/{}_DT_depth_analysis.csv".format(
                self.data.index.name)
        	)

        return metrics

    def min_samples_analysis(self, range_=range(1,31)):
        print("\n######")
        print("Testing different leaf sample size thresholds")
        metrics = defaultdict(list)  # keep track of test/train accuracies
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data,
            self.target,
            random_state=self.random_state)

        for m in range_:
            clf = DecisionTreeClassifier(min_samples_leaf=m, random_state=self.random_state)
            clf.fit(X_train, y_train)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_gini'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train)
                )
            metrics['test_gini'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test)
                )

            clf = DecisionTreeClassifier(criterion='entropy',
                min_samples_leaf=m,
                random_state=self.random_state)
            clf.fit(X_train, y_train)
            preds_train = clf.predict(X_train)
            preds_test = clf.predict(X_test)

            metrics['train_entropy'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train)
                )
            metrics['test_entropy'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test)
                )

        self.plot_metric(metrics, range_,
            xlabel='Min Samples per Leaf',
            title='Accuracy vs Minimum Samples per Leaf for the {}'.format(
                self.data.index.name)
                )

        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv(
        	"./results/DT/{}_DT_min_sample_analysis.csv".format(
                self.data.index.name)
        	)

        return metrics

    def max_node_analysis(self, range_=range(2, 200, 10)):
        print("\n######")
        print("Testing different maximum leaf nodes")
        metrics = defaultdict(list)
        X_train, X_test, y_train, y_test = prep_data_for_clf(
            self.data,
            self.target,
            random_state=self.random_state)

        for m in range_:
        	clf = DecisionTreeClassifier(
                max_leaf_nodes=m,
                random_state=self.random_state)
        	clf.fit(X_train, y_train)
        	preds_train = clf.predict(X_train)
        	preds_test = clf.predict(X_test)

        	metrics['train_gini'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train)
                )
        	metrics['test_gini'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test)
                )

        	clf = DecisionTreeClassifier(criterion='entropy',
                max_leaf_nodes=m,
                random_state=self.random_state)
        	clf.fit(X_train, y_train)
        	preds_train = clf.predict(X_train)
        	preds_test = clf.predict(X_test)

        	metrics['train_entropy'].append(
                accuracy_score(y_true=y_train, y_pred=preds_train)
                )
        	metrics['test_entropy'].append(
                accuracy_score(y_true=y_test, y_pred=preds_test)
                )

        self.plot_metric(metrics, range_,
            xlabel='Max Leaf Node',
            title='Accuracy vs Maximum Leaf Nodes for the {}'.format(
                self.data.index.name)
                )
        if self.save:
            pd.DataFrame(metrics, index=range_).to_csv(
        	"./results/DT/{}_DT_max_depth_analysis.csv".format(
                self.data.index.name)
        	)

        return metrics

    def plot_metric(self, metrics, range_, xlabel, title):
    	plt.gcf().set_size_inches(8, 5)
    	n_items = len(metrics[list(metrics.keys())[0]])

    	#  for col in accs:
    	for col in metrics:
    	    plt.plot(range_, metrics[col], 'o-')

    	plt.legend([i for i in metrics], ncol=1, loc='best')
    	plt.xlabel(xlabel)
    	plt.ylabel('Accuracy Score (Weighted)')
    	plt.title(title)

    	plt.xticks(range_, rotation=45)
    	plt.grid()
    	plt.show()
