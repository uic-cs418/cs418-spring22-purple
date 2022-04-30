#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from IPython.display import display, Latex, Markdown
import matplotlib.pyplot as plt

def cleanCases(gun_cases):
    # gun violence cases dataset cleaned
    cases = pd.read_csv(gun_cases)
    cases = cases.loc[:, ['date', 'state', 'n_killed', 'n_injured']]

    cases['date'] = pd.to_datetime(cases['date'])
    cases['year'] = cases['date'].dt.year
    cases['harmed'] = cases['n_killed'] + cases['n_injured']
    cases = cases.drop(columns=['date', 'n_killed', 'n_injured'])
    cases = cases[(cases['year'] > 2013) & (cases['year'] < 2018)]
    cases = cases.groupby(['state', 'year']).agg('sum')
    cases = cases.reset_index()
    return cases

def cleanLaws(gun_laws):
    # gun laws dataset cleaned
    laws = pd.read_csv(gun_laws)
    laws = laws[(laws['year'] > 2014) & (laws['year'] < 2018)]
    return laws

def cleanPopulation(population):
    # state population dataset cleaned
    #importing state population counts from 2010 - 2019
    state_populations = pd.read_csv(population)

    #cleaning state population data
    state_populations = state_populations.drop(columns=['2010', '2011', '2012', '2013', '2018', '2019'])
    return state_populations

def combineDatasets(gun_cases, gun_laws, population):
    cases = cleanCases(gun_cases)
    laws = cleanLaws(gun_laws)
    state_population = cleanPopulation(population)
    
    # adding population counts to cases dataset by matching year and state
    for i, row in cases.iterrows():
        cases.at[i, ('population')] = float(state_population[state_population.state == row['state']][str(row['year'])])

    #compute the proportion of harmed individuals due to a gun incident out of the state population 
    cases['proportion_harmed'] = cases['harmed']/cases['population']
    cases.drop(columns=['harmed', 'population'], inplace=True)
    # Check if gun cases were reduced based on previous year values (only valid for 2015-2017)
    harmed = cases['proportion_harmed'].tolist()
    decrease = list()

    # False = (Decrease = 0), True = (Decrease = 1)
    for i in range(0, len(harmed), 4):
        decrease.append(False)                           # Year 2014; Temporary Place-Holder
        decrease.append(harmed[i] > harmed[i+1])         # Year 2015
        decrease.append(harmed[i+1] > harmed[i+2])       # Year 2016
        decrease.append(harmed[i+2] > harmed[i+3])       # Year 2017

    cases['decrease_in_gun_violence'] = pd.Series(decrease)
    cases = cases.drop(labels=range(0, len(decrease), 4), axis=0)
    cases = cases[(cases['state'] != 'District of Columbia')]
    
    # Combined dataset
    data = cases.merge(laws, on=['state', 'year'], how='outer')
    return data


# Create the features and labels for classification.
def create_features_labels(dataset): 
    X = dataset.drop(columns = ['decrease_in_gun_violence', 'state', 'year']) #keep all law reform info as features, as well as the proportion of individuals harmed
    y = dataset['decrease_in_gun_violence']   # Labels are if the number of cases reduced
    y = y.replace({True: 1, False: 0})
    return X, y

# Create a baseline classifier `MajorityLabelClassifier` to test our classifier against. This will always predict the class equal to the mode of the labels.
class MajorityLabelClassifier():
    # Initialize parameter for the classifier
    def __init__(self):
        self.mode = 0
    
    # Fit the data by taking training data X and their labels y and storing the learned parameter
    def fit(self, X, y):
        modes = dict()           # Stores all the modes of the training data
        y = y.tolist()
        for i in range(len(X)):
            if y[i] in modes.keys():
                modes[y[i]] += 1
            else:
                modes[y[i]] = 1
                
        # Find the most frequent mode and store it
        total = 0
        for key in modes:
            if modes[key] > total:
                total = modes[key]
                self.mode = key
    
    # Predict the label for each instance X as the learned parameter
    def predict(self, X):
        labels = list()
        for i in X:
            labels.append(self.mode)
        return labels
    
    # Calculate the accuracy of our classifier using the true and predicted labels
    def evaluate_accuracy(self, y, y_predict):
        accurate_pred = 0
        total = len(y_predict)
        true_labels = y.tolist()
        
        for i in range(total):
            if true_labels[i] == y_predict[i]:
                accurate_pred += 1
        return accurate_pred/total


class DecisionTree():
    def __init__(self):
        self.clf_model = None

    def fit(self, X_train, y_train):
        self.clf_model = DecisionTreeClassifier()   
        self.clf_model.fit(X_train,y_train)
        return self.clf_model

    def predict(self, X_test):
        return self.clf_model.predict(X_test)


    def get_decision_tree_eval_metrics(self, X, y, y_test, y_predict):
        #evaluation metrics
        train_accuracy = self.clf_model.score(X,y)  #this gives training accuracy
        test_accuracy = accuracy_score(y_test, y_predict) #this gives testing accuracy

        cm = confusion_matrix(y_test, y_predict) #confusion matrix
        precision, recall, f1, support =  precision_recall_fscore_support(y_test, y_predict) #precision, recall, f1 score, and support for each class (decrease_in_gun_violence=0 and decrease_in_gun_violence=1)

        return train_accuracy, test_accuracy, cm, precision, recall, f1

    def display_metrics(self, X, y, y_test, y_predict):
        train_accuracy, test_accuracy, cm, precision, recall, f1 =  self.get_decision_tree_eval_metrics(X, y, y_test, y_predict)
        
        print(f'Train Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}')

        print(f'precision: {precision}')
        print(f'recall   : {recall}')
        print(f'f1 score : {f1}')

        #plot confusion matrix
        sns.set(rc = {'figure.figsize':(4,4)})
        sns.heatmap(cm, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues);
        plt.ylabel('True');
        plt.xlabel('Predicted');
        plt.title('Decision Tree Confusion Matrix');
        plt.tight_layout();

    def display_tree(self, X):
        feature_names_list = list(X.columns) 
        plt.figure(figsize=(15,15))
        plot_tree(self.clf_model, feature_names = feature_names_list,  
                class_names = ['No Decrease', 'Decrease'], 
                filled = True, impurity = False); #class names match to `decrease` values [0=did not decrease, 1=did decrease]


class SVM():
    def __init__(self):
        self.svc = None

    def learn_classifier(self, X_train, y_train, kernel):
        """ learns a classifier from the input features and labels using the kernel function supplied
        Inputs:
            X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
            y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
            kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
        Outputs:
            sklearn.svm.SVC: classifier learnt from data
        """
    
        self.svc =  svm.SVC(kernel = kernel)
        return self.svc.fit(X_train, y_train)

    def evaluate_classifier(self, classifier, X_validation, y_validation):
        """ evaluates a classifier based on a supplied validation data
        Inputs:
            classifier: sklearn.svm.classes.SVC: classifer to evaluate
            X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
            y_validation: numpy.ndarray(int): dense binary vector of class labels
        Outputs:
            double: accuracy of classifier on the validation data
        """
        y_pred = classifier.predict(X_validation)
        return sklearn.metrics.accuracy_score(y_validation, y_pred)

    def best_model_selection(self, kf, X, y):
        """
        Select the kernel giving best results using k-fold cross-validation.
        Other parameters should be left default.
        Input:
        kf (sklearn.model_selection.KFold): kf object defined above
        X (scipy.sparse.csr.csr_matrix): training data
        y (array(int)): training labels
        Return:
        best_kernel (string)
        """
        kernel_accuracies = dict()
        for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
            fold_accuracies = []
            for train_index, test_index in kf.split(X):
                
                # Use the documentation of KFold cross-validation to split
                # training data and test data from create_features() and create_labels()
                X_train, y_train = X.loc[train_index], y.loc[train_index]
                X_test,  y_test  = X.loc[test_index] , y.loc[test_index]
                
                # call learn_classifer() using training split of kth fold
                classifier = self.learn_classifier(X_train, y_train, kernel)
                
                # evaluate on the test split of kth fold
                fold_accuracy = self.evaluate_classifier(classifier, X_test, y_test)
                fold_accuracies.append(fold_accuracy)

            # record avg accuracies and determine best model (kernel)
            avg_fold_accuracy = np.mean(fold_accuracies)
            kernel_accuracies[kernel] = avg_fold_accuracy

        # return best kernel as string
        print(kernel_accuracies)
        best_kernel = max(kernel_accuracies, key = kernel_accuracies.get)
        return best_kernel

    def display_metrics(self, X, y, y_test, y_predict):
        #evaluation metrics
        train_accuracy = self.svc.score(X,y)  #this gives training accuracy
        test_accuracy = accuracy_score(y_test, y_predict) #this gives testing accuracy

        print(f'Train Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}')

        cm = confusion_matrix(y_test, y_predict) #confusion matrix
        precision, recall, f1, support =  precision_recall_fscore_support(y_test, y_predict) #precision, recall, f1 score, and support for each class (decrease_in_gun_violence=0 and decrease_in_gun_violence=1)

        print(f'precision: {precision}')
        print(f'recall   : {recall}')
        print(f'f1 score : {f1}')

        #plot confusion matrix
        sns.heatmap(cm, annot = True, fmt = ".3f", square = True, cmap = plt.cm.Blues);
        plt.ylabel('True');
        plt.xlabel('Predicted');
        plt.title('SVC Confusion Matrix');
        plt.tight_layout();
