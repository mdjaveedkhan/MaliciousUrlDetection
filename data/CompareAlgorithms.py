#importing required libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier



# Creating holders to store the model performance results
ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

# function to call for storing the results
def storeResults(model, a, b, c, d):
    ML_Model.append(model)
    accuracy.append(round(a, 3))
    f1_score.append(round(b, 3))
    recall.append(round(c, 3))
    precision.append(round(d, 3))


def CompareAlgorithms():
    # Loading data into dataframe

    data = pd.read_csv("../phishing.csv")

    data = data.drop(['Index'], axis=1)

    # Splitting the dataset into dependant and independant fetature
    y = data['class']
    X = data.drop('class', axis=1)
    X.shape, y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    # Logistic Regression Classifier Model
    # instantiate the model
    log = LogisticRegression()

    # fit the model
    log.fit(X_train, y_train)

    # predicting the target value from the model for the samples

    y_train_log = log.predict(X_train)
    y_test_log = log.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_log = metrics.accuracy_score(y_train, y_train_log)
    acc_test_log = metrics.accuracy_score(y_test, y_test_log)
    print("Logistic Regression : Accuracy on training Data: {:.3f}".format(acc_train_log))
    print("Logistic Regression : Accuracy on test Data: {:.3f}".format(acc_test_log))
    print()

    f1_score_train_log = metrics.f1_score(y_train, y_train_log)
    f1_score_test_log = metrics.f1_score(y_test, y_test_log)
    print("Logistic Regression : f1_score on training Data: {:.3f}".format(f1_score_train_log))
    print("Logistic Regression : f1_score on test Data: {:.3f}".format(f1_score_test_log))
    print()

    recall_score_train_log = metrics.recall_score(y_train, y_train_log)
    recall_score_test_log = metrics.recall_score(y_test, y_test_log)
    print("Logistic Regression : Recall on training Data: {:.3f}".format(recall_score_train_log))
    print("Logistic Regression : Recall on test Data: {:.3f}".format(recall_score_test_log))
    print()

    precision_score_train_log = metrics.precision_score(y_train, y_train_log)
    precision_score_test_log = metrics.precision_score(y_test, y_test_log)
    print("Logistic Regression : precision on training Data: {:.3f}".format(precision_score_train_log))
    print("Logistic Regression : precision on test Data: {:.3f}".format(precision_score_test_log))

    print(metrics.classification_report(y_test, y_test_log))

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Logistic Regression', acc_test_log, f1_score_test_log,recall_score_train_log, precision_score_train_log)
    #######################################################################################################################
    # KNNClassifier Model

    # instantiate the model
    knn = KNeighborsClassifier(n_neighbors=1)

    # fit the model
    knn.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_knn = knn.predict(X_train)
    y_test_knn = knn.predict(X_test)

    # computing the accuracy,f1_score,Recall,precision of the model performance

    acc_train_knn = metrics.accuracy_score(y_train, y_train_knn)
    acc_test_knn = metrics.accuracy_score(y_test, y_test_knn)
    print("K-Nearest Neighbors : Accuracy on training Data: {:.3f}".format(acc_train_knn))
    print("K-Nearest Neighbors : Accuracy on test Data: {:.3f}".format(acc_test_knn))
    print()

    f1_score_train_knn = metrics.f1_score(y_train, y_train_knn)
    f1_score_test_knn = metrics.f1_score(y_test, y_test_knn)
    print("K-Nearest Neighbors : f1_score on training Data: {:.3f}".format(f1_score_train_knn))
    print("K-Nearest Neighbors : f1_score on test Data: {:.3f}".format(f1_score_test_knn))
    print()

    recall_score_train_knn = metrics.recall_score(y_train, y_train_knn)
    recall_score_test_knn = metrics.recall_score(y_test, y_test_knn)
    print("K-Nearest Neighborsn : Recall on training Data: {:.3f}".format(recall_score_train_knn))
    print("Logistic Regression : Recall on test Data: {:.3f}".format(recall_score_test_knn))
    print()

    precision_score_train_knn = metrics.precision_score(y_train, y_train_knn)
    precision_score_test_knn = metrics.precision_score(y_test, y_test_knn)
    print("K-Nearest Neighbors : precision on training Data: {:.3f}".format(precision_score_train_knn))
    print("K-Nearest Neighbors : precision on test Data: {:.3f}".format(precision_score_test_knn))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_knn))

    training_accuracy = []
    test_accuracy = []
    # try max_depth from 1 to 20
    depth = range(1, 20)
    for n in depth:
        knn = KNeighborsClassifier(n_neighbors=n)

        knn.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(knn.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    # plotting the training & testing accuracy for n_estimators from 1 to 20
    plt.plot(depth, training_accuracy, label="training accuracy")
    plt.plot(depth, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig('../static/vis/knn_acc.jpg')

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('K-Nearest Neighbors', acc_test_knn, f1_score_test_knn,
                 recall_score_train_knn, precision_score_train_knn)
    #######################################################################################################################
    # SVM Classifier Model

    # defining parameter range
    param_grid = {'gamma': [0.1], 'kernel': ['rbf', 'linear']}

    svc = GridSearchCV(SVC(), param_grid)

    # fitting the model for grid search
    svc.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_train_svc = svc.predict(X_train)
    y_test_svc = svc.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_svc = metrics.accuracy_score(y_train, y_train_svc)
    acc_test_svc = metrics.accuracy_score(y_test, y_test_svc)
    print("Support Vector Machine : Accuracy on training Data: {:.3f}".format(acc_train_svc))
    print("Support Vector Machine : Accuracy on test Data: {:.3f}".format(acc_test_svc))
    print()

    f1_score_train_svc = metrics.f1_score(y_train, y_train_svc)
    f1_score_test_svc = metrics.f1_score(y_test, y_test_svc)
    print("Support Vector Machine : f1_score on training Data: {:.3f}".format(f1_score_train_svc))
    print("Support Vector Machine : f1_score on test Data: {:.3f}".format(f1_score_test_svc))
    print()

    recall_score_train_svc = metrics.recall_score(y_train, y_train_svc)
    recall_score_test_svc = metrics.recall_score(y_test, y_test_svc)
    print("Support Vector Machine : Recall on training Data: {:.3f}".format(recall_score_train_svc))
    print("Support Vector Machine : Recall on test Data: {:.3f}".format(recall_score_test_svc))
    print()

    precision_score_train_svc = metrics.precision_score(y_train, y_train_svc)
    precision_score_test_svc = metrics.precision_score(y_test, y_test_svc)
    print("Support Vector Machine : precision on training Data: {:.3f}".format(precision_score_train_svc))
    print("Support Vector Machine : precision on test Data: {:.3f}".format(precision_score_test_svc))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_svc))

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Support Vector Machine', acc_test_svc, f1_score_test_svc,
                 recall_score_train_svc, precision_score_train_svc)
    #######################################################################################################################
    # Naive Bayes Classifier Model
    # instantiate the model
    nb = GaussianNB()

    # fit the model
    nb.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_nb = nb.predict(X_train)
    y_test_nb = nb.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_nb = metrics.accuracy_score(y_train, y_train_nb)
    acc_test_nb = metrics.accuracy_score(y_test, y_test_nb)
    print("Naive Bayes Classifier : Accuracy on training Data: {:.3f}".format(acc_train_nb))
    print("Naive Bayes Classifier : Accuracy on test Data: {:.3f}".format(acc_test_nb))
    print()

    f1_score_train_nb = metrics.f1_score(y_train, y_train_nb)
    f1_score_test_nb = metrics.f1_score(y_test, y_test_nb)
    print("Naive Bayes Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_nb))
    print("Naive Bayes Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_nb))
    print()

    recall_score_train_nb = metrics.recall_score(y_train, y_train_nb)
    recall_score_test_nb = metrics.recall_score(y_test, y_test_nb)
    print("Naive Bayes Classifier : Recall on training Data: {:.3f}".format(recall_score_train_nb))
    print("Naive Bayes Classifier : Recall on test Data: {:.3f}".format(recall_score_test_nb))
    print()

    precision_score_train_nb = metrics.precision_score(y_train, y_train_nb)
    precision_score_test_nb = metrics.precision_score(y_test, y_test_nb)
    print("Naive Bayes Classifier : precision on training Data: {:.3f}".format(precision_score_train_nb))
    print("Naive Bayes Classifier : precision on test Data: {:.3f}".format(precision_score_test_nb))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_nb))

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Naive Bayes Classifier', acc_test_nb, f1_score_test_nb,
                 recall_score_train_nb, precision_score_train_nb)
    #######################################################################################################################
    # Decision Tree Classifier model
    # instantiate the model
    tree = DecisionTreeClassifier(max_depth=30)

    # fit the model
    tree.fit(X_train, y_train)
    # predicting the target value from the model for the samples

    y_train_tree = tree.predict(X_train)
    y_test_tree = tree.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_tree = metrics.accuracy_score(y_train, y_train_tree)
    acc_test_tree = metrics.accuracy_score(y_test, y_test_tree)
    print("Decision Tree : Accuracy on training Data: {:.3f}".format(acc_train_tree))
    print("Decision Tree : Accuracy on test Data: {:.3f}".format(acc_test_tree))
    print()

    f1_score_train_tree = metrics.f1_score(y_train, y_train_tree)
    f1_score_test_tree = metrics.f1_score(y_test, y_test_tree)
    print("Decision Tree : f1_score on training Data: {:.3f}".format(f1_score_train_tree))
    print("Decision Tree : f1_score on test Data: {:.3f}".format(f1_score_test_tree))
    print()

    recall_score_train_tree = metrics.recall_score(y_train, y_train_tree)
    recall_score_test_tree = metrics.recall_score(y_test, y_test_tree)
    print("Decision Tree : Recall on training Data: {:.3f}".format(recall_score_train_tree))
    print("Decision Tree : Recall on test Data: {:.3f}".format(recall_score_test_tree))
    print()

    precision_score_train_tree = metrics.precision_score(y_train, y_train_tree)
    precision_score_test_tree = metrics.precision_score(y_test, y_test_tree)
    print("Decision Tree : precision on training Data: {:.3f}".format(precision_score_train_tree))
    print("Decision Tree : precision on test Data: {:.3f}".format(precision_score_test_tree))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_tree))

    training_accuracy = []
    test_accuracy = []
    # try max_depth from 1 to 30
    depth = range(1, 30)
    for n in depth:
        tree_test = DecisionTreeClassifier(max_depth=n)

        tree_test.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(tree_test.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(tree_test.score(X_test, y_test))

    # plotting the training & testing accuracy for max_depth from 1 to 30
    plt.plot(depth, training_accuracy, label="training accuracy")
    plt.plot(depth, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("max_depth")
    plt.legend()
    plt.savefig('../static/vis/dt_acc.jpg')

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Decision Tree', acc_test_tree, f1_score_test_tree,
                 recall_score_train_tree, precision_score_train_tree)
    #######################################################################################################################
    # Random Forest Classifier Model
    # instantiate the model
    forest = RandomForestClassifier(n_estimators=10)

    # fit the model
    forest.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_train_forest = forest.predict(X_train)
    y_test_forest = forest.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_forest = metrics.accuracy_score(y_train, y_train_forest)
    acc_test_forest = metrics.accuracy_score(y_test, y_test_forest)
    print("Random Forest : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    print("Random Forest : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    print()

    f1_score_train_forest = metrics.f1_score(y_train, y_train_forest)
    f1_score_test_forest = metrics.f1_score(y_test, y_test_forest)
    print("Random Forest : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    print("Random Forest : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    print()

    recall_score_train_forest = metrics.recall_score(y_train, y_train_forest)
    recall_score_test_forest = metrics.recall_score(y_test, y_test_forest)
    print("Random Forest : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    print("Random Forest : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    print()

    precision_score_train_forest = metrics.precision_score(y_train, y_train_forest)
    precision_score_test_forest = metrics.precision_score(y_test, y_test_forest)
    print("Random Forest : precision on training Data: {:.3f}".format(precision_score_train_forest))
    print("Random Forest : precision on test Data: {:.3f}".format(precision_score_test_forest))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_forest))

    training_accuracy = []
    test_accuracy = []
    # try max_depth from 1 to 20
    depth = range(1, 20)
    for n in depth:
        forest_test = RandomForestClassifier(n_estimators=n)

        forest_test.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(forest_test.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(forest_test.score(X_test, y_test))

    # plotting the training & testing accuracy for n_estimators from 1 to 20
    plt.figure(figsize=None)
    plt.plot(depth, training_accuracy, label="training accuracy")
    plt.plot(depth, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_estimators")
    plt.legend()
    plt.savefig('../static/vis/rf_acc.jpg')

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Random Forest', acc_test_forest, f1_score_test_forest,
                 recall_score_train_forest, precision_score_train_forest)
    #######################################################################################################################
    # Gradient Boosting Classifier Model
    # instantiate the model
    gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)

    # fit the model
    gbc.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_gbc = gbc.predict(X_train)
    y_test_gbc = gbc.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_gbc = metrics.accuracy_score(y_train, y_train_gbc)
    acc_test_gbc = metrics.accuracy_score(y_test, y_test_gbc)
    print("Gradient Boosting Classifier : Accuracy on training Data: {:.3f}".format(acc_train_gbc))
    print("Gradient Boosting Classifier : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
    print()

    f1_score_train_gbc = metrics.f1_score(y_train, y_train_gbc, average='macro')
    f1_score_test_gbc = metrics.f1_score(y_test, y_test_gbc, average='macro')
    print("Gradient Boosting Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_gbc))
    print("Gradient Boosting Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_gbc))
    print()

    recall_score_train_gbc = metrics.recall_score(y_train, y_train_gbc, average='macro')
    recall_score_test_gbc = metrics.recall_score(y_test, y_test_gbc, average='macro')
    print("Gradient Boosting Classifier : Recall on training Data: {:.3f}".format(recall_score_train_gbc))
    print("Gradient Boosting Classifier : Recall on test Data: {:.3f}".format(recall_score_test_gbc))
    print()

    precision_score_train_gbc = metrics.precision_score(y_train, y_train_gbc, average='macro')
    precision_score_test_gbc = metrics.precision_score(y_test, y_test_gbc, average='macro')
    print("Gradient Boosting Classifier : precision on training Data: {:.3f}".format(precision_score_train_gbc))
    print("Gradient Boosting Classifier : precision on test Data: {:.3f}".format(precision_score_test_gbc))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_gbc))

    training_accuracy = []
    test_accuracy = []
    # try learning_rate from 0.1 to 0.9
    depth = range(1, 10)
    for n in depth:
        forest_test = GradientBoostingClassifier(learning_rate=n * 0.1)

        forest_test.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(forest_test.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(forest_test.score(X_test, y_test))

    # plotting the training & testing accuracy for n_estimators from 1 to 50
    plt.figure(figsize=None)
    plt.plot(depth, training_accuracy, label="training accuracy")
    plt.plot(depth, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("learning_rate")
    plt.legend()
    plt.savefig('../static/vis/gb_acc.jpg')

    training_accuracy = []
    test_accuracy = []
    # try learning_rate from 0.1 to 0.9
    depth = range(1, 10, 1)
    for n in depth:
        forest_test = GradientBoostingClassifier(max_depth=n, learning_rate=0.7)

        forest_test.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(forest_test.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(forest_test.score(X_test, y_test))

    # plotting the training & testing accuracy for n_estimators from 1 to 50
    plt.figure(figsize=None)
    plt.plot(depth, training_accuracy, label="training accuracy")
    plt.plot(depth, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("max_depth")
    plt.legend();
    plt.savefig('../static/vis/gb_acc_dep.jpg')

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Gradient Boosting Classifier', acc_test_gbc, f1_score_test_gbc,
                 recall_score_train_gbc, precision_score_train_gbc)
    ###################################################################################################################
    #  catboost Classifier Model
    # instantiate the model
    cat = CatBoostClassifier(learning_rate=0.1)

    # fit the model
    cat.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_cat = cat.predict(X_train)
    y_test_cat = cat.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_cat = metrics.accuracy_score(y_train, y_train_cat)
    acc_test_cat = metrics.accuracy_score(y_test, y_test_cat)
    print("CatBoost Classifier : Accuracy on training Data: {:.3f}".format(acc_train_cat))
    print("CatBoost Classifier : Accuracy on test Data: {:.3f}".format(acc_test_cat))
    print()

    f1_score_train_cat = metrics.f1_score(y_train, y_train_cat, average='macro')
    f1_score_test_cat = metrics.f1_score(y_test, y_test_cat, average='macro')
    print("CatBoost Classifier : f1_score on training Data: {:.3f}".format(f1_score_train_cat))
    print("CatBoost Classifier : f1_score on test Data: {:.3f}".format(f1_score_test_cat))
    print()

    recall_score_train_cat = metrics.recall_score(y_train, y_train_cat, average='macro')
    recall_score_test_cat = metrics.recall_score(y_test, y_test_cat, average='macro')
    print("CatBoost Classifier : Recall on training Data: {:.3f}".format(recall_score_train_cat))
    print("CatBoost Classifier : Recall on test Data: {:.3f}".format(recall_score_test_cat))
    print()

    precision_score_train_cat = metrics.precision_score(y_train, y_train_cat, average='macro')
    precision_score_test_cat = metrics.precision_score(y_test, y_test_cat, average='macro')
    print("CatBoost Classifier : precision on training Data: {:.3f}".format(precision_score_train_cat))
    print("CatBoost Classifier : precision on test Data: {:.3f}".format(precision_score_test_cat))

    # computing the classification report of the model

    print(metrics.classification_report(y_test, y_test_cat))

    training_accuracy = []
    test_accuracy = []
    # try learning_rate from 0.1 to 0.9
    depth = range(1, 10)
    for n in depth:
        forest_test = CatBoostClassifier(learning_rate=n * 0.1)

        forest_test.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(forest_test.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(forest_test.score(X_test, y_test))


    # plotting the training & testing accuracy for n_estimators from 1 to 50
    plt.figure(figsize=None)
    plt.plot(depth, training_accuracy, label="training accuracy")
    plt.plot(depth, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("learning_rate")
    plt.legend()
    plt.savefig('../static/vis/cb_acc.jpg')

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('CatBoost Classifier', acc_test_cat, f1_score_test_cat,
                 recall_score_train_cat, precision_score_train_cat)

    # Multi-layer Perceptron Classifier Model
    # instantiate the model
    mlp = MLPClassifier()
    # mlp = GridSearchCV(mlpc, parameter_space)

    # fit the model
    mlp.fit(X_train, y_train)

    # predicting the target value from the model for the samples
    y_train_mlp = mlp.predict(X_train)
    y_test_mlp = mlp.predict(X_test)

    # computing the accuracy, f1_score, Recall, precision of the model performance

    acc_train_mlp = metrics.accuracy_score(y_train, y_train_mlp)
    acc_test_mlp = metrics.accuracy_score(y_test, y_test_mlp)
    print("Multi-layer Perceptron : Accuracy on training Data: {:.3f}".format(acc_train_mlp))
    print("Multi-layer Perceptron : Accuracy on test Data: {:.3f}".format(acc_test_mlp))
    print()

    f1_score_train_mlp = metrics.f1_score(y_train, y_train_mlp, average='micro')
    f1_score_test_mlp = metrics.f1_score(y_test, y_test_mlp, average='micro')
    print("Multi-layer Perceptron : f1_score on training Data: {:.3f}".format(f1_score_train_mlp))
    print("Multi-layer Perceptron : f1_score on test Data: {:.3f}".format(f1_score_train_mlp))
    print()

    recall_score_train_mlp = metrics.recall_score(y_train, y_train_mlp, average='micro')
    recall_score_test_mlp = metrics.recall_score(y_test, y_test_mlp, average='micro')
    print("Multi-layer Perceptron : Recall on training Data: {:.3f}".format(recall_score_train_mlp))
    print("Multi-layer Perceptron : Recall on test Data: {:.3f}".format(recall_score_test_mlp))
    print()

    precision_score_train_mlp = metrics.precision_score(y_train, y_train_mlp, average='micro')
    precision_score_test_mlp = metrics.precision_score(y_test, y_test_mlp, average='micro')
    print("Multi-layer Perceptron : precision on training Data: {:.3f}".format(precision_score_train_mlp))
    print("Multi-layer Perceptron : precision on test Data: {:.3f}".format(precision_score_test_mlp))

    # storing the results. The below mentioned order of parameter passing is important.

    storeResults('Multi-layer Perceptron', acc_test_mlp, f1_score_test_mlp,
                 recall_score_train_mlp, precision_score_train_mlp)

    # creating dataframe
    result = pd.DataFrame({'ML Model': ML_Model,
                           'Accuracy': accuracy,
                           'f1_score': f1_score,
                           'Recall': recall,
                           'Precision': precision,
                           })

    # Sorting the datafram on accuracy
    sorted_result = result.sort_values(by=['Accuracy', 'f1_score'], ascending=False).reset_index(drop=True)
    print(sorted_result)

    # Create a DataFrame from the sorted results
    data = {
        'ML Model': [
            'Gradient Boosting Classifier',
            'CatBoost Classifier',
            'Random Forest',
            'Support Vector Machine',
            'Multi-layer Perceptron',
            'Decision Tree',
            'K-Nearest Neighbors',
            'Logistic Regression',
            'Naive Bayes Classifier'
        ],
        'Accuracy': [0.974, 0.972, 0.967, 0.964, 0.963, 0.962, 0.956, 0.934, 0.605],
        'f1_score': [0.974, 0.972, 0.971, 0.968, 0.963, 0.966, 0.961, 0.941, 0.454],
        'Recall': [0.988, 0.990, 0.993, 0.980, 0.984, 0.991, 0.991, 0.943, 0.292],
        'Precision': [0.989, 0.991, 0.990, 0.965, 0.984, 0.993, 0.989, 0.927, 0.997]
    }

    df = pd.DataFrame(data)

    # Set 'ML Model' as index
    df.set_index('ML Model', inplace=True)

    # Plot the scores for each model
    fig, ax = plt.subplots(figsize=(10, 10))
    df.plot(kind='bar', ax=ax)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylim([0, 1])  # Assuming the scores range from 0 to 1
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Scores')
    plt.legend(loc='lower right')
    plt.savefig('../static/vis/Algcomp.jpg')
    plt.clf()

    confusion_matrix = metrics.confusion_matrix(y_test, y_test_gbc)


    class_names = [0, 1]

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actuals')
    plt.xlabel('Predicted')
    plt.savefig('../static/vis/cnf_gb.jpg')
    plt.clf()


    # dump information to that file
    pickle.dump(gbc, open('../gbc_malicious.pkl', 'wb'))

   #gbc = pickle.load(open("newmodel.pkl", "rb"))

#CompareAlgorithms()
























