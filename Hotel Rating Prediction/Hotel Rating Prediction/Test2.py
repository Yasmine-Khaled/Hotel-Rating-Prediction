import os

from MS2 import TestScript
import pickle
import pandas as pd
from sklearn import metrics

def test2(filepath):
    df = pd.read_csv(filepath)
    #df = df.head(1000)
    df = TestScript.TestScript(df, "MS2/").clean()
    X = df.drop(columns=['Reviewer_Score'])
    y = df['Reviewer_Score']

    logisticRegression = pickle.load(open("MS2/Logistic Regression.sav", 'rb'))
    decisionTree = pickle.load(open("MS2/Decision Tree.sav", 'rb'))
    randomForest = pickle.load(open("MS2/Random Forest.sav", 'rb'))
    svc = pickle.load(open("MS2/SVC.sav", 'rb'))
    knn = pickle.load(open("MS2/KNN.sav", 'rb'))
    voting = pickle.load(open("MS2/Hard Voting.sav", 'rb'))

    logistic_pred = logisticRegression.predict(X)
    print("Logistic Regression test accuracy = {:.2f} %".format(metrics.accuracy_score(y, logistic_pred) * 100))
    df['logistic_Regression'] = logistic_pred
    print("----------------------")

    decisionTree_pred = decisionTree.predict(X)
    print("Decision Tree test accuracy = {:.2f} %".format(metrics.accuracy_score(y, decisionTree_pred) * 100))
    df['decision_Tree'] = decisionTree_pred
    print("----------------------")

    randomForest_pred = randomForest.predict(X)
    print("Random Forest test accuracy = {:.2f} %".format(metrics.accuracy_score(y, randomForest_pred) * 100))
    df['random_Forest'] = randomForest_pred
    print("----------------------")

    svc_pred = svc.predict(X)
    print("SVC test accuracy = {:.2f} %".format(metrics.accuracy_score(y, svc_pred) * 100))
    df['svc'] = svc_pred
    print("----------------------")

    knn_pred = knn.predict(X)
    print("KNN test accuracy = {:.2f} %".format(metrics.accuracy_score(y, knn_pred) * 100))
    df['knn'] = knn_pred
    print("----------------------")

    voting_pred = voting.predict(X)
    print("Hard Voting Classifier test accuracy = {:.2f} %".format(metrics.accuracy_score(y, voting_pred) * 100))
    df['voting'] = voting_pred
    print("----------------------")
    df.to_csv('test2.csv')
    os.startfile('test2.csv')



