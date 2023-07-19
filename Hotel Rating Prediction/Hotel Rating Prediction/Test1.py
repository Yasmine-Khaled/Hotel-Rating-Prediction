from sklearn import metrics
from sklearn.metrics import r2_score
from MS1 import TestScript
import pickle
import pandas as pd
import os
def test1(filepath):
    df = pd.read_csv(filepath)
    #df = df.head(1000)
    df = TestScript.TestScript(df, "MS1/").clean()
    X = df.drop(columns=['Reviewer_Score'])
    y = df['Reviewer_Score']

    linearRegression = pickle.load(open("MS1/Linear_Regression.sav", 'rb'))
    polynomialRegression = pickle.load(open("MS1/polynomial_regression.sav", 'rb'))
    poly_features = pickle.load(open("MS1/polynomial_features.sav", 'rb'))
    randomForest = pickle.load(open("MS1/random_forest.sav", 'rb'))
    ridgeRegression = pickle.load(open("MS1/Ridge_Regression.sav", 'rb'))

    linear_pred = linearRegression.predict(X)
    print('linear Mean Square Error = ', metrics.mean_squared_error(y, linear_pred))
    score = r2_score(y, linear_pred)
    print("R-squared score: {:.2f}".format(score)+"\n-------------------------")
    df['Linear_Regression'] = linear_pred

    X_poly = poly_features.fit_transform(X)
    poly_pred = polynomialRegression.predict(X_poly)
    print('polynomial regression Mean Square Error = ', metrics.mean_squared_error(y, poly_pred))
    score = r2_score(y, poly_pred)
    print("R-squared score: {:.2f}".format(score)+"\n-------------------------")
    df['Polynomial_Regression'] = poly_pred

    randomForest_pred = randomForest.predict(X)
    print('random forest Mean Square Error = ', metrics.mean_squared_error(y, randomForest_pred))
    score = r2_score(y, randomForest_pred)
    print("R-squared score: {:.2f}".format(score)+"\n-------------------------")
    df['Random_Forest'] = randomForest_pred

    ridge_pred = ridgeRegression.predict(X)
    print('ridge regression Mean Square Error = ', metrics.mean_squared_error(y, ridge_pred))
    score = r2_score(y, ridge_pred)
    print("R-squared score: {:.2f}".format(score)+"\n-------------------------")
    df['Ridge_Regression'] = ridge_pred
    df.to_csv('test1.csv')
    os.startfile('test1.csv')
