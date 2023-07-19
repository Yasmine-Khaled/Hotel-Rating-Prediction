import pickle
import nltk
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from TrainingClean import TrainingClean
from TestScript import TestScript
from sklearn.metrics import r2_score

nltk.download('wordnet')
nltk.download('omw-1.4')

def wordCloud_generator(data, title=None,name=''):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10
                          ).generate(" ".join(data.values))
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=30)
    plt.savefig(name+'.png')
    plt.show()


data = pd.read_csv('hotel-regression-dataset.csv')
data.dropna(subset=['Reviewer_Score'], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=0)
TrainDF = pd.concat([X_train, y_train], axis=1)
TestDF = pd.concat([X_test, y_test], axis=1)
TrainDF.reset_index(inplace=True)
TestDF.reset_index(inplace=True)
wordCloud_generator(data['Negative_Review'], title="Most used words in Negative Review",name='Negative_Review')
wordCloud_generator(data['Positive_Review'], title="Most used words in Positive Review",name='Positive_Review')
TrainDF = TrainingClean(TrainDF).clean()
TestDF = TestScript(TestDF, "").clean()
X_train, X_test, y_train, y_test = TrainDF.drop(columns=['Reviewer_Score']), TestDF.drop(columns=['Reviewer_Score']),\
                                   TrainDF['Reviewer_Score'], TestDF['Reviewer_Score']

# models
linear = LinearRegression()
params_linear = {'fit_intercept': [True, False]}
GridSearch_linear = GridSearchCV(linear, params_linear, return_train_score=True)
GridSearch_linear.fit(X_train, y_train)
best_params = GridSearch_linear.best_params_
best_score = GridSearch_linear.best_score_
print(f"Best parameters : {best_params}")
print(f"Best Score : {best_score}")
best_model = LinearRegression(**best_params)
best_model.fit(X_train, y_train)
best_prediction = best_model.predict(X_test)
pickle.dump(best_model, open('Linear_Regression.sav', 'wb'))
print('linear Mean Square Error = ', metrics.mean_squared_error(y_test, best_prediction))
print("r2 score = ", r2_score(y_test, best_prediction))
print("________________________________________________________________________________________________")

# polynomial Regression
polynomial_model = Pipeline([('poly', PolynomialFeatures()), ('linear', LinearRegression())])
polynomial_params = {'poly__include_bias': [False, True], 'poly__degree': np.arange(2, 11),
                     'linear__fit_intercept': [True, False]}
polynomial_grid = GridSearchCV(polynomial_model, polynomial_params)
polynomial_grid.fit(X_train, y_train)
print(f"Best parameters : {polynomial_grid.best_params_}")
print(f"Best Score : {polynomial_grid.best_score_}")
poly_features = PolynomialFeatures(degree=polynomial_grid.best_params_['poly__degree'],
                                   include_bias=polynomial_grid.best_params_['poly__include_bias'])
x_poly_train = poly_features.fit_transform(X_train)
poly_reg = LinearRegression(fit_intercept=polynomial_grid.best_params_['linear__fit_intercept'])
poly_reg.fit(x_poly_train, y_train)
x_poly_test = poly_features.fit_transform(X_test)
poly_predict = poly_reg.predict(x_poly_test)
pickle.dump(poly_reg, open('polynomial_regression.sav', 'wb'))
pickle.dump(poly_features, open('polynomial_features.sav', 'wb'))
print('polynomial Mean Square Error = ', metrics.mean_squared_error(y_test, poly_predict))

data['Expected'] = pd.Series(y_test)
data['Predicted'] = pd.Series(poly_predict)

figure = plt.figure(figsize=(15, 10))

axes = sns.scatterplot(data=data, x='Expected', y='Predicted',
                       hue='Predicted', palette='cool',
                       legend=False)

start = min(y_test.min(), poly_predict.min())
end = max(y_test.max(), poly_predict.max())

axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt.plot([start, end], [start, end], 'k--')
print("r2 score = ", r2_score(y_test, poly_predict))
print("________________________________________________________________________________________________")

# random forest Regression
randomForest = RandomForestRegressor()
randomForest_params = {'max_depth': np.arange(2, 15)}
grid = GridSearchCV(randomForest, randomForest_params)
grid.fit(X_train, y_train)
best_params = grid.best_params_
best_score = grid.best_score_
print(f"Best parameters : {best_params}")
print(f"Best Score : {best_score}")
randomForest = RandomForestRegressor(**best_params)
randomForest.fit(X_train, y_train)
y_pred = randomForest.predict(X_test)
pickle.dump(randomForest, open('random_forest.sav', 'wb'))
print('random forest Mean Square Error = ', metrics.mean_squared_error(y_test, y_pred))
# plot actual vs predicted target values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Random Forest Regression Results')
plt.show()
print("r2 score = ", r2_score(y_test, y_pred))
print("________________________________________________________________________________________________")

# Ridge Regression
ridge = linear_model.Ridge()
params_ridge = {'solver': ['svd', 'cholesky', 'lsqr', 'sag'],
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                'fit_intercept': [True, False]}
GridSearch_ridge = GridSearchCV(ridge, params_ridge)
GridSearch_ridge.fit(X_train, y_train)
best_params_ridge = GridSearch_ridge.best_params_
best_score_ridge = GridSearch_ridge.best_score_
print(f"Best parameters : {best_params_ridge}")
print(f"Best Score : {best_score_ridge}")
best_model_ridge = linear_model.Ridge(**best_params_ridge)
best_model_ridge.fit(X_train, y_train)
best_prediction_ridge = best_model_ridge.predict(X_test)
pickle.dump(best_model_ridge, open('Ridge_Regression.sav', 'wb'))
print('ridge Mean Square Error = ', metrics.mean_squared_error(y_test, best_prediction_ridge))
print("r2 score = ", r2_score(y_test, best_prediction_ridge))
print("________________________________________________________________________________________________")

print(TrainDF.shape)
print(TestDF.shape)
print(data.describe().to_string())
