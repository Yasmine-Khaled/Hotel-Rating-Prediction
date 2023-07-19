import itertools
import pickle
import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from TrainingClean import TrainingClean
from TestScript import TestScript
import time
from wordcloud import WordCloud
from sklearn.metrics import roc_curve
from sklearn import metrics

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

def perform_grid_search(model, param_grid, x, y):
    # Perform grid search to find the best hyperparameters
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(x, y)
    params = grid.best_params_
    best_score = grid.best_score_
    print(f"Best parameters : {params}")
    print(f"Best Score : {best_score}")

    # Return the best model
    return params


def Evaluation(test, pred):
    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(test, pred) * 100:.2f}%")
    print(classification_report(test, pred, target_names=["High_Reviewer_Score", "Intermediate_Reviewer_Score",
                                                          "Low_Reviewer_Score"]))



def plot_confusion_matrix(y_true, y_pred, title=''):
    cm=confusion_matrix(y_true, y_pred)
    normalize = True
    classes=['High_Reviewer_Score','Intermediate_Reviewer_Score','Low_Reviewer_Score']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix for '+title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'.png')
    plt.show()


def plot_roc_curve(model, x, y, model_name):
    y_pred_proba = model.predict_proba(x)
    y_test_bin = (y == 1).astype(int)
    y_pred_proba_bin = y_pred_proba[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test_bin, y_pred_proba_bin)
    auc_score = metrics.roc_auc_score(y_test_bin, y_pred_proba_bin)
    plt.plot(fpr, tpr, "*-")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend([model_name, 'Random chance'])
    plt.title('ROC Curve For {} (AUC={:.3f})'.format(model_name, auc_score))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(model_name + 'Roc_Curve.png')
    plt.show()


data = pd.read_csv('hotel-classification-dataset.csv')
Accuracy, TrainingTime, TestTime = {}, {}, {}

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=0)
TrainDF = pd.concat([X_train, y_train], axis=1)
TestDF = pd.concat([X_test, y_test], axis=1)
TrainDF.reset_index(inplace=True)
TestDF.reset_index(inplace=True)
wordCloud_generator(data['Negative_Review'], title="Most used words in Negative Review",name='Negative_Review')
wordCloud_generator(data['Positive_Review'], title="Most used words in Positive Review",name='Positive_Review')
print(TrainDF['Reviewer_Score'].value_counts())
TrainDF = TrainingClean(TrainDF).clean()

# Plot a histogram of the data
# plt.hist(TrainDF['Average_Score'], bins=30)
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.show()

print(TrainDF.info())
TrainDF.to_csv('train.csv')
print(TrainDF['Reviewer_Score'].value_counts())
TestDF = TestScript(TestDF, "").clean()
TestDF.to_csv('test.csv')
X_train, X_test, y_train, y_test = TrainDF.drop(columns=['Reviewer_Score']), TestDF.drop(columns=['Reviewer_Score']), \
                                   TrainDF['Reviewer_Score'], TestDF['Reviewer_Score']

# models
print("Logistic Regression model: ")
logistic_regression = LogisticRegression()
logistic_grid = {
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'sag', 'lbfgs', 'saga'],
    'multi_class': ['ovr', 'multinomial']
}
best_params = perform_grid_search(logistic_regression, logistic_grid, X_train, y_train)
logisticRegression = LogisticRegression(**best_params, random_state=42)
start_time = time.time()
logisticRegression.fit(X_train, y_train)
TrainingTime['Logistic Regression'] = time.time() - start_time
print(f"Training time :{TrainingTime['Logistic Regression']:.2f} seconds")
pickle.dump(logisticRegression, open('Logistic Regression.sav', 'wb'))
start_time = time.time()
y_pred_logreg = logisticRegression.predict(X_test)
TestTime['Logistic Regression'] = time.time() - start_time
print(f"Testing time :{TestTime['Logistic Regression']:.2f} seconds")
Evaluation(y_test, y_pred_logreg)
Accuracy['Logistic Regression'] = float(f"{accuracy_score(y_test, y_pred_logreg) * 100:.2f}")
plot_confusion_matrix(y_test, y_pred_logreg, 'Logistic Regression')
plot_roc_curve(logisticRegression, X_test, y_test, "Logistic Regression")
print('_____________________________________________')

print("Decision Tree model: ")
Decision_tree = DecisionTreeClassifier()
Decision_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}
best_params = perform_grid_search(Decision_tree, Decision_grid, X_train, y_train)
DecisionTree = DecisionTreeClassifier(**best_params, random_state=42)
start_time = time.time()
DecisionTree.fit(X_train, y_train)
TrainingTime['Decision Tree'] = time.time() - start_time
print(f"Training time :{TrainingTime['Decision Tree']:.2f} seconds")
pickle.dump(DecisionTree, open('Decision Tree.sav', 'wb'))
start_time = time.time()
y_pred_Dtree = DecisionTree.predict(X_test)
TestTime['Decision Tree'] = time.time() - start_time
print(f"Testing time :{TestTime['Decision Tree']:.2f} seconds")
Evaluation(y_test, y_pred_Dtree)
Accuracy['Decision Tree'] = float(f"{accuracy_score(y_test, y_pred_Dtree) * 100:.2f}")
plot_confusion_matrix(y_test, y_pred_Dtree, 'Decision Tree')
plot_roc_curve(DecisionTree, X_test, y_test, "Decision Tree")
print('_____________________________________________')

print("Random Forest model: ")
# Random_forest = RandomForestClassifier()
# Random_grid = {
#     'n_estimators': [100, 1000],
#     'max_depth': [2, 3, 4, 5, None],
#     'min_samples_split': [2, 3, 4],
#     'min_samples_leaf': [1, 2, 3]}
# best_params = perform_grid_search(Random_forest, Random_grid, X_train, y_train)
# RandomForest = RandomForestClassifier(**best_params, random_state=42)
RandomForest = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=5, min_samples_leaf=3,
                                      min_samples_split=3)
start_time = time.time()
RandomForest.fit(X_train, y_train)
TrainingTime['Random Forest'] = time.time() - start_time
print(f"Training time :{TrainingTime['Random Forest']:.2f} seconds")
pickle.dump(RandomForest, open('Random Forest.sav', 'wb'))
start_time = time.time()
y_pred_random = RandomForest.predict(X_test)
TestTime['Random Forest'] = time.time() - start_time
print(f"Testing time :{TestTime['Random Forest']:.2f} seconds")
print(f"Accuracy: {accuracy_score(y_test, y_pred_random) * 100:.2f}%")
labels = ["Low_Reviewer_Score", "Intermediate_Reviewer_Score", "High_Reviewer_Score"]
print(classification_report(y_test, y_pred_random, target_names=labels, zero_division=0))
Accuracy['Random Forest'] = float(f"{accuracy_score(y_test, y_pred_random) * 100:.2f}")
plot_confusion_matrix(y_test, y_pred_random, 'Random Forest')
plot_roc_curve(RandomForest, X_test, y_test, "Random Forest")
print('_____________________________________________')

print("SVC model: ")
svc = SVC(random_state=42)
start_time = time.time()
svc.fit(X_train, y_train)
TrainingTime['SVC'] = time.time() - start_time
print(f"Training time :{TrainingTime['SVC']:.2f} seconds")
pickle.dump(svc, open('SVC.sav', 'wb'))
start_time = time.time()
y_pred_svc = svc.predict(X_test)
TestTime['SVC'] = time.time() - start_time
print(f"Testing time :{TestTime['SVC']:.2f} seconds")
Evaluation(y_test, y_pred_svc)
Accuracy['SVC'] = float(f"{accuracy_score(y_test, y_pred_svc) * 100:.2f}")
plot_confusion_matrix(y_test, y_pred_svc, 'SVC')
print('_____________________________________________')

print("KNN model: ")
knn = KNeighborsClassifier()
knn_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
best_params = perform_grid_search(knn, knn_grid, X_train, y_train)
best_knn = KNeighborsClassifier(**best_params)
start_time = time.time()
best_knn.fit(X_train, y_train)
TrainingTime['KNN'] = time.time() - start_time
print(f"Training time :{TrainingTime['KNN']:.2f} seconds")
pickle.dump(best_knn, open('KNN.sav', 'wb'))
start_time = time.time()
y_pred_knn = best_knn.predict(X_test)
TestTime['KNN'] = time.time() - start_time
print(f"Testing time :{TestTime['KNN']:.2f} seconds")
Evaluation(y_test, y_pred_knn)
Accuracy['KNN'] = float(f"{accuracy_score(y_test, y_pred_knn) * 100:.2f}")
plot_confusion_matrix(y_test, y_pred_knn, 'KNN')
plot_roc_curve(best_knn, X_test, y_test, "KNN")
print('_____________________________________________')

# Create the hard voting classifier
print("Hard Voting Classifier: ")
voting_clf = VotingClassifier(
    estimators=[('lr', logisticRegression), ('dt', DecisionTree), ('rf', RandomForest), ('svc', svc),
                ('knn', best_knn)],
    voting='hard'
)
start_time = time.time()
voting_clf.fit(X_train, y_train)
TrainingTime['Hard Voting'] = time.time() - start_time
print(f"Training time :{TrainingTime['Hard Voting']:.2f} seconds")
start_time = time.time()
y_pred_voting = voting_clf.predict(X_test)
TestTime['Hard Voting'] = time.time() - start_time
print(f"Testing time :{TestTime['Hard Voting']:.2f} seconds")
Evaluation(y_test, y_pred_voting)
Accuracy['Hard Voting'] = float(f"{accuracy_score(y_test, y_pred_voting) * 100:.2f}")
plot_confusion_matrix(y_test, y_pred_voting, 'Hard Voting')
pickle.dump(best_knn, open('Hard Voting.sav', 'wb'))

# creating the bar plots
models = list(TrainingTime.keys())
training_times = list(TrainingTime.values())
plt.bar(models, training_times, color='purple', label='Training Time')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Training Times for Each Model')
plt.legend()
plt.savefig('TrainingTime.png')
plt.show()

models_test = list(TestTime.keys())
testing_times = list(TestTime.values())
plt.bar(models_test, testing_times, color='lightblue', label='Testing Time')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title(' Testing Times for Each Model')
plt.legend()
plt.savefig('TestTime.png')
plt.show()

models_acc = list(Accuracy.keys())
model_accuracy = list(Accuracy.values())
plt.bar(models_acc, model_accuracy, color='pink')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy for Each Model')

# Display the accuracy values on top of each bar
for i, v in enumerate(model_accuracy):
    plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')
plt.savefig('Accuracy.png')
plt.show()

# ROC_CURVE_PLT:

pred_probs = [clf.predict_proba(X_test) for clf in [logisticRegression, best_knn, DecisionTree, RandomForest]]

fprs, tprs, thresholds = [], [], []

for pred_prob in pred_probs:
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:, 1], pos_label=1)
    fprs.append(fpr)
    tprs.append(tpr)
    thresholds.append(thresh)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

auc_scores = []

for pred_prob in pred_probs:
    if pred_prob.shape[1] == 2:
        auc_score = roc_auc_score(y_test, pred_prob[:, 1])
    else:
        auc_score = roc_auc_score(y_test, pred_prob, multi_class='ovr')
    auc_scores.append(auc_score)

plt.style.use('seaborn')

colors = ['Green', 'purple', 'blue', 'yellow']
classifiers = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']

for i in range(len(classifiers)):
    plt.plot(fprs[i], tprs[i], linestyle='--', color=colors[i],
             label='{} (AUC={:.3f})'.format(classifiers[i], auc_scores[i]))

p_fpr, p_tpr, _ = roc_curve(y_test, [0 for _ in range(len(y_test))], pos_label=1)
plt.plot(p_fpr, p_tpr, linestyle='--', color='red', label='Rondom Chance')

plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('ROC.png', dpi=300)
plt.show()
