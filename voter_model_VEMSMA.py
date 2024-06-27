import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import pickle


dataset = pd.read_csv("miRNA_SM_pairs.csv")

X = dataset.drop('Labels', axis=1)
y = dataset['Labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

clf1 = SVC(kernel='linear', probability=True, max_iter=30)
clf2 = RandomForestClassifier(n_estimators=50)
clf3 = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
clf4 = MLPClassifier(hidden_layer_sizes=(151,151,151), activation='relu', solver='adam', max_iter=3)
clf5 = GaussianNB()

voting_classifier_hard = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('knn', clf3), ('ann', clf4), ('gnb', clf5)], voting='hard')

voting_classifier_hard.fit(X_train, y_train)
y_pred_vch = voting_classifier_hard.predict(X_test)
cm = confusion_matrix(y_test, y_pred_vch)
print(cm)

voting_classifier_hard_Recall = recall_score(y_test, y_pred_vch)
voting_classifier_hard_Precision = precision_score(y_test, y_pred_vch)
voting_classifier_hard_f1 = f1_score(y_test, y_pred_vch)
voting_classifier_hard_accuracy = accuracy_score(y_test, y_pred_vch)

#5-fold-cross-validation

from statistics import stdev
score = cross_val_score(voting_classifier_hard, X_train, y_train, cv=5, scoring='recall')
voting_classifier_hard_cv_score = score.mean()
voting_classifier_hard_cv_stdev = stdev(score)
print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', voting_classifier_hard_cv_score)
print('Cross Validation Recall standard deviation: ', voting_classifier_hard_cv_stdev)

ndf = [(voting_classifier_hard_Recall, voting_classifier_hard_Precision, voting_classifier_hard_f1, voting_classifier_hard_accuracy, voting_classifier_hard_cv_score, voting_classifier_hard_cv_stdev)]

voting_classifier_hard_score = pd.DataFrame(data = ndf, columns=
                        ['Recall','Precision','F1 Score', 'Accuracy', 'Avg CV Recall', 'Standard Deviation of CV Recall'])
voting_classifier_hard_score.insert(0, 'Voting Classifier', 'Hard Voting')
voting_classifier_hard_score

#AUC-ROC
from sklearn.metrics import roc_auc_score
ROCAUCscore = roc_auc_score(y_test, y_pred_vch)
print(f"AUC-ROC Curve for Voting Classifier with soft voting: {ROCAUCscore:.4f}")

voting_classifier_soft = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('knn', clf3), ('ann', clf4), ('gnb', clf5)], voting='soft')

voting_classifier_soft.fit(X_train, y_train)
y_pred_vcs = voting_classifier_soft.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred_vcs)
print(cm2)

voting_classifier_soft_Recall = recall_score(y_test, y_pred_vcs)
voting_classifier_soft_Precision = precision_score(y_test, y_pred_vcs)
voting_classifier_soft_f1 = f1_score(y_test, y_pred_vcs)
voting_classifier_soft_accuracy = accuracy_score(y_test, y_pred_vcs)

#5-fold-cross-validation

from statistics import stdev
score = cross_val_score(voting_classifier_soft, X_train, y_train, cv=5, scoring='recall')
voting_classifier_soft_cv_score = score.mean()
voting_classifier_soft_cv_stdev = stdev(score)
print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', voting_classifier_soft_cv_score)
print('Cross Validation Recall standard deviation: ', voting_classifier_soft_cv_stdev)

ndf2 = [(voting_classifier_soft_Recall, voting_classifier_soft_Precision, voting_classifier_soft_f1, voting_classifier_soft_accuracy, voting_classifier_soft_cv_score, voting_classifier_soft_cv_stdev)]

voting_classifier_soft_score = pd.DataFrame(data = ndf2, columns=
                        ['Recall','Precision','F1 Score', 'Accuracy', 'Avg CV Recall', 'Standard Deviation of CV Recall'])
voting_classifier_soft_score.insert(0, 'Voting Classifier', 'Soft Voting')
voting_classifier_soft_score

#AUC-ROC
from sklearn.metrics import roc_auc_score
ROCAUCscore = roc_auc_score(y_test, y_pred_vcs)
print(f"AUC-ROC Curve for Voting Classifier with soft voting: {ROCAUCscore:.4f}")

# save the model to disk
filename = 'finalized_soft_voting_model.sav'
pickle.dump(voting_classifier_soft, open(filename, 'wb'))

# save the model to disk
filename = 'finalized_hard_voting_model.sav'
pickle.dump(voting_classifier_hard, open(filename, 'wb'))
