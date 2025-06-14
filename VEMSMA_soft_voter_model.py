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
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB


dataset = pd.read_csv("input_data_voting_model.csv")

X = dataset.drop('Labels', axis=1)
y = dataset['Labels']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


clf5 = GaussianNB()
clf4 = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50, warm_start=True)
clf3 = KNeighborsClassifier(n_neighbors=20)
clf2 = RandomForestClassifier(n_estimators=10, max_depth=10)
clf1 = SVC(kernel='linear', probability=True,max_iter=60)


voting_classifier_soft = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('knn', clf3), ('ann', clf4), ('gnb', clf5)], voting='soft')


voting_classifier_soft.fit(X_train, y_train)

y_pred_vcs = voting_classifier_soft.predict(X_test)


cm = confusion_matrix(y_test, y_pred_vcs)
print(cm)

voting_classifier_soft_Recall = recall_score(y_test, y_pred_vcs)
voting_classifier_soft_Precision = precision_score(y_test, y_pred_vcs)
voting_classifier_soft_f1 = f1_score(y_test, y_pred_vcs)
voting_classifier_soft_accuracy = accuracy_score(y_test, y_pred_vcs)
voting_classifier_soft_auc= roc_auc_score(y_test, y_pred_vcs)


from statistics import stdev
score = cross_val_score(voting_classifier_soft, X_train, y_train, cv=5, scoring='accuracy')
voting_classifier_soft_cv_score1 = score.mean()
voting_classifier_soft_cv_stdev1 = stdev(score)


###############################################################################
score = cross_val_score(voting_classifier_soft, X_train, y_train, cv=5, scoring='recall')
voting_classifier_soft_cv_score2 = score.mean()
voting_classifier_soft_cv_stdev2 = stdev(score)

###############################################################################
score = cross_val_score(voting_classifier_soft, X_train, y_train, cv=5, scoring='precision')
voting_classifier_soft_cv_score3 = score.mean()
voting_classifier_soft_cv_stdev3 = stdev(score)

###############################################################################
score = cross_val_score(voting_classifier_soft, X_train, y_train, cv=5, scoring='f1')
voting_classifier_soft_cv_score4 = score.mean()
voting_classifier_soft_cv_stdev5 = stdev(score)
###############################################################################

score = cross_val_score(voting_classifier_soft, X_train, y_train, cv=5, scoring='roc_auc')
voting_classifier_soft_cv_score5 = score.mean()
voting_classifier_soft_cv_stdev5 = stdev(score)
###############################################################################



#print('Cross Validation Accuracy scores are: {}'.format(score))
print('Average Cross Validation Accuracy score: ', voting_classifier_soft_cv_score1)
#print('Cross Validation Accuracy standard deviation: ', voting_classifier_soft_cv_stdev1)



#print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', voting_classifier_soft_cv_score2)
#print('Cross Validation Recall standard deviation: ', voting_classifier_soft_cv_stdev2)


#print('Cross Validation Precision scores are: {}'.format(score))
print('Average Cross Validation Precision score: ', voting_classifier_soft_cv_score3)
#print('Cross Validation Precision standard deviation: ', voting_classifier_soft_cv_stdev3)


#print('Cross Validation F1 scores are: {}'.format(score))
print('Average Cross Validation F1 score: ', voting_classifier_soft_cv_score4)
#print('Cross Validation F1 standard deviation: ', voting_classifier_soft_cv_stdev4)


#print('Cross Validation AUC scores are: {}'.format(score))
print('Average Cross Validation AUC: ', voting_classifier_soft_cv_score5)
#print('Cross Validation F1 standard deviation: ', voting_classifier_soft_cv_stdev5)



filename = 'finalized_soft_voting_model.sav'
pickle.dump(voting_classifier_soft, open(filename, 'wb'))



