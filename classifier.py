# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:32:46 2021

@author: ltroa
"""

import csv
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics._classification import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("train.csv")
exam = pd.read_csv("test.csv")

x = data.drop('target', axis=1)
x = x.drop('id', axis=1)

antiTuring = LabelEncoder()
y = pd.DataFrame(antiTuring.fit_transform(data['target']), columns = ['target'])

# print(x.describe())

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.3, stratify=y)
x_train, x_cv, y_train, y_cv = train_test_split(X_train, Y_train, stratify=Y_train, test_size=.3)

print(x_train.shape[0],x_cv.shape[0],X_test.shape[0])
x_train.head()
y_train.head()

train_class_distribution = pd.DataFrame(y_train).value_counts()
test_class_distribution = pd.DataFrame(Y_test).value_counts()
cv_class_distribution = pd.DataFrame(y_cv).value_counts()
my_colors = 'rgbkymc'
train_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

my_colors = 'rgbkymc'
test_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

my_colors = 'rgbkymc'
cv_class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()


# feature importance

from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

#pyplot.figure(figsize=(50,30))
#model = XGBClassifier()
#model.fit(x_train, y_train)
#plot_importance(model, max_num_features=25)

#pyplot.show()

a = [10 ** x for x in range(-5,1)]

logerrorCV=[]
for i in a:
    c = SGDClassifier(alpha = i,
                      penalty = 'l2',
                      loss='log',
                      random_state=42,
                      class_weight='balanced')
    c.fit(x_train, y_train.values.ravel())
    sc = CalibratedClassifierCV(c, method="sigmoid")
    sc.fit(x_train, y_train.values.ravel())
    pred_y = sc.predict_proba(x_cv)
    res = log_loss(y_cv, pred_y, labels=c.classes_, eps=1e-15)
    logerrorCV.append(res)
    print("a = ", i, "; log loss: ", res)
    
fig, ax = plt.subplots()
ax.plot(a, logerrorCV, c = 'g')
for i, txt in enumerate(np.round(logerrorCV, 3)):
    ax.annotate((a[i], np.round(txt, 3)), (a[i], logerrorCV[i]))
plt.grid()
plt.title("CV Error for alpha")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.show()



ba = np.argmin(logerrorCV)
c = SGDClassifier(alpha = a[ba], penalty = 'l2', loss = 'log', random_state=42, class_weight="balanced")
c.fit(x_train, y_train.values.ravel())
sc = CalibratedClassifierCV(c, method = "sigmoid")
sc.fit(x_train, y_train.values.ravel())

pred_y = sc.predict_proba(x_train)
print('Best alpha = ', a[ba], '; train log loss: ',
      log_loss(y_train, pred_y, labels=c.classes_, eps=1e-15))

pred_y = sc.predict_proba(x_cv)
print('Best alpha = ', a[ba], '; cv log loss: ',
      log_loss(y_cv, pred_y, labels=c.classes_, eps=1e-15))

pred_y = sc.predict_proba(X_test)
print('Best alpha = ', a[ba], '; test log loss: ',
      log_loss(Y_test, pred_y, labels=c.classes_, eps=1e-15))


exam = exam.drop('id', axis=1)


c = SGDClassifier(alpha = .1, penalty='l2', loss='log', random_state=42, )
c.fit(x_train, y_train.values.ravel())
sc = CalibratedClassifierCV(c, method = "sigmoid")
sc.fit(x_train, y_train.values.ravel())

res = sc.predict_proba(exam)
outpA = []
id = 200000

for i in range(len(res)):
    pred = res[i]
    inA = [id, pred[0], pred[1], pred[2], pred[3],
           pred[4], pred[5], pred[6], pred[7], pred[8]]
    outpA.append(inA)
    id += 1

turner = pd.DataFrame(outpA, 
                      columns=['id', 'Class_1', 'Class_2', 'Class_3',
                               'Class_4', 'Class_5', 'Class_6', 'Class_7', 
                               'Class_8', 'Class_9'])
turner.to_csv(path_or_buf='submission.csv', index = False)
turner









# RANDOM FORREST, initial accuracy: .35

# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X_train, Y_train)

# y_pred = rf.predict(X_test)
# print("Accuracy: ", metrics.accuracy_score(Y_test, y_pred))


# DECISION TREE, initial accuracy: .23

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix

# dtree_model = DecisionTreeClassifier().fit(X_train, Y_train)
# dtree_predictions = dtree_model.predict(X_test)
  
# print("Accuracy: ", metrics.accuracy_score(Y_test, dtree_predictions))

# SVM, initial accuracy: n/a
  
# training a linear SVM classifier
# from sklearn.svm import SVC
# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, Y_train)
# svm_predictions = svm_model_linear.predict(X_test)
  
# model accuracy for X_test  
# accuracy = svm_model_linear.score(X_test, Y_test)
# print("Accuracy: ", accuracy)

#KNN, initial accuracy: .27

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, Y_train)
  
# accuracy = knn.score(X_test, Y_test)
# print(accuracy)