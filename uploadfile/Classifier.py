import os
import pandas as pd
import numpy as np
import time



# "notselectfeature", "chi2", "pca", "pearson","likepaper"
chose_name_file = "notselectfeature"

data  = pd.read_csv('feets_2000data_49feature_' + chose_name_file + '_MinMaxScaler.csv')


Y_new = data["Class"]
value, count = np.unique(Y_new, return_counts=True)
value = value.astype("int")
    
X_new = data.drop(["Class"],axis=1)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)

detail  = pd.read_csv('detail_Type.csv')
Code_Group = detail["Code_Group"]
Group = detail["Group"]
dictt = dict()
co = 0
for i in Code_Group:
    dictt[i] = Group[co]
    co += 1

list_scores = []
first = True
total_cm = []
from sklearn.metrics import confusion_matrix
for train_index, test_index in rskf.split(X_new, Y_new):
    
    
    X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
    Y_train, Y_test = Y_new.iloc[train_index], Y_new.iloc[test_index]
    
   
    clf = RandomForestClassifier().fit(X_train,Y_train) 
    # clf = xgb.XGBClassifier().fit(X_train,Y_train)
    # clf = svm.SVC().fit(X_train,Y_train)
    # clf = KNeighborsClassifier(n_neighbors = 5).fit(X_train,Y_train)

    scores = clf.score(X_test, Y_test)
    Y_pre = clf.predict(X_test)
    list_scores.append(scores)
    print(scores)
    
    if(first):
        total_cm = confusion_matrix(Y_test, Y_pre,labels=value)
        first = False
    else:
        cm = confusion_matrix(Y_test, Y_pre,labels=value)
        total_cm += cm
    
    

import matplotlib.pyplot as plt
import statistics 
import seaborn as sn

sum_score = sum(list_scores)
average = sum_score/(len(list_scores))
average = average*100
print("Average_of_score: ",average)


cm = total_cm
recall = np.diag(cm) / np.sum(cm, axis = 1)
recall = pd.DataFrame(recall)
recall = recall.replace(np.nan,0)
recall = np.array(recall)

precision = np.diag(cm) / np.sum(cm, axis = 0)
precision = pd.DataFrame(precision)
precision = precision.replace(np.nan,0)
precision = np.array(precision)

AVG_precision = sum(precision)/len(precision)
AVG_precision = AVG_precision*100
AVG_precision = AVG_precision[0]

AVG_recall = sum(recall)/len(recall)
AVG_recall = AVG_recall*100
AVG_recall = AVG_recall[0]

AVG_f1 = 2 * ((AVG_precision * AVG_recall)/(AVG_precision + AVG_recall))

print("AVG_precision",AVG_precision)
print("AVG_recall",AVG_recall)
print("AVG_f1",AVG_f1)


cm_with_precision = cm/np.sum(cm, axis = 0)
cm_with_precision = np.nan_to_num(cm_with_precision)

cm_with_recall = (cm.T/np.sum(cm, axis = 1)).T
cm_with_recall = np.nan_to_num(cm_with_recall)

df_cm = pd.DataFrame(cm, index = value,
                  columns = value)
df_cm_precision = pd.DataFrame(cm_with_precision, index = value,
                  columns = value)
df_cm_recall = pd.DataFrame(cm_with_recall, index = value,
                  columns = value)
plt.figure(figsize = (8,5))

sn.heatmap(df_cm, annot=True)
sn.set(font_scale=0.8)
plt.xlabel('Predicted',size = 10)
plt.ylabel('True',size = 10)
plt.title('confusion matrix',size = 10)

plt.figure(figsize = (8,5))
sn.heatmap(df_cm_precision, annot=True)
sn.set(font_scale=0.8)
plt.xlabel('Predicted',size = 10)
plt.ylabel('True',size = 10)
plt.title('precision',size = 10)

plt.figure(figsize = (8,5))
sn.heatmap(df_cm_recall, annot=True)
sn.set(font_scale=0.8)
plt.xlabel('Predicted',size = 10)
plt.ylabel('True',size = 10)
plt.title('recall',size = 10)