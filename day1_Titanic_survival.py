# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv('data//train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels

y=df['Survived']
X=df.iloc[:,1:]

# %%
# train-test split
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=1919810)
print(f'X_train: {np.shape(X_train)}')
print(f'y_train: {np.shape(y_train)}')
print(f'X_test: {np.shape(X_test)}')
print(f'y_test: {np.shape(y_test)}')

# %%
# build model
# build three classification models
# SVM, KNN, Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

models=dict()
# models['LinearRegression']=LinearRegression(positive=True)
models['SVM']=SVC(kernel='linear')   # kernel specifies the kernel type, 'rbf' is the radial basis function
models['KNeighbor']=KNeighborsClassifier(n_neighbors=25,weights='distance')  # n_neighbors specifies the number of neighbors
models['RandomForest']=RandomForestClassifier(n_estimators=200,max_depth=9,max_features=0.60,random_state=114514)  # n_estimators specifies the number of trees in the forest

# %%
# predict and evaluate
# define functions to evaluate the models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np


def plot_cm(model,y_true,y_pred,name=None):
    _, ax=plt.subplots()
    if name:
        ax.set_title(name)
    cm=confusion_matrix(y_true,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()


def plot_cm_ratio(model,y_true,y_pred,name=None):
    _, ax=plt.subplots()
    if name:
        ax.set_title(name)
    cm=confusion_matrix(y_true,y_pred)
    cm_ratio=np.zeros(cm.shape)
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm_ratio[i,j]=cm[i,j]/cm[i].sum()
    disp=ConfusionMatrixDisplay(confusion_matrix=cm_ratio,display_labels=model.classes_)
    disp.plot(ax=ax)
    plt.show()

def model_perf(model, y_true, y_pred, name=None):
    if name:
        print(f'For model {name}: \n')
    cm=confusion_matrix(y_true, y_pred)
    for i in range(len(model.classes_)):
        tp=cm[i,i]
        fp=cm[:,i].sum()-cm[i,i]
        fn=cm[i,:].sum()-cm[i,i]
        tn=cm.sum()-tp-fp-fn
        tpr=tp/(tp+fn)
        fpr=fp/(tn+fp)
        acc=(tp+tn)/cm.sum()
        print(f'For class {model.classes_[i]}: \n TPR is {tpr}; \n FPR is {fpr}; \n ACC is {acc}. \n')
    return None


def ovo_eval(model,name=None):
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    plot_cm(model,y_test,prediction,name)
    plot_cm_ratio(model,y_test,prediction,name)
    model_perf(model,y_test,prediction,name)
    score=model.score(X_test,y_test)
    print(f'Overall Accuracy: {score}')
    return score


record=dict()
for name,model in models.items():
    record[name]=ovo_eval(model,name)

from pprint import pprint
pprint(record)
# %%
