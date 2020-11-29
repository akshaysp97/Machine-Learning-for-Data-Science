#!/usr/bin/env python
# coding: utf-8

# # Sklearn Implementation of Decision Tree

from sklearn import tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
 
train_data = pd.read_csv('trainData.csv')

encoding = preprocessing.LabelEncoder()
train_data = train_data.apply(encoding.fit_transform)

X = train_data.drop(train_data.columns[[6]], axis=1)
Y = train_data[train_data.columns[[6]]]

clf = tree.DecisionTreeClassifier(criterion = 'entropy').fit(X,Y)

test = [[2,0,0,0,0,0]]
print(clf.predict(test))

