#!/usr/bin/env python
# coding: utf-8

# # Sklearn Logistic Regression

import numpy as np
from sklearn.linear_model import LogisticRegression

filename = 'classification.txt'
data = np.genfromtxt(filename, delimiter = ',')
X = data[:, :3]
Y = data[:, 4]
model = LogisticRegression().fit(X,Y)
w = model.intercept_
w = np.append(w,model.coef_)
print("Final Weights = \n", w)
print("Accuracy =", model.score(X,Y))