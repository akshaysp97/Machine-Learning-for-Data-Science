#!/usr/bin/env python
# coding: utf-8

# # Sklearn Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression

filename = 'linear-regression.txt'
data = np.genfromtxt(filename, delimiter = ',')
X = data[:, :2]
Y = data[:, 2]
model = LinearRegression().fit(X,Y)
w = model.intercept_
w = np.append(w,model.coef_)
print("Final Weights = \n", w)