#!/usr/bin/env python
# coding: utf-8

# # Perceptron Learning

import numpy as np
import pandas as pd

df = pd.read_csv("classification.txt", header =None)
data = df.drop(columns=4)
data["const"] = 1
data = data[["const",0,1,2,3]]
X = data.drop(columns=3).values # want nd array like w so we can do dot products
y = data[3].values # set the labels, use values because we want an array
w = np.zeros(X.shape[1]) #initializing the weight vector

r = .1 # learning rate
iterations = 10000

for counts in range(iterations):
    i = np.random.randint(0,X.shape[0]) # get a random index
    x = X[i] # select a particular row based on i
    d = y[i]
    pred = 1 if (np.dot(x,w) > 0) else -1
    w =  w + r * (d-pred)/2 * x #updating w # the /2 makes the (d-pred) either 1 or -1 

predictions = np.ones(X.shape[0]) #intilizing every prediction as 1
predictions[np.dot(X,w) <= 0] = -1 #going back and changing everything that is not labeled as 1
accuracy = (predictions == y).mean()

print("Final Weights = ", w)
print("Accuracy = ", accuracy)