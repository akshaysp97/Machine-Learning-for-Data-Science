#!/usr/bin/env python
# coding: utf-8

# # Pocket Algorithm

import numpy as np
import pandas as pd

df = pd.read_csv("classification.txt", header =None)
data = df.drop(columns=3)

# Pocket Training
data["const"] = 1
data = data[["const",0,1,2,4]]
X = data.drop(columns=4).values # want nd array like w so we can do dot products
y = data[4].values
w = np.zeros(X.shape[1]) #initializing the weight vector

r = .1 # learning rate
iterations = 7000
pocket = w #intialize pocket
pocket_error = 2 #intialize with something greater than one

error_rates = []
for counts in range(iterations):
    i = np.random.randint(0,X.shape[0]) # get a random index
    x = X[i] # select a particular row based on i
    d = y[i]
    pred = 1 if (np.dot(x,w) > 0) else -1
    w =  w + r * (d-pred) * x #updating w
    #test pocket
    predictions = np.ones(X.shape[0])
    predictions[np.dot(X,w) <= 0] = -1
    loss = (predictions != y).mean() #how good the weights were in their current iteration
    error_rates.append(loss) #keep tracking of loss for each iteration
    if loss < pocket_error:
        pocket = w
        pocket_error = loss
    
predictions = np.ones(X.shape[0])
predictions[np.dot(X,pocket) <= 0] = -1   
(predictions==y).mean(), pocket

import matplotlib.pyplot as plt
misclass = np.array(error_rates) * len(X)
plt.figure(figsize = (15,5))
plt.plot(range(1,iterations+1), misclass)