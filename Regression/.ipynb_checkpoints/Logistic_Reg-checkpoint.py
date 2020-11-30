#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

import numpy as np

def Logis_Regression(X,Y,w):
    for i in range(7000):
        yS = np.multiply(np.dot(X, w.T), Y)                 #S=w.T*x 
        Ein = 1/(1+np.exp(yS))                              
        Ein = np.multiply(Ein, np.multiply(X, Y))           
        Ein = np.sum(Ein, axis = 0)                         #Ein(w) = sum((y*x)*(1/1+e^y*w.T*x))
        delEin = -(1/len(X))*Ein                            #delEin(w) = -1/N*Ein(w)
        w -= 0.1 * delEin                                   #w = w-η*delEin(w)
    return w


def pred_accuracy(X,Y,w):
    S = np.dot(X, w.T)
    err = np.ones((len(X),1));
    err[S < 0] = -1
    err = np.multiply(err, Y)
    err[err < 0] = 0
    summ = np.sum(err, axis=0)[0]
    acc = summ / (float(len(X)))
    return acc


if __name__ == '__main__':
    filename = 'classification.txt'
    data = np.genfromtxt(filename, delimiter = ",")
    X = data[:, :3]                                      #Coordinates of X
    m, n = np.shape(X)
    ones = np.ones((m,1))
    X = np.concatenate((ones, X), axis = 1)              #Appending x0(i) = 1 to X 
    w = np.random.rand(1,n+1)                            #Choose w at random
    Y = data[:,4]                                        #Classification label
    Y = Y[:, np.newaxis]                                 #Transformation to column vector 
    weights = Logis_Regression(X,Y,w)
    accuracy = pred_accuracy(X,Y,weights)
    print("Final Weights = ", w)
    print("Accuracy = ", accuracy)