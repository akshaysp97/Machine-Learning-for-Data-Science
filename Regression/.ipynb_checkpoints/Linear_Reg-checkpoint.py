#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

import numpy as np

def Lin_Regression(X,Z):
    m,n = np.shape(X)
    ones = np.ones((m,1))
    X = np.concatenate((ones, X), axis = 1)      #Appending x0(i) = 1 to X matrix
    D = X.T
    dd = np.linalg.inv(D*D.T)*D                  #w = (DD.T)^-1*D*Y
    w = dd*Z
    return w


if __name__ == '__main__':
    filename = 'linear-regression.txt'
    data = np.genfromtxt(filename, delimiter = ",")
    X = np.mat(data[:, :2])           #Independent varibales
    Z = data[:,2]                     #Dependent variable
    Z = np.mat(Z[:, np.newaxis])  
    weights = Lin_Regression(X,Z)
    print("Final Weights = \n", weights)