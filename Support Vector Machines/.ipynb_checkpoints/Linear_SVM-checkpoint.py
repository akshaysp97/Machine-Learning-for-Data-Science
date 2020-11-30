#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine

# # (a) Linearly Seperable Data

import numpy as np
from cvxopt import solvers, matrix
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt("linsep.txt", delimiter=",")
X = data[:,:2]
y = data[:,-1]

x0 = X[:,0]
x1 = X[:,1]

plt.scatter(x0,x1, c=y)
plt.colorbar()

d = 2
Q = np.identity(2)
Q = np.concatenate([np.zeros(d).reshape(1,-1), Q])
Q = np.concatenate([np.zeros(d+1).reshape(-1,1), Q], axis=1)
Q #identity matrix plus an extra row and column (first of each)

p = np.zeros(d+1)
p #required for QP

A = np.multiply(np.concatenate([y.reshape(-1,1)]*d, axis=1),X)
A = np.concatenate([y.reshape(-1,1), A], axis=1)               #first column is Y, remaining are x*y

C = np.ones((X.shape[0],1))

Q = matrix(Q)
p = matrix(p)
A = matrix(A)
C = matrix(C)
sol = solvers.qp(Q,p,-A,-C,)

print(sol['x'])

b, w0, w1 = np.array(sol['x'])

x = np.linspace(x0.min(),x0.max(), 2) #used to generate the line that we will go onto plot using the bias and weights
line = lambda x: (-w0/w1)*x - b/w1
plt.scatter(x0,x1, c=y)
plt.plot(x,line(x), "r")

def dist(x0,x1, b, w0, w1): #returns distance to the line
    return abs(w0*x0 + w1*x1 + b)/np.sqrt(w0**2 + w1**2)

b, w0, w1 = np.array(sol['x'])
distances_to_line = np.apply_along_axis(lambda row: dist(row[0], row[1], b, w0, w1), 1, X).flatten() #we have to flatten in order to have correct indexes so we can find SV

min_distance = distances_to_line.min()
support_vectors = X[distances_to_line<=min_distance+.000001,:] #for rounding error and indexing
plt.scatter(x0,x1, c=y)
plt.plot(x,line(x), "r")
plt.scatter(support_vectors[:,0], support_vectors[:,1], c="b")

print(support_vectors)