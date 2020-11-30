#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine

# # (b) Non Linear Data

import numpy as np
import cvxopt

data = np.loadtxt('nonlinsep.txt',dtype='float',delimiter=',')
X = data[:,0:2]
Y = data[:,2]
rows, cols = X.shape
kernel = np.empty((rows, rows))
for i in range(rows):
    for j in range(rows):
        kernel[i, j] = (1 + np.dot(X[i], X[j])) ** 2

Q = np.zeros((rows, rows))
for i in range(rows):
    for j in range(rows):
        Q[i,j] = kernel[i, j]

P = cvxopt.matrix(np.outer(Y,Y) * Q)
q = cvxopt.matrix(np.ones(rows) * -1)
A = cvxopt.matrix(Y, (1,rows))
b = cvxopt.matrix(0.0)
G = cvxopt.matrix(np.diag(np.ones(rows) * -1))
h = cvxopt.matrix(np.zeros(rows))

alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(1,rows)[0]
support_vector_indices = np.where(alphas>0.00001)[0]
support_vectors = X[support_vector_indices]

print("Support Vectors : \n", support_vectors)