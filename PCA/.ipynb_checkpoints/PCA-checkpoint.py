#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis(PCA)


import numpy as np

def mean(data):
    mu = np.mean(data, axis = 0)
    X = np.array(data - mu)
    return X

def PCA(data,k):
    cov = np.cov(data.T)
    eigenvec, eigenval, vh = np.linalg.svd(cov)
    sort_eigval = np.argsort(-eigenval)
    eigenvec = eigenvec[:, sort_eigval]
    sort_eigvec = eigenvec[:, :k]
    return sort_eigvec

if __name__ == '__main__':
    input_data = np.genfromtxt('pca-data.txt', delimiter='\t')
    data = np.array(input_data)
    k = 2
    norm_data = mean(data)
    principal_comp = PCA(norm_data,k)
    print(principal_comp)