#!/usr/bin/env python
# coding: utf-8

# # Gaussian Mixture Model(GMM)

import numpy as np
import math
import random
import matplotlib.pyplot as plt

#Function to calculate the conditional probability of Xi from the Multivariate Normal Distribution
def Multivariate(Xi, mu, cov, d):
    factor = 1/((2*np.pi)**(-d/2))
    invcov = abs(np.linalg.det(cov))**(-1/2)
    mat_mul1 = np.dot((Xi-mu).T, np.linalg.inv(cov))
    mat_mul2 = np.dot(mat_mul1, (Xi-mu))
    pdf = factor * invcov * np.exp(-1/2 * mat_mul2)
    return pdf

#Function to calculate the maximum likelihood
def max_likelihood():
    new_likelihood = 0
    for i in range(N):
        val = 0
        for j in range(K):
            val += amplitude[j] * Multivariate(data[i].T, mean[j].T, covariance[j], 2)
        new_likelihood += np.log(val)

    return new_likelihood

# Expectation Step to calculate Ric
def Estep():
    denominator = np.zeros(N)
    for i in range(N):
        numerator = np.zeros(K)  
        for k in range(K):
            numerator[k] = float(amplitude[k]) * Multivariate(data[i].T, mean[k].T, covariance[k], 2)
            denominator[i] += numerator[k]
        for k in range(K):
            r[k][i] = numerator[k]/denominator[i]


# Maximization Step to calculate model parameters
def Mstep():
    for k in range(K):
        amplitude[k] = np.sum(r[k]) / N
        
        numerator = np.zeros(mean.shape[1])
        denominator = np.sum(r[k])
        for i in range(N):
            numerator += r[k][i]* data[i]
        mean[k] = numerator / denominator

        summation = np.zeros([data.shape[1], data.shape[1]])
        for i in range(N):
            if data[i].ndim == 1:
                data_temp = data[i].reshape(data.shape[1], 1)
                mu_temp = mean[k].reshape(mean.shape[1], 1)
                diff_temp = data_temp - mu_temp
                summation += r[k][i] * np.dot(diff_temp, diff_temp.T)
            else:
                summation += r[k][i] * np.dot(data[i]-mean[i], (data[i]-mean[i]).T)
        
        covariance[k] = summation / denominator


if __name__ == '__main__':
    training_data = np.genfromtxt('clusters.txt', delimiter=',')
    K = 3  
    amplitude = np.array([55/150,75/150,20/150]) 
    covariance = np.array([[[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]]])
    mean = np.array([[5.620165735, 5.026226344], [-0.974765718, -0.684193041], [3.083182557, 1.776213738]])
    data = np.array(training_data)
    N = len(data)
    r = np.zeros([K,len(data)])
    
    #Set up recursive procedure until convergence
    threshold = 1e-3
    likelihood = None
    new_lld = max_likelihood()
    while True:
        likelihood = new_lld
        Estep()
        Mstep()
        new_lld = max_likelihood()
        
        if (new_lld - likelihood < threshold):
            break
            
    print("The means are:\n", mean)
    print("\nThe amplitudes are:", amplitude)
    print("\nThe covariances are:\n", covariance)