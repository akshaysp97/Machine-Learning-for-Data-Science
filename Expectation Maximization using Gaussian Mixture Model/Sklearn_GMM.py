#!/usr/bin/env python
# coding: utf-8

# # Sklearn_GMM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

training_data = np.genfromtxt('clusters.txt', delimiter=',')
gmm = GaussianMixture(n_components=3,random_state=1)
gmm.fit(training_data)
y = gmm.predict(training_data)
labels = gmm.predict(training_data)
mu = gmm.means_
cov = gmm.covariances_


print("The means are:\n", mu)
print("\n The amplitudes are:", gmm.weights_)
print("\n The covariances are:\n", cov)


x = np.linspace(-5, 10)
y = np.linspace(-5, 10)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)
CS = plt.contourf(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),levels=np.logspace(0, 2, 10))
plt.colorbar(CS, extend='both')
plt.scatter(training_data[:,0],training_data[:,1], c=labels, s=20, cmap='inferno')
plt.title("Gaussian mixture")
plt.xlabel('x')
plt.ylabel('y')
plt.show()