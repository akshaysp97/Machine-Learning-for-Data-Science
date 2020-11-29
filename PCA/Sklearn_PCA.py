#!/usr/bin/env python
# coding: utf-8

# # PCA Sklearn

import numpy as np
from sklearn.decomposition import PCA

data = np.genfromtxt('pca-data.txt', delimiter = "\t")
pca=PCA(n_components=2)
pca.fit(data)
print(pca.components_.T)