#!/usr/bin/env python
# coding: utf-8

# # Sklearn_KMeans

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clusters.txt", header = None)
x1 = df[0]
x2 = df[1]

kmeans = KMeans(n_clusters=3, init='random', random_state=0).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)