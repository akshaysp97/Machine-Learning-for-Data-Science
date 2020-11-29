#!/usr/bin/env python
# coding: utf-8

# # K-Means

import pandas as pd
import random
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


df = pd.read_csv("clusters.txt", header = None)
x1 = df[0]
x2 = df[1]


# Function to calculate differences
def dist(row,c):
    d = np.zeros(len(c)) #use the list of zeroes so that we can use argmax optimization here, instead of using empty list, can now index
    for group, centriod in enumerate(c): # each row of the random centriods
           d[group]= ((row[0] - centriod[0])**2 + (row[1] - centriod[1])**2)**.5
    return d.argmin()


#Compute K-means
def kmeans(k, data):

    c = np.array([[6,5],[0,2],[2,4]])
    #iterative process
    clusterAssignment = df.apply(lambda row: dist(row,c), axis =1)
    c = df.groupby(clusterAssignment).mean().values
    while (clusterAssignment != df.apply(lambda row: dist(row,c), axis =1)).any(): #find if the cluster assignment(even one) changed
        clusterAssignment = df.apply(lambda row: dist(row,c), axis =1)
        c = df.groupby(clusterAssignment).mean().values
    clusterAssignment = df.apply(lambda row: dist(row,c), axis =1)
    return clusterAssignment, c


kmeans(3,df)
a,c = kmeans(3,df)
centroid = df.groupby(a).mean().values
print("\n The centroids are:\n", centroid)

plt.figure()
colors = ['b', 'y', 'g']
for i,l in enumerate(a):
    plt.plot(x1[i],x2[i],color=colors[l],marker='o',ls='None')
    plt.plot(centroid[:,0], centroid[:,1], 'ro')
    plt.title("\nKmeans Clusters")
    plt.xlabel('y')
    plt.ylabel('x')
plt.show()