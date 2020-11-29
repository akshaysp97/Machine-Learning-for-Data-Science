#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt("fastmap-data.txt")
data = np.concatenate([data, data[:,[1,0,2]]]) #had to complete the table 

words = open("fastmap-wordlist.txt").read().split()
objects = np.arange(1,11) #shifting up generated numeric values for the words

# Basic distance function, refers to data matrix, just looks up distances
def dist(a,b):
    if a==b:
        return 0
    return data[(data[:,:2]==[a,b]).all(1),2][0]

dist(1,2)

def pivot(objects, dist):
    Oa = np.random.choice(objects) # random intiate
    print(f"Initial Random Choice: {Oa}")
    i = np.array([dist(Oa, b) for b in objects]).argmax() # index of maximum distances in the objects list
    Ob = objects[i]

    i = np.array([dist(Ob, a) for a in objects]).argmax() # finding index of furthest point
    Oa = objects[i]
    
    return int(Oa), int(Ob)

pivot(objects, dist) 

X = np.zeros((len(objects),2))
pivotArray = np.zeros((2,2))
col = -1

def FastMap(k, D, O):
    global col, X,pivotArray
    if k <= 0:
        return X
    else:
        col += 1
    a,b = pivot(O,D)
    pivotArray[0, col] = a
    pivotArray[1, col] = b
    if D(a,b) == 0:
        X[:,col] = 0
        return X
    else: 
        x = np.array([(D(a,i)**2 + D(a,b)**2 - D(i,b)**2)/ (2*D(a,b)) for i in O])
        X[:,col] = x
        # Update distances
        dist_new = lambda i,j: np.sqrt(D(i,j)**2 - (x[i-1] - x[j-1])**2)
        return FastMap(k-1, dist_new, O)

FastMap(2, dist, objects)

plt.scatter(X[:,0], X[:,1])
for i, word in zip(objects, words):
    plt.text(X[i-1,0], X[i-1,1], word)
    #print(i, word)