#!/usr/bin/env python
# coding: utf-8

# # Hidden Markov Model

import numpy as np
import pandas as pd

# Setting up the environment
S = np.arange(1,11) # state space
K = len(S) # Number of states

Y = np.array([8,6,4,6,5,4,5,5,7,9]) # Observations
O = np.arange(0,12) # all possible observations

T = len(Y) # path length
N = 12 # Size of the observation space

Pi = np.array([.1 for element in range(K)]) # Intial probabilities for each event at time = 0
# For loop so that each event had the same probabilty of occuring before any observation is made

# Constructing transition matrix
# Matrix shows probability of event i transitioning to event j at A[i-1,j-1] -1 because events are one-based index and matrix is 0 based DIFFICULTY
A = np.zeros((K,K)) # Transition Matrix

for i in S: # Correcting
    for j in S:
        if (i==1 and j==2) or (i==10 and j==9): # Edge case
            A[i-1,j-1] = 1
        elif j==i+1 or j==i-1: # Else cases
            A[i-1,j-1] = .5

            
pd.DataFrame(A, index=S, columns =S)


# Emission Matrix 
# The probability of observing j given event  i at B[i-1,j-1]
B = np.zeros((K,N))

for i in S:
    for j in O:
        if j == i-1 or i==j or j == i+1:
            B[i-1,j] = 1/3
            
pd.DataFrame(B,index=S,columns=O)

def viterbi(O,S,Pi,Y,A,B):
    T = len(Y) # Number of time stamps
    K = len(S) # Number of events 
    X = np.zeros(T) # Initilzing the path with 0s
    Z = np.zeros(T) # Used to store indexes
    T1 = np.zeros((K, T)) # Probabilities of the current most likely path that leads to i 
    T2 = np.zeros((K, T)) # Stores the indexes so that they can go backward from t to t-1
    for i in range(K):  # iterate over state space
        T1[i, 0] = Pi[i] * B[i, Y[0]] # generating the probabilities
        T2[i, 0] = 0
    
    for j in range(1, T): # start from second observation
        for i in range(K): #index of row over tables, row by row
            T1[i,j] = max([T1[k,j-1]*A[k,i]*B[i,Y[j]] for k in range(K)]) # Gives the most likely value for j
            T2[i,j] = np.array([T1[k,j-1]*A[k,i]*B[i,Y[j]] for k in range(K)]).argmax() # Takes the index of the most likely path
    
    Z[T-1] = T1[:,T-1].argmax() # Who had the highest probability at each time stamp
    X[T-1] = S[int(Z[T-1])]
    for j in list(range(1,T))[::-1]: # loop back from T to t=2, going backwards
        Z[j-1] = T2[int(Z[j]), j]
        X[j-1] = S[int(Z[j-1])]
    return X
        
viterbi(O,S,Pi,Y,A,B)