#!/usr/bin/env python
# coding: utf-8

# # Decision Tree

import numpy as np 
import pandas as pd 
import math
from pprint import pprint
import pydot

#Read training data
training_dataset = pd.read_csv('trainData.csv', names=['Occupied','Price','Music','Location','VIP','Favorite Beer','Enjoy'], header=0)  

#Function to calculate the Entropy of data
def entropy(label):
    values,count = np.unique(label,return_counts = True)
    entropy = np.sum([(-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)) for i in range(len(values))])
    return entropy

#Function to calculate Weighted Entropy
def weighted_entropy(data,feature,label="Enjoy"):
    values,count = np.unique(data[feature],return_counts = True)
    wtd_entropy = np.sum([(count[i]/np.sum(count))*entropy(data.where(data[feature]==values[i]).dropna()[label]) for i in range(len(values))])
    return wtd_entropy

#Function to calculate Information Gain
def Information_gain(data,feature,label="Enjoy"):
    total_entropy = entropy(data[label])
    split_entropy = weighted_entropy(data,feature,label="Enjoy")
    inf_gain = total_entropy - split_entropy
    return inf_gain

#Function to select the feature which best splits the dataset
def best_split(df, target_name = "Enjoy"):
    IG = [(Information_gain(df, att), att) for att in df.columns if att != target_name] #calculates info gain for attributes and stores in list
    IG.sort(reverse = True)
    best_att = IG[0][1] #taking name of best attribute
    return best_att

#Function to build decision tree based on ID3 algorithm
def ID3(df, target_attribute_name = "Enjoy"):
    '''
    takes a df as arguement in which last column is the target variable
    '''
    
    if len(df[target_attribute_name].unique()) == 1: # y is pure
        return df[target_attribute_name].unique()[0]
    elif df.drop(target_attribute_name, axis =1).drop_duplicates(keep=False).shape[0]==0:
        return df[target_attribute_name].value_counts().idxmax()
    else:
        best_att = best_split(df)
        nodes = []
        for cat in df[best_att].unique(): #catagories in the attribute
            nodes.append((best_att, cat, ID3(df[df[best_att]==cat]))) #each catagory get its own subtree, recursion
        return nodes

#Function to print the decision tree
def printTree(data, tree):
    for branch in tree:
        pprint(branch)
        print()
        #every nested list is a subtree, subtree to the node occupied etc
        #attribute and value determines node, subtree is thrird part, unless its a leaf, string yes or no
        
tree = ID3(training_dataset)
printTree(training_dataset,tree)

#Function to predict new/unseen instance
def predict(tree, s):
    '''
    tree- decision tree created by make_tree
    s - sample on which to predict(dictionary)
    '''
    while True:
        if not isinstance(tree, list): # if tree is not a list, you've reached a leaf
            return tree
        att = tree[0][0] # first term of first subtree, node is an attribute followed by another node or a decision
        for branch in tree:
            if s[att] == branch[1]: #if the sample matches the value for the relavant attribute, then you take that branch and start searching again
                tree = branch[2]
                break
# find out what attribute you are looking at, then go through all of the nodes

#Test case
sample = {'Occupied': 'Moderate', 'Price': 'Cheap', 'Music': 'Loud', 'Location': 'City-Center', 'VIP': 'No', 'FavoriteBeer': 'No','Enjoy':''}
test = pd.DataFrame([sample])
test

#Prediction for test case
enjoy = predict(tree,sample)
print(enjoy)

#Test case with prediction
test['Enjoy'] = enjoy
test

