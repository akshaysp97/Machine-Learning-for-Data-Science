#!/usr/bin/env python
# coding: utf-8

# # Neural Network

import numpy as np

def get_image_data(train_data):
    with open(train_data, 'rb') as f:
        f.readline()   
        f.readline()   
        x, y = f.readline().split()  
        x = int(x)
        y = int(y)
        factor = int(f.readline().strip())

        data = []
        for _ in range(x * y):
            data.append(f.read(1)[0] / factor)

        return data


def read_training_data():
    training_data = []
    labels = []
    with open('downgesture_train.list') as f:
        for image in f.readlines():
            image = image.strip()
            training_data.append(get_image_data(image))
            if 'down' in image:
                labels.append([1,])
            else:
                labels.append([0,])  

        return training_data, labels


def build_layers(training_data, label):
    dim = 0
    input_numbers = 0
    input_dimensions = 0
    output_numbers = 0
    output_dimensions = 0
    layer = []
    dim = training_data.ndim;
    if dim != 0:
        input_numbers, input_dimensions = training_data.shape
    else:
        pass
    
    dim = label.ndim;
    if dim !=0:
        if dim == 1:
            output_numbers = label.shape[0]
            output_dimensions = 1;
        else:
            output_numbers, output_dimensions = label.shape
    else:
        pass

    layer.append(input_dimensions+1) 

    for i in hidden_layer:
        layer.append(i)

    layer.append(output_dimensions) 
    layer = np.array(layer)

    return layer


def weight_initialize(layer):
    weights = []
    for l in range(1, len(layer)):
            weights.append(((upper_bound)-(lower_bound))*np.random.normal(size=(layer[l-1], layer[l]))+(lower_bound))                     
            np.random.random
    
    return weights


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def feed_forward(input_data,wt):
    X = [np.concatenate((np.ones(1).T, np.array(input_data)), axis=0)] 
    W = wt
    wijxi = []
    xj = []

    for l in range(0, len(W)):
        wijxi = np.dot(X[l], W[l])
        xj = sigmoid(wijxi)
        if l < len(W)-1:
            xj[0] = 1 
        X.append(xj)  

    return X[-1], X  


def partial_deriv_error(x, y):
    return 2 * (x-y)

def partial_deriv_theta(xi):
    return xi * (1.0-xi)

def back_propagate(output, label_data,wt,X):
    W = list(wt) 
    error = []
    delta = []
    x = []
    d = []
    w = []
    y = []

    y = np.atleast_2d(label_data)   
    x = np.atleast_2d(output)
    error = np.average(x - y)
    delta = [partial_deriv_error(x, y) * partial_deriv_theta(x)] 

    for l in range(len(X)-2, 0, -1):
        d = np.atleast_2d(delta[-1])
        x = np.atleast_2d(X[l])
        w = np.array(W[l])

        delta.append(partial_deriv_theta(x) * delta[-1].dot(w.T) )    
        W[l] -= learning_rate * x.T.dot(d)

    x = np.atleast_2d(X[l-1])
    d = np.atleast_2d(delta[-1])            
    W[l-1] -= learning_rate * x.T.dot(d)        

    wt = W
    return error,wt

def neural_nw(training_data, label):

    training_data = np.array(training_data)
    label = np.array(label)
    layer = build_layers(training_data, label)        
    error = 0
    counter = 0
    X = []
    wt = weight_initialize(layer)

    for i in range (0, iteration): 
        j = np.random.randint(training_data.shape[0])
        _result,X = feed_forward(training_data[j],wt)
        error,wt = back_propagate(_result, label[j],wt,X)
        if abs(error) <= tolerance :
            counter += 1
            if counter >= converge:
                break
            else:
                pass
        else:
            counter = 0   
    return wt

def predict(x,wt):
    output = []
    output,temp = feed_forward(x[0],wt)
    for i in range(len(output)):
        if output[i] >= threshold:
            output[i] = 1
        else:
            output[i] = 0
    return output

def test_data_pred(labels,threshold,wt):
    count = 0
    true_pred = 0
    dimen = np.array(labels).ndim;
    if dimen == 1:
        lst_threshold = np.array(threshold)
    else:
        lst_threshold = np.array(threshold)*np.array(labels).shape[1]

    with open('downgesture_test.list') as f:
        for test_data in f.readlines():
            count += 1
            test_data = test_data.strip()
            prob = predict([get_image_data(test_data),],wt)

            if np.all(prob >= lst_threshold) == ('down' in test_data):
                if np.all(prob >= lst_threshold) :
                    print("{}: Predicted Output = {}".format(test_data, prob)) 
                else:
                    print("{}: Predicted Output = {}".format(test_data, prob))                   
                true_pred += 1
            else :
                if np.all(prob >= lst_threshold):
                    print("{}: Predict Output = {}".format(test_data, prob)) 
                else:
                    print("{}: Predict Output = {}".format(test_data, prob)) 

    print("\nAccuracy = {}%".format(true_pred / count*100))


if __name__ == '__main__':
    
    hidden_layer=np.array([100,])
    iteration=1000
    learning_rate=0.1 
    lower_bound=0
    upper_bound=1
    tolerance = 1e-6
    threshold = 0.5
    converge = 10
    training_data,labels = read_training_data()                                    
    wt = neural_nw (training_data, labels)
    test_data_pred(labels,threshold,wt)