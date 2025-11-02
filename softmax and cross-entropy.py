import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0) #equations for softmax
#squashes the weights to a probability between 0 and 1

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

#one-hot encoded class means one class is 1 and all other is zero

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted)) #formula for cross_entropy
    return loss

#y must be one hot encoded
#if class 0: [1,0,0]
#if class 1: [0,1,0]
#if class 2: [0,0,1]

Y = np.array([1,0,0])

Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'loss1 numpy: {l1:.4f}')
print(f'loss2 numpy: {l2:.4f}')




