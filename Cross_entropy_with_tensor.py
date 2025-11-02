import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss() #it applies nn.logSoftmax and nn.NULLLoss
#3 samples 
Y = torch.tensor([2, 0, 1]) #it should be class 2 for first sample
#class 0 for second sample and class 1 for third sample

# so do not apply softmax yourself
#Y should not be one hot so hot class labels
# Y = torch.tensor([0]) #class 0
# nsamples x nclasses = 3x3
#this is the raw values not the softmax values
Y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]])
Y_pred_bad = torch.tensor([[0.5,2.0,0.3],[0.1,1.0,2.1],[2.0,1.0,0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1) #with dimension 1
_, predictions2 = torch.max(Y_pred_bad, 1) #with dimension 1

print(predictions1) #choose class 0
print(predictions2) #choos class 1

#nn.BCELoss() for binary classifictaion problem