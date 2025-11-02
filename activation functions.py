# To make the model non-linear fro more complex tasks

#TanH function = 2/1+exp(-2x) - 1 (typically used in hidden layer)

#RelU = max(0,x) (zero for negative values, linear for positive)

#leaky ReLU = ax for negative (a is very small)
#improved version of ReLu to solve the vanish gradient problem

#Softmax function = good choice for last layer of multi class layer

import torch
import torch.nn as nn
import torch.nn.functional as F

# use (create a nn module)
class NeuralNet(nn.Module): 
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) #from input size to hidden size
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1) #from hidden size to one output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

#option 2 use activation function in forward pass
class NeuralNet(nn.Module): 
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) #from input size to hidden size
        self.linear2 = nn.Linear(hidden_size, 1) #from hidden size to one output


    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        #F.leakyRelu
        return out
