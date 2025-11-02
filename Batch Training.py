# batch size = number of training samples in one forward and backward pass

#number of iterations = number of passes, each pass using [batch size] number of samples

#e.g 100 samples, batch size = 20 --> 100/20 = 5 iterations per 1 epoch

import torch
import torchvision
from torch.utils.data import dataset, DataLoader
import numpy as np
import math

class wineDataset(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('file path')

    def __getitem__(self, index):
        #dataset[0]

    def __len__(self):
        #len(dataset) 
        #return number of samples

dataset = wineDataset

dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True, num_workers = 2)#num_workers makes it faster
'''Divides data into mini-batches (here, of size 4).

Shuffles samples each epoch to avoid learning order bias.

Loads data in parallel using multiple worker threads (num_workers=2) for speed.'''

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        '''you automatically get mini-batches of (inputs, labels) ready for training — you don’t have to slice them manually.'''
        #forward, backward, update
        #for this example is printing the information for every five steps
        if (i+1)% 5 == 0
            print('info')

torchvision.datasets.MNIST()
#fashion-mnist, cifar, coco