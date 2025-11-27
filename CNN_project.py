#using the CIFAR - 10 dataset which includes 10 classes of images

#max pooling - taking the max value of each pool (a part of a tensor)
#prevents over fitting as it abstracts the image

import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#dataset has PILI images of range [0,1]
#we transform them to Tensors of normalised range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#implement Convolutional net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #The kernel size is the dimension of the pooling window, often expressed as w x h
        # kernel is frequently used because it's small enough to retain important features while downsampling the image. 
        # The stride is the number of pixels the kernel moves at each step.
        #this input channel is three as we has three colors for images
        self.conv1 = nn.Conv2d(3, 6, 5) #ouput is 6 and kernel is 5 (5x5)
        self.pool = nn.MaxPool2d(2, 2) # kernel size of 2 and stride of 2
        self.conv2 = nn.Conv2d(6, 16, 5) # input size must equal output size of previous CNN
        #this is the Fully connected layer that does the classification
        
        # to calculate ouput size: (W - F + 2P)/S + 1 : 5x5 through 3x3 filter, padding = 0 and stride=1
        
        self.fc1 = nn.Linear(16*5*5, 120) #input is fix as it is the calculated output size of the 3d tensor
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #10 must be fix


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))) #apply relu and pooling
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 16*5*5)#-1 will pytorch will auto give correct size (number of batch),(num of sample in our batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss() #use this as it is a multi-class clasification problem
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#use Scarstic gradient decent

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(images, labels) in enumerate(train_loader): #loop over the training loader to get all the batches
        #original shape: [4,3,32,32] = 4,3,1024
        #input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device) #push the labels and images to device to get GPU support
        labels = labels.to(device)

        #foward pass 
        outputs = model(images)
        loss = criterion(outputs,labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 2000 == 0:
            print(f'epoch{epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

print('finished training')

#test
#we dont need the gradient anymore
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_classes_correct = [0 for i in range(10)]
    n_classes_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) #here is the trained model and uses the test images

        #returns value, index  ---- the index is the class label
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0) #number of sample in the current batch ()
        n_correct += (predicted == labels).sum().item() #for each correct prediction we add one

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_classes_correct[label] +=1
            n_classes_samples [label] +=1

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy of the CNN = {acc}')

    #accuracy of the class
    for i in range(10):
        acc = 100.0 * n_classes_correct[i] / n_classes_samples[i]
        print(f'accuracy of {classes[i]:} {acc}%')









