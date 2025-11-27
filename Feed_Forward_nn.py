#MNIST
#dataloader, Transformation
#multilayer Neural net, activation functions
#loss and optimizer
#training loop (batch training)
#model evaluation
#GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")#this saves the lock file

#device config, will run on GPU is it supported
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters

input_size = 784 #data is 28x28 but we will flatten it to 1D which is 784
hidden_size = 100
num_classes = 10 #digit from 0 to 9
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#MNIST classic data set found in torch and we import the data set to the same folder and set it as a training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True) #we transform the data to tensor and downloaded if is not downloaded

#setup test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor()) #we transform the data to tensor and no need to download

#setup dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples) #unpacks the train loader
print(samples.shape, labels.shape) 

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show() #shows the images of the numbers
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images',img_grid)
writer.close()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        #we dont do softmax here as we will use the cross entropy so it will apply it for us
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()
#sys.exit()

#training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0.0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # shape is 100, 1, 28, 28
        #inputsize is 784 so we need 100, 784 so we need to reshape
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward 
        outputs = model(images)
        loss = criterion(outputs, labels) #get the predicted outputs and actual labels

        #backward
        optimizer.zero_grad()
        loss.backward() #computes backpropagration
        optimizer.step() #updates the parameters

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        running_correct += (predictions == labels).sum().item()

        if(i+1) % 100 == 0:
            print(f'epoch{epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            
#test
#we dont need the gradient anymore
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        #we want to reshape here as well
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images) #here is the trained model and uses the test images

        #returns value, index  ---- the index is the class label
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0] #number of sample in the current batch (100)
        n_correct += (predictions == labels).sum().item() #for each correct prediction we add one

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')



