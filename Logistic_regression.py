import torch
import torch.nn as nn
from sklearn import datasets #to get binary classificaion dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #we want have seperation between training and testing data
import numpy as np

#0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler() #make our features have zero mean and variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) #from one row to one column vector
y_test = y_test.view(y_test.shape[0], 1)
#1) model setup
#f = wx + b, sigmoid at end
class LogisticRegression(nn.Module):

    def __init__(self,n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)


#2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() #binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#3) training loop
num_epochs = 100

for epochs in range(num_epochs):
    #forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    #backward pass
    loss.backward()
    #updates
    optimizer.step()
    optimizer.zero_grad()

    if (epochs+1) % 10 == 0: 
        print(f'epoch: {epochs+1}, loss = {loss.item():.4f}')

#evaluation
with torch.no_grad():
    y_predicted = model(X_test)
    #sigmod returns value between 0 and 1
    y_predicted_cls = y_predicted.round() #classes
    #check accuracy if predicted is equal to y_test divide by total
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0]) #number of test samples
    print(f'accuracy = {acc:.4f}')