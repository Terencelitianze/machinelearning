import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)
#train the model...
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#during model training we can save a check point
checkpoint = {
    "epoch": 90, #create a dictionary and state the number of epoch runned
    "model_state": model.state_dict(), #save the state of model using state_dict
    "optim_state":optimizer.state_dict() #save the state of the optimizer using the state_dict
}

#call torch.save to save the dictionary 
torch.save(checkpoint, "checkpoint.pth")

#after saving we can load the checkpoint
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"] #get the epoch using the epoch key

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

#this will load all the parameters into our model that was saved
model.load_state_dict(checkpoint["model_state"]) 
optimizer.load_state_dict(checkpoint["optim_state"])





FILE = "model.pth" #pytorch file
torch.save(model.state_dict(), FILE) #we save the model to this path

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE)) #loads the model from the FILE path
loaded_model.eval() #sets the model to evaluation mode

#printing the parameters of the model to show it works
for param in loaded_model.parameters():
    print(param)
