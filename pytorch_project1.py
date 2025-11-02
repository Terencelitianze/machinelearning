import torch
import numpy as np
'''
x = torch.randn(3, requires_grad=True)
print(x) 
y = x+2
print(y) 
z = y*y*2
#z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) #dz/dx
print(x.grad)

x = torch.randn(3, requires_grad=True)
print(x)

#stopping pytorch from having gradient function
#x.requires_grad_(False)
#y = x.detach()
with torch.no_grad():
    y = x+2
    print(y)

weights = torch.ones(4,requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad) #the gradient is accumulative

    weights.grad.zero_() #this clears the accumulative part of grad
    

#backpropagration

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad= True)


#forward pass
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

#backward pass
loss.backward()
print(w.grad) '''



