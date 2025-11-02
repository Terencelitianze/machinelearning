import torch
import torchvision #transforms images or numpy to tensors

class wineDataset(Dataset):

    def __init__(self, transform = None):
        #data loading
        xy = np.loadtxt('file path')

        self.transform = transform

    def __getitem__(self, index):
        #dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(self)

    def __len__(self):
        #len(dataset) 
        #return number of samples

class ToTensor:
    def __call__(self,sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)
    
class multransform:
    def __init__(self,factor):
        self.factor = factor

    def __call__(self,sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = wineDataset(transform=ToTensor()) #transforms the dataset to tensor values

composed = torchvision.transforms.Compose([ToTensor(), multransform(2)])
dataset = wineDataset(transform=composed) #transforms the dataset to tensor values and multiplies each tensor value by 2