import torch
import torch.nn as nn 
import torchvision
from torchvision import transforms
from settings import BATCH_SIZE

# takes in a dataset, returns train_loader, test_loader
def get_loaders(data_fn):
    train_dataset = data_fn(
        root='./data', 
        train=True, 
        transform=transforms.ToTensor(),  
        download=True
    )
    test_dataset = train_dataset = data_fn(
        root='./data', 
        train=False, 
        transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,                             
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    return train_loader, test_loader
      