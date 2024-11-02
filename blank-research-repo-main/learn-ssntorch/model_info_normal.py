# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

from settings import *
from icecream import ic
import atexit

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# output constants
OUTPUT_PATH = 'models/lineRecognizer'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
    def forward(self, x):
      x = self.fc1(x)
      x = torch.relu(x)
      x = self.fc2(x)
      x = torch.relu(x)
      x = self.fc3(x)

      return x
  
def calc_loss(loss, preds, label):
  return loss(preds, label)

def num_correct(preds, label):
  with torch.no_grad():
    return (preds.argmax(1) == label).type(torch.float).sum().item()