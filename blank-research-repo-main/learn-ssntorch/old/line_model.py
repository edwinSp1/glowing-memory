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

from create_testcases2 import create_line_testcases
from icecream import ic
import atexit
from settings import *

# Network Architecture
num_inputs = LENGTH
num_hidden = LENGTH*HIDDEN_RATIO
num_outputs = 2

# temp dynamics
num_steps = 25
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.process_line = nn.Sequential(
            nn.Linear(LENGTH, LENGTH*10),
            nn.ReLU(),
            nn.Linear(LENGTH*10, LENGTH*10),
            nn.ReLU(), 
            nn.Linear(LENGTH*10, 2),
        )

    def forward(self, x):
        
      return self.process_line(x)
    

