# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

#from model2 import Net

inp = torch.rand(3, 3)
print(inp)
inp = inp > 0

print(torch.any(inp))