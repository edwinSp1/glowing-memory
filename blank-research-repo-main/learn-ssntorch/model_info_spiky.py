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

from icecream import ic
import atexit
from settings import *

# output constants
OUTPUT_PATH = 'models/lineRecognizer'
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=beta)

    def process(self, x):
      mem1 = self.lif1.reset_mem()
      mem2 = self.lif2.reset_mem()

      spk_rec = []
      for step in range(num_steps):
        cur1 = self.fc1(x)
        cur1 = torch.relu(cur1)
        #ic(cur1)
        #attack_layer(cur1)
        #ic(cur1)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        cur2 = torch.relu(cur2)
        #attack_layer(cur2)
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc3(spk2)
        #attack_layer(cur3)
        spk_rec.append(cur3)

      return torch.stack(spk_rec, dim=0)

    def forward(self, batch):
      res = []
      for x in batch:
        res.append(self.process(x))
      return res
    
def calc_loss(loss, preds, label):
  loss_val = 0
  for pred, lbl in zip(preds, label):
    for spk in pred:
      loss_val += loss(spk, lbl)

  return loss_val

def num_correct(preds, label):
  with torch.no_grad():
    tot = 0
    for pred, lbl in zip(preds, label):
      s = torch.zeros(num_outputs)
      for spk in pred:
        s += spk

      tot += (s.argmax(dim=0) == lbl).item()

    return tot