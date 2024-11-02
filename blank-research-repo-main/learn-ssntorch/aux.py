import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from icecream import ic
t = torch.tensor(
  [0, 0, 100000], dtype=torch.float)
label = torch.tensor(0)
ic(t, label)
loss = nn.CrossEntropyLoss()

ic(loss(t, label))

