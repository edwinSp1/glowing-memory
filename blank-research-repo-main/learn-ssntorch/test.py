
#from model import *
import torch
import torch.nn as nn
m = nn.Conv1d(10, 20, 1)
mx = nn.MaxPool1d(kernel_size=1, stride=1)
m2 = nn.Conv1d(10*2, 10*4, 1)
flat = nn.Flatten(0, 1)
input = torch.randn(10, 10)
print(input.shape)
output = mx(input)
print(output.shape)
"""
output2 = m2(output)
print(output2.shape)
output3 = flat(output2)
print(output3.shape)
"""