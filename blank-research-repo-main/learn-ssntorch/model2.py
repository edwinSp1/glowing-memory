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

from create_testcases import create_testcases
from icecream import ic
import atexit

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# input constants
ROWS = 10
COLS = 10
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000
VALIDATION_SAMPLES = 20
HIDDEN_RATIO = 10


# If this is set to False, the program will try to get the model checkpoint from OUTPUT_PATH.
# otherwise it will generate a new checkpoint from scratch
FIRST_TIME_RUNNING = True

# output constants
OUTPUT_PATH = 'models/10x10Recognizer'
# Network Architecture
num_inputs = ROWS*COLS
num_hidden = ROWS*COLS*HIDDEN_RATIO
num_outputs = 2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        self.lif = snn.Leaky
        self.conv_stack = nn.Sequential(
            # rows*cols -> (rows*2, cols)
            nn.Conv1d(ROWS, ROWS*2, 1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.ReLU(),
            # rows*cols -> (rows*4, cols)
            nn.Conv1d(ROWS*2, ROWS*4, 1),
            nn.Flatten(0, 1),
            nn.ReLU(),
            nn.Linear(ROWS*4*COLS, ROWS*COLS),
            nn.ReLU(),
            nn.Linear(ROWS*COLS, ROWS*COLS//10),
            nn.ReLU(),
            nn.Linear(ROWS*COLS//10, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.conv_stack(x)
        return logits

net = Net()

# Load the network onto CUDA if available
net = Net().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
epoch = 0

accuracy_data = []
loss_data = []
if not FIRST_TIME_RUNNING:
    # reload from checkpoint
    checkpoint = torch.load(OUTPUT_PATH, weights_only=False)
    epoch = 0

    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    accuracy_data = checkpoint['accuracy_data'] 
    loss_data = checkpoint['loss_data']

def validate():
    validation_set = create_testcases(ROWS, COLS, num_samples=VALIDATION_SAMPLES)
    total_loss, accuracy = test_loop(validation_set)

    return total_loss, accuracy

def test_loop(test_set):
    with torch.no_grad():
        correct = 0
        total_loss = 0
        for matrix_hash, label in test_set:
            # put the tensors onto CUDA
            matrix_hash = matrix_hash.to(device)
            label = label.to(device)

            # get model prediction
            pred = net(matrix_hash)
            
            total_loss += loss(pred, label)

            pred_arr = pred.numpy()
            label_arr = label.numpy()
            correct += 1
            for a, b in zip(pred_arr, label_arr):
                if a != b:
                    correct -=1
                    break 
        percentage = correct/len(test_set)*100
        return total_loss, percentage


if __name__ == '__main__':

    #if FIRST_TIME_RUNNING:
    train_set = create_testcases(ROWS, COLS, TRAIN_SAMPLES)
    test_set = create_testcases(ROWS, COLS, TEST_SAMPLES)

    print("train set and test set finished generating")
    print(net(train_set[0][0]))
    # save model on exit
    def exit():
        torch.save({
            "state_dict": net.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy_data": accuracy_data,
            "loss_data": loss_data
        }, OUTPUT_PATH)
        print("model checkpoint saved to file.")

    atexit.register(exit)
    while True:
        loss_val = 0
        correct = 0
        for matrix_hash, label in train_set:
            net.train()
            # put the tensors onto CUDA
            matrix_hash = matrix_hash.to(device)
            label = label.to(device)

            # get model prediction
            pred = net(matrix_hash)
            
            loss_val += loss(pred, label)
            

        optimizer.zero_grad()   
        loss_val.backward()
        optimizer.step()

        train_loss = loss_val
        train_accuracy = correct/len(train_set) * 100
        # test model
        test_loss, test_accuracy = test_loop(test_set)
        ic(epoch)
        ic(train_loss)
        ic(train_accuracy)
        ic(test_loss)
        ic(test_accuracy)

        loss_data.append((train_loss, test_loss))
        epoch += 1

