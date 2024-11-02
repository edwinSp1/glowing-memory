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
SIZE = 3
TRAIN_SAMPLES = 100
TEST_SAMPLES = 10
VALIDATION_SAMPLES = 20
HIDDEN_RATIO = 10


# If this is set to False, the program will try to get the model checkpoint from OUTPUT_PATH.
# otherwise it will generate a new checkpoint from scratch
FIRST_TIME_RUNNING = True

# output constants
OUTPUT_PATH = 'models/10x10Recognizer'
# Network Architecture
num_inputs = SIZE*SIZE
num_hidden = SIZE*SIZE*HIDDEN_RATIO
num_outputs = 2

# temp dynamics
num_steps = 25
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # stage 1: transform from floats to ints using leaky and convolution
        self.lif1 = snn.Leaky(beta=beta)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_inputs)
        
        self.softmax = nn.Softmax(dim=0)

        self.process_line = nn.Sequential(
            nn.Linear(SIZE, SIZE*10),
            nn.ReLU(),
            nn.Linear(SIZE*10, SIZE*10),
            nn.ReLU(),
            nn.Linear(SIZE*10, SIZE),
            nn.ReLU(),
            nn.Linear(SIZE, 2),
            nn.ReLU(),
            nn.Softmax(dim=0)
        )

    def get_pred(self, x):
        preds = []
        for row in x:
            preds.append(self.process_line(row))

        for i in range(SIZE):
            col = []
            for j in range(SIZE):
                col.append(x[j][i])
            col = torch.tensor(col)
            preds.append(self.process_line(col))
        preds = torch.stack(preds)
       
        tot = torch.zeros(2)
        for pred in preds:
            tot += pred
        #ic(tot)
        return tot
    def forward(self, x):
        
        x = torch.flatten(x)
        #ic(x)
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

         # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            #cur1 = self.fc1(x)
            #spk1, mem1 = self.lif1(cur1, mem1)
            #cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(x, mem2)
            
            stacked = []
            for i in range(SIZE):
                cur = []
                for j in range(SIZE):
                    cur.append(spk2[i*3+j])
                #ic(cur)
                stacked.append(torch.stack(cur))
            #ic(stacked)
            spk2_rec.append(torch.stack(stacked))
            mem2_rec.append(mem2)

        #ic(spk2_rec)
        prob = torch.zeros(2)
        for spk in spk2_rec:
            line_pred = self.get_pred(spk)
            prob += line_pred
        #ic(prob)
       
        #prob = self.softmax(prob)
        #ic(prob)
        return prob
        

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
    validation_set = create_testcases(SIZE, SIZE, num_samples=VALIDATION_SAMPLES)
    total_loss, accuracy = test_loop(validation_set)

    return total_loss, accuracy

def test_loop(test_set):
    with torch.no_grad():
        correct = 0
        total_loss = 0
        for matrix_hash, label in test_set:
            # put the tensors onto CUDA
            matrix_hash = matrix_hash.to(device)

            # get model prediction
            pred = net(matrix_hash)
            
            total_loss += loss(pred, label)


        percentage = correct/len(test_set)*100
        return total_loss, percentage


if __name__ == '__main__':

    #if FIRST_TIME_RUNNING:
    train_set = create_testcases(SIZE, SIZE, TRAIN_SAMPLES)
    test_set = create_testcases(SIZE, SIZE, TEST_SAMPLES)

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
            pred = torch.softmax(pred, 0)
            
            loss_val += loss(pred, label)
        #ic(loss_val)
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

