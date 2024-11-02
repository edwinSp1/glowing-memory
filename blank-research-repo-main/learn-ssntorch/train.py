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
import time

from settings import *
import importlib
from datasets import get_loaders

def train(model_path, data_fn):

    model_info = importlib.import_module(model_path)
    importlib.invalidate_caches()

    train_loader, test_loader = get_loaders(data_fn)

    Net, OUTPUT_PATH, num_correct, calc_loss = model_info.Net, model_info.OUTPUT_PATH, model_info.num_correct, model_info.calc_loss

    net = Net()
    start = time.time()
    # Load the network onto CUDA if available
    net = Net().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    epoch = 0

    accuracy_data = []
    loss_data = []
    time_data = []
    if not DONT_LOAD_MODEL:
        # reload from checkpoint
        checkpoint = torch.load(OUTPUT_PATH, weights_only=False)
        epoch = 0

        net.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        accuracy_data = checkpoint['accuracy_data'] 
        loss_data = checkpoint['loss_data']
    


    print("train set and test set finished generating")
    # save model on exit
    def onexit():
        torch.save({
            "state_dict": net.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy_data": accuracy_data,
            "loss_data": loss_data
        }, OUTPUT_PATH)
        print("model checkpoint saved to file.")
    
    atexit.register(onexit)
    
    def test_loop(test_loader):
        with torch.no_grad():
            correct = 0
            total_loss = 0
            for imgs, labels in test_loader:
                preds = net(imgs.reshape(-1, 28*28))
            
                batch_loss = calc_loss(loss, preds, labels)
                total_loss += batch_loss.item()
                
                correct += num_correct(preds, labels)

            percentage = correct/len(test_loader.dataset)*100
            return total_loss, percentage
    
    #ic(test_loop(test_loader))
    for epoch in range(EPOCHS):

        loss_val = 0
        correct = 0

        for (imgs, labels) in train_loader:
            net.train()

            preds = net(imgs.reshape(-1, 28*28))
            #ic(preds)
            batch_loss = calc_loss(loss, preds, labels)
            #ic(batch_loss)
            loss_val += batch_loss.item()
            
            correct += num_correct(preds, labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss = loss_val
        train_accuracy = correct/len(train_loader.dataset) * 100
        # test model
        test_loss, test_accuracy = test_loop(test_loader)
        ic(epoch)
        ic(train_loss)
        ic(train_accuracy)
        ic(test_loss)
        ic(test_accuracy)

        loss_data.append((train_loss, test_loss))
        accuracy_data.append((train_accuracy, test_accuracy))
        time_data.append(time.time()-start)
    return loss_data, accuracy_data, time_data
    

#ic(train('model_info_normal', datasets.MNIST))
