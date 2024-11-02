
import torch
import torchvision
from random import random
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# neuron death
NEURON_DEATH_PROB = 0.05

def attack_layer(layer):
  for i in range(len(layer)):
    if random() < NEURON_DEATH_PROB:
      layer[i] = 0

# print data to console during training
PRINT_DATA = False

# If this is set to False, the program will try to get the model checkpoint from OUTPUT_PATH.
# otherwise it will generate a new checkpoint from scratch
DONT_LOAD_MODEL = True


# Network Architecture
num_inputs = 28*28
num_hidden = 1024
num_outputs = 10



# train/graph settings
EPOCHS = 5
BATCH_SIZE = 100
DATA_FN = torchvision.datasets.MNIST

# temp dynamics (for spiky)
num_steps = 25
beta = 0.95

DATASETS = 1
DATA_SAVE_PATH = 'data/trainingdata'
MODELS = ['model_info_normal', 'model_info_spiky']