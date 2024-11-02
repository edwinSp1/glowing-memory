
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from train import train
from settings import *
import atexit

def to_float(arr):
  return [(float(x), float(y)) for x, y in arr]
start = True

data = {}
if start:
  for i, model_path in enumerate(MODELS):
    acc_comb, loss_comb, time_comb = [], [], []
    for _ in range(DATASETS):
      loss, acc, time = train(model_path, DATA_FN)
      ic(loss, acc, time)
      acc = to_float(acc)
      loss = to_float(loss)
      loss_comb.extend(loss)
      acc_comb.extend(acc)
      time_comb.extend(time)
    data[i] = (loss_comb, acc_comb, time_comb)
else:
  data = torch.load(DATA_SAVE_PATH)

torch.save(data, DATA_SAVE_PATH)

ic(data)

fig, axs = plt.subplots(2)
for i in data:
  loss, acc, time = data[i]
  ic(loss, acc, time)
  axs[i].scatter(time, [a[0] for a in acc])
  axs[i].scatter(time, [a[1] for a in acc])

plt.show()


