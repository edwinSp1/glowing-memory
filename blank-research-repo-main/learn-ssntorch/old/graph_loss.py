import torch
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

checkpoint = torch.load('models/lineRecognizer')
ic('hi')
data = [(int(a), int(b)) for a, b in checkpoint['loss_data']]
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()