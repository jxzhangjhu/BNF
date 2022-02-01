import torch
import torch.nn as nn
import torch.functional as F
import torchpwl

import numpy as np

import matplotlib.pyplot as plt


batch_size = 1
feature = 2
num_breakpoints = 10

x = torch.randn(batch_size, feature)
f = torchpwl.MonoPWL(num_channels=feature, num_breakpoints=num_breakpoints, monotonicity=1)

y = f(x)

slope = f.slope_at(x)
print('Slope is:', slope)


j = torch.autograd.functional.jacobian(f.forward, x)
real_j = torch.zeros(size=[batch_size, feature, feature])
for i in range(batch_size):
    real_j[i, ...] = j[i, :, i, :]
print('Torch Jacobian:', real_j)