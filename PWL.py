import torch
import torch.nn as nn
import torch.functional as F

import numpy as np

import matplotlib.pyplot as plt

class PWL(nn.Module):
    def __init__(self):