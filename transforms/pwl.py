import warnings
import torchpwl

import numpy as np
import torch
import torch.nn as nn

from transforms.base import Transform
from utils import torchutils
from torch import optim

class PWLTransformation(Transform):
    def __init__(self, feature, num_breakpoints, monotonicity=1):
        super().__init__()
        self.mono_pwl_function = torchpwl.MonoPWL(
            num_channels=feature, num_breakpoints=num_breakpoints, monotonicity=monotonicity)

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        outputs = self.mono_pwl_function(inputs)
        logabsdet = torch.sum(torch.log(self.mono_pwl_function.slope_at(inputs)), dim=1)
        return outputs, logabsdet
