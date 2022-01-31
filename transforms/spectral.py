import warnings
import slim

import numpy as np
import torch
import torch.nn as nn

from transforms.base import Transform
from utils import torchutils
from torch import optim


class SpecTransformation(Transform):
    '''
    Implementation of a single layer, D dim to D dim linear transformation with SVD factorization;
    '''
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.speclayer = slim.SpectralLinear(insize=features, outsize=features)

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        input_feature = inputs.shape[1]
        assert input_feature == self.features
        sigma = self.speclayer.Sigma()
        outputs = self.speclayer(inputs)
        logabsdet = torch.sum(torch.log(torch.diag(sigma)))
        return outputs, logabsdet


if __name__ == "__main__":
    np.random.seed(1137)
    torch.manual_seed(114514)
    inputs = torch.randn(4, 2)
    net = SpecTransformation(2)
    b, c = net.forward(inputs)
    print(b, c)
