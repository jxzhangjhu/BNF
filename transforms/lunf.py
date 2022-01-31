import warnings
import slim

import numpy as np
import torch
import torch.nn as nn

import transforms
from transforms.base import Transform
from utils import torchutils
from torch import optim


class LUSpectralTransformation(Transform):
    def __init__(self, in_feature, num_layer, activation_func=torch.nn.LeakyReLU()):
        super().__init__()
        self.in_feature = in_feature
        self.num_layer = num_layer
        self.activation_func = activation_func
        # Setup the linear layers;
        self.net = nn.ModuleList([])
        for i in range(num_layer):
            self.net.append(transforms.LULinear(features=in_feature))

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        input_feature = inputs.shape[1]
        outputs = inputs
        # Initialize the PWA weight and bias
        A_pwa = torch.eye(input_feature).reshape(1, input_feature, input_feature).repeat(batch_size, 1, 1)
        for layer in self.net:
            # Calculate delta_i, w_i and b_i;
            outputs, _ = layer.forward(outputs)
            w_i = layer.weight()
            delta_i = torch.diag_embed(self.activation_func(outputs) / outputs)
            outputs = self.activation_func(outputs)
            # Shape of tmp: bs * out_dim * in_dim
            tmp = delta_i @ w_i
            # Shape of A: bs * out_dim * in_dim; shape of b: bs * out_dim * 1
            A_pwa = torch.bmm(tmp, A_pwa)
        outputs_pwa = torch.bmm(A_pwa, inputs.reshape(batch_size, input_feature, 1))
        return outputs_pwa.reshape(batch_size, -1), torch.log(torch.det(A_pwa))

    def forward_normal(self, inputs):
        batch_size = inputs.shape[0]
        input_feature = inputs.shape[1]
        outputs = inputs
        for layer in self.net:
            outputs, _ = layer.forward(outputs)
            outputs = self.activation_func(outputs)
        return outputs


if __name__ == "__main__":
    np.random.seed(1137)
    torch.manual_seed(114514)
    batch_size = 2
    num_layer = 10
    feature = 4
    hidden_feature = 4
    inputs = torch.randn(batch_size, feature, requires_grad=True)
    net = LUSpectralTransformation(in_feature=feature, num_layer=num_layer)
    b, c = net.forward(inputs)
    print(b, c)

    print('normal out:\n', net.forward_normal(inputs))
