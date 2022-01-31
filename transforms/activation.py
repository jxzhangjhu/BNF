import warnings
import matplotlib.pyplot as plt
import slim

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transforms.base import (
    CompositeTransform,
    InputOutsideDomain,
    InverseTransform,
    Transform,
)
from utils import torchutils


class CompactResAct(Transform):
    def __init__(self, hidden_feature, num_hiddenlayer, sigma_min=0.1, sigma_max=10, activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
        self.CompactResNet = nn.ModuleList([])
        self.CompactResNet.append(
            nn.Linear(in_features=1, out_features=hidden_feature)
            # slim.SpectralLinear(insize=1, outsize=hidden_feature, sigma_min=sigma_min, sigma_max=sigma_max)
        )
        for i in range(num_hiddenlayer):
            self.CompactResNet.append(
                nn.Linear(in_features=hidden_feature, out_features=hidden_feature)
                # slim.SpectralLinear(insize=hidden_feature, outsize=hidden_feature, sigma_min=sigma_min, sigma_max=sigma_max)
            )
        self.CompactResNet.append(
            nn.Linear(in_features=hidden_feature, out_features=1)
            # slim.SpectralLinear(insize=hidden_feature, outsize=1, sigma_min=sigma_min, sigma_max=sigma_max)
        )

    def forward(self, inputs, context=None):
        outputs = inputs
        for layer in self.CompactResNet:
            outputs = self.activation(layer.forward(outputs))
        return outputs

class MonotonicLinear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.w = nn.Parameter(F.softplus(torch.randn([out_feature, 1])))
        self.b = nn.Parameter(torch.randn(out_feature))

    def forward(self, inputs):
        return torch.einsum('bi, oi->bio', inputs, self.w) + self.b

class MonotonicBlock(nn.Module):
    def __init__(self, in_feature, hidden_feature):
        super().__init__()
        self.w = nn.Parameter(F.softmax(torch.randn(hidden_feature), dim=0))
        # print(self.w)
        # print(self.w.shape)
        self.linear = MonotonicLinear(in_feature=in_feature, out_feature=hidden_feature)

    def forward_(self, inputs):
        out1 = torch.tanh(self.linear(inputs))
        outputs = torch.atanh(torch.einsum('bih, h->bi', out1, self.w))
        return outputs

    def slope(self, inputs):
        out1 = torch.tanh(self.linear(inputs))
        out2 = torch.einsum('bih, h->bi', out1, self.w)
        a = torch.einsum('bi, h->bih', 1 / (1 - out2 ** 2), self.w)
        b = torch.einsum('bih, bih->bih', a, 1 - out1 ** 2)
        return torch.einsum('bih, hj->bi', b, self.linear.w)

    def forward(self, inputs):
        out1 = torch.tanh(self.linear(inputs))
        out2 = torch.einsum('bih, h->bi', out1, self.w)
        outputs = torch.tanh(out2)
        a = torch.einsum('bi, h->bih', 1 / (1 - out2 ** 2), self.w)
        b = torch.einsum('bih, bih->bih', a, 1 - out1 ** 2)
        c = torch.einsum('bih, hj->bi', b, self.linear.w)
        logabsdet = torch.sum(torch.log(c), dim=1)
        return outputs, logabsdet

class MonotonicTanhAct(Transform):
    def __init__(self, data_feature, hidden_feature, num_layer):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(num_layer):
            self.layers.append(MonotonicBlock(in_feature=data_feature, hidden_feature=hidden_feature))

    def forward_(self, inputs, context=None):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward_(outputs)
        return outputs

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        outputs = inputs
        logabsdetsum = torch.zeros(batch_size)
        for layer in self.layers:
            outputs, logabsdet = layer(outputs)
            logabsdetsum += logabsdet
        return outputs, logabsdetsum


if __name__ == "__main__":
    np.random.seed(1137)
    torch.manual_seed(114514)

    test_net = CompactResAct(hidden_feature=256, num_hiddenlayer=4, activation=nn.ReLU())

    x = torch.arange(-10.0, 10.0, 0.01)

    y = test_net(x.reshape(-1, 1))

    x_plot = x.detach().numpy()
    y_plot = y.detach().numpy()

    plt.plot(x_plot, y_plot)

    plt.show()