"""Implementations of invertible non-linearities."""

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


class Tanh(Transform):
    def forward(self, inputs, context=None):
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
            raise InputOutsideDomain()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class LogTanh(Transform):
    """Tanh with unbounded output. 

    Constructed by selecting a cut_point, and replacing values to the right of cut_point
    with alpha * log(beta * x), and to the left of -cut_point with -alpha * log(-beta *
    x). alpha and beta are set to match the value and the first derivative of tanh at
    cut_point."""

    def __init__(self, cut_point=1):
        if cut_point <= 0:
            raise ValueError("Cut point must be positive.")
        super().__init__()

        self.cut_point = cut_point
        self.inv_cut_point = np.tanh(cut_point)

        self.alpha = (1 - np.tanh(np.tanh(cut_point))) / cut_point
        self.beta = np.exp(
            (np.tanh(cut_point) - self.alpha * np.log(cut_point)) / self.alpha
        )

    def forward(self, inputs, context=None):
        mask_right = inputs > self.cut_point
        mask_left = inputs < -self.cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = torch.tanh(inputs[mask_middle])
        outputs[mask_right] = self.alpha * torch.log(self.beta * inputs[mask_right])
        outputs[mask_left] = self.alpha * -torch.log(-self.beta * inputs[mask_left])

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = torch.log(1 - outputs[mask_middle] ** 2)
        logabsdet[mask_right] = torch.log(self.alpha / inputs[mask_right])
        logabsdet[mask_left] = torch.log(-self.alpha / inputs[mask_left])
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):

        mask_right = inputs > self.inv_cut_point
        mask_left = inputs < -self.inv_cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = 0.5 * torch.log(
            (1 + inputs[mask_middle]) / (1 - inputs[mask_middle])
        )
        outputs[mask_right] = torch.exp(inputs[mask_right] / self.alpha) / self.beta
        outputs[mask_left] = -torch.exp(-inputs[mask_left] / self.alpha) / self.beta

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = -torch.log(1 - inputs[mask_middle] ** 2)
        logabsdet[mask_right] = (
            -np.log(self.alpha * self.beta) + inputs[mask_right] / self.alpha
        )
        logabsdet[mask_left] = (
            -np.log(self.alpha * self.beta) - inputs[mask_left] / self.alpha
        )
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet


class LeakyReLU(Transform):
    def __init__(self, negative_slope=1e-2):
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        super().__init__()
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.log(torch.as_tensor(self.negative_slope))

    def forward(self, inputs, context=None):
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = F.leaky_relu(inputs, negative_slope=(1 / self.negative_slope))
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = -self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class Sigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def forward(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )
        return outputs, logabsdet

class CubicPolynomial(Transform):
    def __init__(self):
        super(CubicPolynomial, self).__init__()

    def forward(self, inputs, context=None):
        outputs = inputs ** 3
        logabsdet = torchutils.sum_except_batch(
            torch.log(torch.tensor([3.0])) + 2 * torch.log(inputs)
        )
        return outputs, logabsdet

