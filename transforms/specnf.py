import warnings
import slim

import numpy as np
import torch
import torch.nn as nn

from transforms.base import Transform
from utils import torchutils
from torch import optim


class SpecTransformation(Transform):
    def __init__(self, in_feature, hidden_feature, out_feature, num_hidden_layer, activation, sigma_min=0.1, sigma_max=10):
        super().__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.out_feature = out_feature
        self.num_hidden_layer = num_hidden_layer
        # Add parameterized linear layers and activation layers to corresponding container;
        self.linearnet = nn.ModuleList([])
        self.actnet = nn.ModuleList([])
        self.linearnet.append(
            slim.SpectralLinear(insize=in_feature, outsize=hidden_feature, sigma_min=sigma_min, sigma_max=sigma_max,
                                n_U_reflectors=2 * in_feature, n_V_reflectors= 2 * in_feature))
        for _ in range(num_hidden_layer):
            if (activation == 'IDENTITY'):
                self.actnet.append(torch.nn.Identity())
            elif (activation == 'LRELU'):
                self.actnet.append(torch.nn.LeakyReLU())
            elif (activation == 'PRELU'):
                self.actnet.append(torch.nn.PReLU())
            else:
                raise ValueError('activation not supported')
            self.linearnet.append(
                slim.SpectralLinear(insize=in_feature, outsize=hidden_feature, sigma_min=sigma_min, sigma_max=sigma_max,
                                    n_U_reflectors=2 * in_feature, n_V_reflectors=2 * in_feature))
        if (activation == 'IDENTITY'):
            self.actnet.append(torch.nn.Identity())
        elif (activation == 'LRELU'):
            self.actnet.append(torch.nn.LeakyReLU())
        elif (activation == 'PRELU'):
            self.actnet.append(torch.nn.PReLU())
        else:
            raise ValueError('activation not supported')
        self.linearnet.append(
            slim.SpectralLinear(insize=in_feature, outsize=hidden_feature, sigma_min=sigma_min, sigma_max=sigma_max,
                                n_U_reflectors=2 * in_feature, n_V_reflectors=2 * in_feature))

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        total_logabsdet = inputs.new_zeros(batch_size)
        input_feature = inputs.shape[1]
        # Initialize the PWA weight and bias
        A_pwa = self.linearnet[0].effective_W().T.repeat(batch_size, 1, 1)
        b_pwa = self.linearnet[0].bias.repeat(batch_size, 1)
        # First linear layer;
        outputs = self.linearnet[0](inputs)
        for i in range(self.num_hidden_layer + 1):
            # Calculate delta_i, w_i and b_i, note the shape of effective weight from spectrallinear is in_dim * out_dim;
            w_i = self.linearnet[i + 1].effective_W().T
            b_i = self.linearnet[i + 1].bias
            act_outputs = self.actnet[i](outputs)
            delta_i = act_outputs / outputs
            sigma = self.linearnet[i].Sigma()
            # Shape of tmp: bs * out_dim * in_dim
            tmp = torch.einsum('oi, bi->boi', w_i, delta_i)
            # Shape of A: bs * out_dim * in_dim; shape of b: bs * out_dim * 1
            A_pwa = torch.einsum('boh, bhi->boi', tmp, A_pwa)
            b_pwa = torch.einsum('boh, bh->bo', tmp, b_pwa) + b_i
            outputs = self.linearnet[i + 1](act_outputs)

            # Log determinant from linear layer;
            det_linear = torch.sum(torch.log(torch.diag(sigma)))
            # Log determinant from activation layer;
            det_act = torch.sum(torch.log(delta_i), dim=1)
            total_logabsdet += det_linear + det_act
        # Final output;
        outputs_pwa = torch.einsum('boi, bi->bo', A_pwa, inputs) + b_pwa
        # Final logabsdet;
        sigma = self.linearnet[self.num_hidden_layer + 1].Sigma()
        total_logabsdet += torch.sum(torch.log(torch.diag(sigma)))
        return outputs_pwa, total_logabsdet

