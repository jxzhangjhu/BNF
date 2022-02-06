'''
Implementation of Blocked version of MADE
'''

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import distributions, nn
from torch.nn import functional as F
from torch.nn import init

from utils import torchutils

# Basic function to create the block degree given input size and number of blocks;
def _create_block_degree(feature, num_block, mode='stack'):
    block_feature = feature // num_block
    # The dimensions within the same block id is stacked together;
    if mode == 'stack':
        block_degree = torch.arange(0, num_block + 1).repeat_interleave(block_feature)
    # Sequentially assign block id to each dimension;
    elif mode == 'seq':
        block_degree = torch.arange(1, num_block + 2).repeat(block_feature)
    elif mode == 'random':
        block_degree = torch.randint(
            low=1,
            high=num_block + 2,
            size=[feature],
            dtype=torch.long,
        )
    else:
        raise ValueError('Invalid create degree mode. Use stack, seq or random.')
    return block_degree[0:feature]


class BlockMaskedLinear(nn.Linear):
    def __init__(self, in_feature, out_feature, in_block_degree, rand_mask=False, is_output=False, bias=True):
        super().__init__(in_features=in_feature, out_features=out_feature, bias=bias)
        self.in_block_degree = in_block_degree
        self.num_block = torch.max(self.in_block_degree).item()
        self.rand_mask = rand_mask
        self.is_output = is_output
        # Get the mask and block degree of the current layer;
        mask, block_degree = self._get_mask_and_block_degree()
        self.register_buffer('mask', mask)
        self.register_buffer('block_degree', block_degree)

    def _get_mask_and_block_degree(self):
        # Assign sequential block degree for 'block_feature' times for every output unit in each block from 0 to 'num_blocks - 1';
        if self.is_output:
            block_degree = _create_block_degree(feature=self.out_features, num_block=self.num_block, mode='stack')
            mask = (block_degree[..., None] >= self.in_block_degree).float()
        else:
            # Assign random mask from 1 to 'num_blocks' for each hidden unit;
            if self.rand_mask:
                block_degree = _create_block_degree(feature=self.out_features, num_block=self.num_block, mode='random')
                mask = (block_degree[..., None] >= self.in_block_degree).float()
            # Assign sequential mask from 1 to 'num_blocks' for each hidden unit;
            else:
                block_degree = _create_block_degree(feature=self.out_features, num_block=self.num_block, mode='seq')
                mask = (block_degree[..., None] >= self.in_block_degree).float()
        return mask, block_degree


class BlockMaskedFeedForwardLayer(nn.Module):
    def __init__(self,
                 in_feature,
                 out_feature,
                 in_block_degree,
                 rand_mask=False,
                 is_output=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 zero_initialization=False
                 ):
        super().__init__()

        # Batch norm;
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_feature, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear layer;
        self.mask_linear = BlockMaskedLinear(
            in_feature=in_feature,
            out_feature=out_feature,
            in_block_degree=in_block_degree,
            rand_mask=rand_mask,
            is_output=is_output
        )
        self.block_degree = self.linear.block_degree

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.mask_linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class BlockMaskedResidualLayer(nn.Module):
    def __init__(self):
        super().__init__()


class BlockMADE(nn.Module):
    def __init__(self,
                 in_feature,
                 hidden_feature,
                 out_feature,
                 num_block,
                 num_hidden_layer,
                 use_res_layer=True,
                 rand_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False
                 ):
        if use_res_layer and rand_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        # Construct the BlockMADE network;
        self.first_layer = BlockMaskedLinear(
            in_feature=in_feature,
            out_feature=hidden_feature,
            in_block_degree=_create_block_degree(in_feature, num_block=num_block),
            rand_mask=rand_mask,
            is_output=False
        )

        # Hidden layers;
        hidden_layers = []
        if use_res_layer:
            hidden_layer = BlockMaskedFeedForwardLayer
        else:
            hidden_layer = BlockMaskedResidualLayer
        for i in range(num_hidden_layer):
            hidden_layers.append(hidden_layer(
                in_feature=hidden_feature,
                out_feature=hidden_feature,
                in_block_degree=self.first_layer.block_degree,
                rand_mask=False,
                is_output=False,
                activation=F.relu,
                dropout_probability=0.0,
                use_batch_norm=False,
                zero_initialization=True
            ))

        # Last layer;
        self.last_layer = BlockMaskedLinear(
            in_feature=hidden_feature,
            out_feature=out_feature,
            in_block_degree=hidden_layers[-1].block_degree,
            rand_mask=rand_mask,
            is_output=True
        )

        def forward(self, inputs):
            tmp = self.first_layer(inputs)
            for hidden_layer in self.hidden_layers:
                temps = hidden_layer(tmp)
            outputs = self.last_layer(tmp)
            return outputs


if __name__ == '__main__':
    a = _create_block_degree(10, 3, mode='stack')
    print(a)