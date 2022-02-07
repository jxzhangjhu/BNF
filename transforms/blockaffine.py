'''
Implementation of blocked affine transformation with BlockMADE.
'''

import math
import numpy as np
import torch
import torch.nn as nn

from transforms.base import Transform
from utils import torchutils
from torch import optim

import torch
import torch.functional as F
import nn.nets.blockmade as bmade

from tqdm import tqdm
import matplotlib.pyplot as plt


class BlockAffineTransformation(Transform):
    def __init__(self, feature, block_feature, hidden_feature, sigma_max=1.0, sigma_min=0.1):
        super().__init__()
        self.feature = feature
        self.block_feature = block_feature
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        # Calculate number of blocks.
        self.num_block = math.ceil(feature / block_feature)

        # BlockMADE for SVD parameterization of block matrices;
        # U matrices generation BMADE;
        self.U_out_feature = (self.num_block - 1) * block_feature ** 2 + \
                             (feature - (self.num_block - 1) * block_feature) ** 2
        self.UMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.U_out_feature,
            num_block=self.num_block
        )
        # Sigma matrices generation BMADE;
        self.SMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.feature,
            num_block=self.num_block
        )
        # V matrices generation BMADE;
        self.VMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.U_out_feature,
            num_block=self.num_block
        )
        # Bias generation;
        self.BiasMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.feature,
            num_block=self.num_block
        )
        self.BMADEs = nn.ModuleList([self.UMADE, self.SMADE, self.VMADE, self.BiasMADE])

        self.reg_error = torch.tensor(0.0)

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        block_in_features_expect_last = (self.num_block - 1) * self.block_feature
        W_block_expect_last, W_last_block, logabdsdet = self._get_block_parameter(inputs)
        bias = self.BiasMADE(inputs)

        inputs_block_expect_last = inputs[:, 0:block_in_features_expect_last].reshape(
            self.num_block - 1, batch_size, self.block_feature
        )
        inputs_last_block = inputs[:, block_in_features_expect_last:]

        # Blocked transformation expect last block;
        outputs_block_expect_last = torch.einsum(
            'nbij, nbj->nbi',
            W_block_expect_last, inputs_block_expect_last
        )
        # Blocked transformation for last block;
        outputs_last_block = torch.einsum(
            'bij, bj->bi',
            W_last_block, inputs_last_block
        )

        outputs = torch.cat((outputs_block_expect_last.reshape(batch_size, -1), outputs_last_block), dim=1) + bias

        return outputs, logabdsdet

    def _get_block_parameter(self, inputs):
        batch_size = inputs.shape[0]
        # Get flattened outputs;
        u_flatten = self.UMADE(inputs)
        sigma = self.SMADE(inputs)
        v_flatten = self.VMADE(inputs)


        block_features_expect_last = (self.num_block - 1) * self.block_feature ** 2
        last_block_feature = self.feature - (self.num_block - 1) * self.block_feature

        # Parameters for all blocks expect the last block;
        U_block_expect_last = u_flatten[:, 0:block_features_expect_last].reshape(
            self.num_block - 1, batch_size, self.block_feature, self.block_feature)
        # Regularization of Sigma values;
        sigma_clapmed = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(sigma)
        self.sigma = sigma_clapmed
        sigma_clapmed_blocked = sigma_clapmed[:, 0:(self.num_block - 1) * self.block_feature].reshape(
            self.num_block - 1, batch_size, -1
        )
        Sigma_block_expect_last = torch.diag_embed(sigma_clapmed_blocked)
        V_block_expect_last = v_flatten[:, 0:block_features_expect_last].reshape(
            self.num_block - 1, batch_size, self.block_feature, self.block_feature)

        # print('u shape', U_block_expect_last.shape)
        # print('s shape', Sigma_block_expect_last.shape)
        # print('v shape', V_block_expect_last.shape)

        W_block_expect_last = torch.einsum(
            'nbij, nbjk, nbkl->nbil',
            U_block_expect_last, Sigma_block_expect_last, V_block_expect_last)

        # Parameters of the last block;
        U_last_block = u_flatten[:, block_features_expect_last:].reshape(
            batch_size, last_block_feature, last_block_feature)
        Sigma_last_block = torch.diag_embed(sigma_clapmed[:, (self.num_block - 1) * self.block_feature:])
        V_last_block = v_flatten[:, block_features_expect_last:].reshape(
            batch_size, last_block_feature, last_block_feature)
        W_last_block = torch.einsum(
            'bij, bjk, bkl->bil',
            U_last_block, Sigma_last_block, V_last_block
        )

        # Determinant of the block transformation is the product of all singular values;
        logabsdet = torch.sum(torch.log(sigma_clapmed), dim=1)


        # Direct way of calculating determinant;
        # det_expect_last = torch.sum(torch.log(torch.abs(torch.det(W_block_expect_last))), dim=0)
        # det_last_block = torch.log(torch.abs(torch.det(W_last_block)))
        #
        # logabsdet = det_expect_last + det_last_block

        # Compute the orthogonal regularization error;
        self.reg_error = self._orthogonal_error(U_block_expect_last) + self._orthogonal_error(U_last_block) +\
                         self._orthogonal_error(V_block_expect_last) + self._orthogonal_error(V_last_block)

        self.W_1 = U_block_expect_last
        self.W_2 = U_last_block

        return W_block_expect_last, W_last_block, logabsdet

    def _orthogonal_error(self, W):
        if len(W.shape) == 4:
            num_block = W.shape[0]
            batch_size = W.shape[1]
            block_feature = W.shape[2]
            I = torch.eye(block_feature).repeat(num_block, batch_size, 1, 1)
            W_t = torch.transpose(W, dim0=2, dim1=3)
            err_1 = torch.norm(I - torch.einsum('nbij, nbjk->nbik', W, W_t))
            err_2 = torch.norm(I - torch.einsum('nbij, nbjk->nbik', W_t, W))
        elif len(W.shape) == 3:
            batch_size = W.shape[0]
            block_feature = W.shape[1]
            I = torch.eye(block_feature).repeat(batch_size, 1, 1)
            W_t = torch.transpose(W, dim0=1, dim1=2)
            err_1 = torch.norm(I - torch.einsum('bij, bjk->bik', W, W_t))
            err_2 = torch.norm(I - torch.einsum('bij, bjk->bik', W_t, W))
        else:
            raise ValueError('Invalid block weight matrices shape.')
        return torch.mean(err_1 + err_2)

    def forward_(self, inputs, context=None):
        batch_size = inputs.shape[0]
        block_in_features_expect_last = (self.num_block - 1) * self.block_feature
        W_block_expect_last, W_last_block, logabdsdet = self._get_block_parameter(inputs)
        bias = self.BiasMADE(inputs)

        inputs_block_expect_last = inputs[:, 0:block_in_features_expect_last].reshape(
            self.num_block - 1, batch_size, self.block_feature
        )
        inputs_last_block = inputs[:, block_in_features_expect_last:]

        # Blocked transformation expect last block;
        outputs_block_expect_last = torch.einsum(
            'nbij, nbj->nbi',
            W_block_expect_last, inputs_block_expect_last
        )
        # Blocked transformation for last block;
        outputs_last_block = torch.einsum(
            'bij, bj->bi',
            W_last_block, inputs_last_block
        )

        outputs = torch.cat((outputs_block_expect_last.reshape(batch_size, -1), outputs_last_block), dim=1) + bias

        return outputs


if __name__ == '__main__':
    batch_size = 1
    in_feature = 6
    hidden_feature = 256
    block_feature = 2

    inputs = torch.randn(batch_size, in_feature)

    num_iter = 5000
    val_interval = 250
    lr = 0.001

    BlockNet = BlockAffineTransformation(
        feature=in_feature,
        block_feature=block_feature,
        hidden_feature=hidden_feature,
        sigma_max=10,
        sigma_min=1
    )

    # Setup optimizer;
    optimizer = optim.Adam(BlockNet.parameters(), lr=lr)

    tbar = tqdm(range(num_iter))
    train_loss = np.zeros(shape=(num_iter))
    logabsdet_error = np.zeros(shape=(num_iter//val_interval))
    cnt_val = 0

    for i in tbar:
        # Training iterations;
        BlockNet.train()
        optimizer.zero_grad()
        _, logabsdet = BlockNet(inputs)
        loss = BlockNet.reg_error
        train_loss[i] = loss.detach().numpy()
        loss.backward()
        optimizer.step()
        if (i + 0) % val_interval == 0:
            BlockNet.eval()
            print('Current loss:', train_loss[i])

            # Test the orthogonal property of one of the U matrix;
            W_2 = BlockNet.W_2
            print('Reconstructed one of U.T * U\n', torch.bmm(W_2, torch.transpose(W_2, 1, 2)).detach())

            # Print out the real Jacobian matrix, should observe diagonal block pattern;
            j = torch.autograd.functional.jacobian(BlockNet.forward_, inputs)
            # j = torch.autograd.functional.jacobian(block_made.forward, inputs)
            real_j = torch.zeros(size=[batch_size, in_feature, in_feature])
            for i in range(batch_size):
                real_j[i, ...] = j[i, :, i, :]
            print(real_j.detach())
            b1 = real_j[:, 0:2, 0:2]
            b2 = real_j[:, 2:4, 2:4]
            b3 = real_j[:, 4:6, 4:6]

            logabsdet_1 = torch.log(torch.abs(torch.det(b1)))
            logabsdet_2 = torch.log(torch.abs(torch.det(b2)))
            logabsdet_3 = torch.log(torch.abs(torch.det(b3)))

            real_logabsdet = logabsdet_1 + logabsdet_2 + logabsdet_2
            print('Real logdet:', real_logabsdet.detach())
            print('Estimated logabsdet', logabsdet.mean().detach())
            logabsdet_error[cnt_val] = torch.abs(real_logabsdet - logabsdet.mean())
            cnt_val += 1

    # Plot training loss;
    plt.subplot(2, 1, 1)
    plt.title('orthogonal error (training loss).')
    plt.plot(train_loss)
    plt.subplot(2, 1, 2)
    plt.title('logabsdet error.')
    plt.plot(logabsdet_error)
    plt.show()