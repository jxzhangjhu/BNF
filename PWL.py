import torch
import torch.nn as nn
import torch.functional as F

import numpy as np

import matplotlib.pyplot as plt

def get_monotonicity(monotonicity, num_channels):
    if isinstance(monotonicity, (int, float)):
        if not monotonicity in (-1, 0, 1):
            raise ValueError("monotonicity must be one of -1, 0, +1")
        return monotonicity * torch.ones(num_channels)
    else:
        if not (isinstance(monotonicity, torch.Tensor) and list(monotonicity.shape) == [num_channels]):
            raise ValueError("monotonicity must be either an int or a tensor with shape [num_channels]")
        if not torch.all(
                torch.eq(monotonicity, 0) | torch.eq(monotonicity, 1) | torch.eq(monotonicity, -1)
        ).item():
            raise ValueError("monotonicity must be one of -1, 0, +1")
        return monotonicity.float()


class BasePWL(torch.nn.Module):
    def __init__(self, num_breakpoints):
        super(BasePWL, self).__init__()
        if not num_breakpoints >= 1:
            raise ValueError(
                "Piecewise linear function only makes sense when you have 1 or more breakpoints."
            )
        self.num_breakpoints = num_breakpoints

    def slope_at(self, x):
        dx = 1e-3
        return -(self.forward(x) - self.forward(x + dx)) / dx


class BasePWLX(BasePWL):
    def __init__(self, num_channels, num_breakpoints, num_x_points):
        super(BasePWLX, self).__init__(num_breakpoints)
        self.num_channels = num_channels
        self.num_x_points = num_x_points
        self.x_positions = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_x_points))
        self._reset_x_points()

    def _reset_x_points(self):
        torch.nn.init.normal_(self.x_positions, std=2.0)

    def get_x_positions(self):
        return self.x_positions

    def get_sorted_x_positions(self):
        return torch.sort(self.get_x_positions(), dim=1)[0]

    def get_spreads(self):
        sorted_x_positions = self.get_sorted_x_positions()
        return (torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions)[:, :-1]

    def unpack_input(self, x):
        shape = list(x.shape)
        if len(shape) == 2:
            return x
        elif len(shape) < 2:
            raise ValueError(
                "Invalid input, the input to the PWL module must have at least 2 dimensions with channels at dimension dim(1)."
            )
        assert shape[1] == self.num_channels, (
                "Invalid input, the size of dim(1) must be equal to num_channels (%d)" % self.num_channels
        )
        x = torch.transpose(x, 1, len(shape) - 1)
        assert x.shape[-1] == self.num_channels
        return x.reshape(-1, self.num_channels)

    def repack_input(self, unpacked, old_shape):
        old_shape = list(old_shape)
        if len(old_shape) == 2:
            return unpacked
        transposed_shape = old_shape[:]
        transposed_shape[1] = old_shape[-1]
        transposed_shape[-1] = old_shape[1]
        unpacked = unpacked.view(*transposed_shape)
        return torch.transpose(unpacked, 1, len(old_shape) - 1)


class BaseSlopedPWL(BasePWLX):
    def get_biases(self):
        raise NotImplementedError()

    def get_slopes(self):
        raise NotImplementedError()

    def forward(self, x):
        old_shape = x.shape
        x = self.unpack_input(x)
        bs = x.shape[0]
        sorted_x_positions = self.get_sorted_x_positions()
        skips = torch.roll(sorted_x_positions, shifts=-1, dims=1) - sorted_x_positions
        slopes = self.get_slopes()
        skip_deltas = skips * slopes[:, 1:]
        biases = self.get_biases().unsqueeze(1)
        cumsums = torch.cumsum(skip_deltas, dim=1)[:, :-1]

        betas = torch.cat([biases, biases, cumsums + biases], dim=1)
        breakpoints = torch.cat([sorted_x_positions[:, 0].unsqueeze(1), sorted_x_positions], dim=1)

        # find the index of the first breakpoint smaller than x
        # TODO(pdabkowski) improve the implementation
        s = x.unsqueeze(2) - sorted_x_positions.unsqueeze(0)
        # discard larger breakpoints
        s = torch.where(s < 0, torch.tensor(float("inf"), device=x.device), s)
        b_ids = torch.where(
            sorted_x_positions[:, 0].unsqueeze(0) <= x,
            torch.argmin(s, dim=2) + 1,
            torch.tensor(0, device=x.device),
        ).unsqueeze(2)

        selected_betas = torch.gather(betas.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids).squeeze(2)
        selected_breakpoints = torch.gather(
            breakpoints.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids
        ).squeeze(2)
        selected_slopes = torch.gather(slopes.unsqueeze(0).expand(bs, -1, -1), dim=2, index=b_ids).squeeze(2)
        cand = selected_betas + (x - selected_breakpoints) * selected_slopes
        return self.repack_input(cand, old_shape)


class PWL(BaseSlopedPWL):
    r"""Piecewise Linear Function (PWL) module.
    The module takes the Tensor of (N, num_channels, ...) shape and returns the processed Tensor of the same shape.
    Each entry in the input tensor is processed by the PWL function. There are num_channels separate PWL functions,
    the PWL function used depends on the channel.
    The x coordinates of the breakpoints are initialized randomly from the Gaussian with std of 2. You may want to
    use your own custom initialization depending on the use-case as the optimization is quite sensitive to the
    initialization of breakpoints. As long as your data is normalized (zero mean, unit variance) the default
    initialization should be fine.
    Arguments:
        num_channels (int): number of channels (or features) that this PWL should process. Each channel
            will get its own PWL function.
        num_breakpoints (int): number of PWL breakpoints. Total number of segments constructing the PWL is
            given by num_breakpoints + 1. This value is shared by all the PWL channels in this module.
    """

    def __init__(self, num_channels, num_breakpoints):
        super(PWL, self).__init__(num_channels, num_breakpoints, num_x_points=num_breakpoints)
        self.slopes = torch.nn.Parameter(torch.Tensor(self.num_channels, self.num_breakpoints + 1))
        self.biases = torch.nn.Parameter(torch.Tensor(self.num_channels))
        self._reset_params()

    def _reset_params(self):
        BasePWLX._reset_x_points(self)
        torch.nn.init.ones_(self.slopes)
        with torch.no_grad():
            self.biases.copy_(self.get_sorted_x_positions()[:, 0])

    def get_biases(self):
        return self.biases

    def get_x_positions(self):
        return self.x_positions

    def get_slopes(self):
        return self.slopes


class MonoPWL(PWL):
    r"""Piecewise Linear Function (PWL) module with the monotonicity constraint.
    The module takes the Tensor of (N, num_channels, ...) shape and returns the processed Tensor of the same shape.
    Each entry in the input tensor is processed by the PWL function. There are num_channels separate PWL functions,
    the PWL function used depends on the channel. Each PWL is guaranteed to have the requested monotonicity.
    The x coordinates of the breakpoints are initialized randomly from the Gaussian with std of 2. You may want to
    use your own custom initialization depending on the use-case as the optimization is quite sensitive to the
    initialization of breakpoints. As long as your data is normalized (zero mean, unit variance) the default
    initialization should be fine.
    Arguments:
        num_channels (int): number of channels (or features) that this PWL should process. Each channel
            will get its own PWL function.
        num_breakpoints (int): number of PWL breakpoints. Total number of segments constructing the PWL is
            given by num_breakpoints + 1. This value is shared by all the PWL channels in this module.
        monotonicity (int, Tensor): Monotonicty constraint, the monotonicity can be either +1 (increasing),
            0 (no constraint) or -1 (decreasing). You can provide either an int to set the constraint
            for all the channels or a long Tensor of shape [num_channels]. All the entries must be in -1, 0, +1.
    """

    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(MonoPWL, self).__init__(num_channels=num_channels, num_breakpoints=num_breakpoints)
        self.register_buffer("monotonicity", get_monotonicity(monotonicity, self.num_channels))
        self.epsilon = 1e-4
        with torch.no_grad():
            mono_mul = torch.where(
                torch.eq(self.monotonicity, 0.0),
                torch.tensor(1.0, device=self.monotonicity.device),
                self.monotonicity,
            )
            self.biases.copy_(self.biases * mono_mul)

    def get_slopes(self):
        return torch.where(
            torch.eq(self.monotonicity, 0.0).unsqueeze(1),
            self.slopes,
            torch.abs(self.slopes) * self.monotonicity.unsqueeze(1),
        ) + self.epsilon
