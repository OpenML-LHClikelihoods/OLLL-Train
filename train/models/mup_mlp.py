from typing import List

import numpy as np
import torch
from torch import nn
import mup  # Microsoft μP library
from mup import MuReadout, set_base_shapes, make_base_shapes

from .activation import switchable_activation


class MuMLP(nn.Module):
    """A μP-aware MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(
        self,
        n_features,
        type_token_list,
        hidden_channels,
        hidden_layers,
        out_shape=1,
        activation="gelu",
        transforms=None,
        fv_input=True,
        num_groups=1,
        dropout_prob=None,
        gain=1.0,
        loss='MSE',
        init_bias=True,
        batchnorm=False,
        BN_eps=1e-5, #default PyTorch value
        BN_momentum=0.1, #default PyTorch value
        out_soft_bound=False,
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = n_features
        self.out_shape = out_shape
        self.loss = loss

        # Optional soft saturation of the *mean* outputs, in preprocessed target
        # space. The nLL preprocessings are exponential to invert (expm1 for
        # log_w_negatives, sinh for asinh), so a single runaway prediction comes
        # back as a physically absurd nLL: on the 2018-16 EWkino boundary scan a
        # handful of otherwise-unremarkable rows decoded to ~6e6 against a truth
        # max of 2460.
        #
        # This is NOT a clamp. Inside the training range the map is exactly the
        # identity, so gradients and in-distribution predictions are untouched;
        # outside it grows logarithmically instead of linearly, which turns the
        # exponential inverse into a polynomial one. Extrapolation still works
        # and stays strictly ordered -- a point further out of distribution still
        # gets a larger nLL -- it just cannot explode. A hard clip would instead
        # pin everything beyond the edge to one value, making "just outside" and
        # "wildly outside" indistinguishable to the consumer.
        #
        # Applied here rather than in the consumer so it is traced into the
        # exported ONNX graph and protects everyone. Bounds are filled from the
        # TRAINING targets by set_output_clamp(); until then they are +-inf and
        # the map is a no-op.
        # persistent=False keeps these out of state_dict: every checkpoint
        # written before this existed still loads, and the bounds are anyway
        # re-derived from the training data on each init_model, so storing them
        # would only risk a stale copy overriding the live one.
        self.out_soft_bound = out_soft_bound
        self.register_buffer("out_lo", torch.full((self.out_shape,), float("-inf")),
                             persistent=False)
        self.register_buffer("out_hi", torch.full((self.out_shape,), float("inf")),
                             persistent=False)

        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(np.prod(self.in_shape), hidden_channels))
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        if batchnorm: 
            layers.append(nn.BatchNorm1d(hidden_channels, eps=BN_eps, momentum=BN_momentum))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(switchable_activation(activation=activation, num_groups=1))
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))
            if batchnorm: 
                layers.append(nn.BatchNorm1d(hidden_channels, eps=BN_eps, momentum=BN_momentum))

        # Last activation
        layers.append(switchable_activation(activation=activation, num_groups=1))

        # Output layer
        if self.loss in ('HETEROSC', 'MAE_TO_HETEROSC'):
            layers.append(MuReadout(hidden_channels, np.prod(2 * self.out_shape)))
        else:
            layers.append(MuReadout(hidden_channels, np.prod(self.out_shape)))

        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(lambda m: self.init_weights(m, activation=activation, init_bias=init_bias))

    @staticmethod
    def init_weights(m, activation='gelu', init_bias=True):
        """
        μP-compatible initialization for MLP layers.

        Args:
            m: layer (nn.Linear or MuReadout)
            activation: activation function name ('gelu', 'relu', etc.)
            init_bias: whether to initialize biases
        """
        if isinstance(m, MuReadout):
            # Readout layer: weights and biases initialized to zero
            if m.weight is not None:
                nn.init.zeros_(m.weight)
            if (m.bias is not None) and init_bias:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Hidden layers: Kaiming normal with activation gain
            #gain = nn.init.calculate_gain(activation)
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            #m.weight.data.mul_(gain)

            # Bias initialization
            if (m.bias is not None): #and init_bias:
                fan_in = m.weight.size(1)
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(m.bias, -bound, bound)

    def set_output_clamp(self, lo, hi):
        """Pin the soft-saturation knees (preprocessed space), per output column.

        Called once after the data is prepared, with the per-column min/max of
        the TRAINING targets. No margin is needed: the map is exactly the
        identity between lo and hi, so training points are never distorted.
        """
        lo = torch.as_tensor(lo, dtype=self.out_lo.dtype, device=self.out_lo.device)
        hi = torch.as_tensor(hi, dtype=self.out_hi.dtype, device=self.out_hi.device)
        assert lo.shape == self.out_lo.shape, f"clamp lo {lo.shape} != {self.out_lo.shape}"
        assert (hi > lo).all(), "output clamp needs hi > lo on every column"
        self.out_lo.copy_(lo)
        self.out_hi.copy_(hi)

    def _clamp_means(self, means: torch.Tensor) -> torch.Tensor:
        """Identity on [out_lo, out_hi], logarithmic outside.

            x                       lo <= x <= hi
            hi + log1p(x - hi)      x > hi
            lo - log1p(lo - x)      x < lo

        Continuous and C1 at both knees (d/dx log1p(x-hi) = 1 at x = hi), and
        strictly increasing everywhere, so the inverse preprocessing still maps
        distinct predictions to distinct, correctly ordered nLLs.
        """
        if not self.out_soft_bound:
            return means
        over = torch.clamp(means - self.out_hi, min=0.0)
        under = torch.clamp(self.out_lo - means, min=0.0)
        inside = torch.minimum(torch.maximum(means, self.out_lo), self.out_hi)
        return inside + torch.log1p(over) - torch.log1p(under)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of μP-aware MLP."""
        x = self.mlp(inputs)

        if self.loss in ('HETEROSC', 'MAE_TO_HETEROSC'):
            # Split last `out_shape` dimensions, apply softplus for positivity
            x_sigmas = torch.max(torch.nn.functional.softplus(x[:, -self.out_shape:]),
                                 torch.tensor(1e-15, device=x.device))
            x = torch.cat((self._clamp_means(x[:, :-self.out_shape]), x_sigmas), dim=1)
            return x
        else:
            return self._clamp_means(x)