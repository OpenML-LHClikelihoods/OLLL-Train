from typing import List

import numpy as np
import torch
from torch import nn

from .activation import switchable_activation
from preprocessing import compute_invariants


class LogLayer(nn.Module):
    """A simple log layer."""

    def __init__(self):
        super(LogLayer, self).__init__()

    def forward(self, x):
        return torch.log(x)


class FVLinearLayer(nn.Module):
    """A simple linear layer for FVs."""

    def __init__(self, n_particles, hidden_fv_channels):
        super(FVLinearLayer, self).__init__()
        self.n_particles = n_particles
        self.hidden_fv_channels = hidden_fv_channels
        self.linear = nn.Linear(n_particles, hidden_fv_channels, bias=False)

        # Custom initialization function
        # self._initialize_weights()

    def forward(self, x):
        particles = x[:, -4 * self.n_particles :].reshape(-1, self.n_particles, 4)
        particles = particles.permute(
            0, 2, 1
        )  # permute to apply linear layer at particle level
        res = self.linear(particles)
        # res = particles
        return res.permute(0, 2, 1)

    # def _initialize_weights(self):
    #     """Initialize weights with indentity matrix"""
    #     nn.init.eye_(self.linear.weight)


class SPLayer(nn.Module):
    """A simple scalar product layer."""

    def __init__(self):
        super(SPLayer, self).__init__()

    def forward(self, x):
        return compute_invariants(x, reshape=False)


class FV_MLP(nn.Module):
    """FV MLP

    Builds linear combinations of FVs, and then calculates scalar products of these FVs. The scalar products are then used as input to a MLP which can also take additional features as input.
    """

    def __init__(
        self,
        n_features,
        type_token_list,
        hidden_channels,
        hidden_fv_channels,
        hidden_layers,
        activation="gelu",
        transforms=None,
        fv_input=True,
        num_groups=1,
        dropout_prob=None,
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.n_particles = len(type_token_list)
        fvs_features = 4 * self.n_particles
        self.out_shape = 1

        fv_layers: List[nn.Module] = []
        # build linear combinations of FVs (without bias)
        fv_layers.append(
            FVLinearLayer(
                n_particles=self.n_particles, hidden_fv_channels=hidden_fv_channels
            )
        )
        # calculate scalar products
        fv_layers.append(SPLayer())
        n_invariants = hidden_fv_channels * (hidden_fv_channels - 1) // 2
        # apply log to the scalar products
        fv_layers.append(LogLayer())
        # batch normalization
        fv_layers.append(nn.BatchNorm1d(n_invariants))
        self.inv_nn = nn.Sequential(*fv_layers)

        self.in_shape = n_invariants + n_features - fvs_features
        layers: List[nn.Module] = [nn.Linear(np.prod(self.in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(
                switchable_activation(activation=activation, num_groups=num_groups)
            )
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(
            switchable_activation(activation=activation, num_groups=num_groups)
        )
        layers.append(nn.Linear(hidden_channels, np.prod(self.out_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward pass of baseline MLP."""
        assert x.shape[0] == 1, "FV_MLP only supports one dataset"
        invs = self.inv_nn(x[0])
        invs = invs.reshape(-1, invs.shape[0], invs.shape[1])
        mlp_inputs = torch.cat(
            [x[:, :, : -4 * self.n_particles], invs], dim=2
        )  # remove FVs from input and add invariants
        return self.mlp(mlp_inputs)
