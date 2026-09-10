from typing import List

import numpy as np
import torch
from torch import nn

from .activation import switchable_activation
from .mlp import MLP


class subamp_MLP(nn.Module):
    """
    Split the regression of the squared amplitude into subamplitudes. Each subamplitude is learned by a separate MLP. In the end, the subamplitudes are summed and squared to obtain the toal amplitude.
    """

    def __init__(
        self,
        n_features,
        type_token_list,
        hidden_channels,
        hidden_layers,
        out_shape=1,
        num_subamps=1,
        activation="gelu",
        transforms=None,
        fv_input=True,
        num_groups=1,
        dropout_prob=None,
    ):
        super().__init__()

        self.subamps = nn.ModuleList()
        for i in range(num_subamps):
            self.subamps.append(
                MLP(
                    n_features,
                    type_token_list,
                    hidden_channels,
                    hidden_layers,
                    out_shape,
                    activation,
                    transforms,
                    fv_input,
                    num_groups,
                    dropout_prob,
                )
            )

    def forward(self, inputs: torch.Tensor):
        """Forward pass of subamp MLP."""
        subamps = [subamp(inputs) for subamp in self.subamps]
        # sum subamps and square
        return torch.stack(subamps).sum(dim=0) ** 2
