from typing import List

import numpy as np
import torch
from torch import nn

from .activation import switchable_activation


class MLP(nn.Module):
    """A simple baseline MLP.

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
        loss='MSE',
        batchnorm=False,
        BN_eps=1e-5, #default PyTorch value
        BN_momentum=0.1, #default PyTorch value
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = n_features
        self.out_shape = out_shape
        self.loss = loss
        layers: List[nn.Module] = [nn.Linear(np.prod(self.in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_channels, eps=BN_eps, momentum=BN_momentum))
        for _ in range(hidden_layers - 1):
            layers.append(switchable_activation(activation=activation, num_groups=1))
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))
            if batchnorm: 
                layers.append(nn.BatchNorm1d(hidden_channels, eps=BN_eps, momentum=BN_momentum))

        layers.append(switchable_activation(activation=activation, num_groups=1))
        if self.loss in ('HETEROSC', 'MAE_TO_HETEROSC'):
            layers.append(nn.Linear(hidden_channels, np.prod(2*self.out_shape)))
        else:
            layers.append(nn.Linear(hidden_channels, np.prod(self.out_shape)))
        self.mlp = nn.Sequential(*layers)
        #initialize sigma output layer to small positive values
        # if self.loss == 'HETEROSC':
        #     torch.nn.init.constant_(self.mlp[-1].weight[-self.out_shape:], 0.1)
        #     torch.nn.init.constant_(self.mlp[-1].bias[-self.out_shape:], 1.0)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of baseline MLP."""

        if self.loss in ('HETEROSC', 'MAE_TO_HETEROSC'):
            #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            x = self.mlp(inputs)    # Separate the last 4 outputs and process them
            #print(self.out_shape)
            #print('x.shape:', x.shape)
            x_sigmas = torch.max(torch.nn.functional.softplus(x[:, -self.out_shape:]),
                                 torch.tensor(1e-15, device=x.device))
            #print('x_sigmas:', x_sigmas)
            # Combine the unmodified first outputs with the processed last 4 outputs
            x = torch.cat((x[:, :-self.out_shape], x_sigmas), dim=1)
            #print('x:', x)
            return x
        else:
            return self.mlp(inputs)