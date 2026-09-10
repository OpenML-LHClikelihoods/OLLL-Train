import torch
from torch import nn
import numpy as np

from models import MLP


class DSI(nn.Module):
    """
    Deep set + invariant approach
    This network combines implicit bias from permutation invariance (deep sets)
    and Lorentz invariance (Lorentz inner products),
    but in a way that breaks both invariances when combining the two approaches

    There are two types of MLP networks:
    - prenet: MLP (one for each particle type) that processes fourmomenta,
      the results are combined in a deep set to form a permutation-invariant result
      This can be viewed as an optional preprocessing step
    - net: MLP that combines the deep set result with Lorentz invariants
      to extract the final result
    """

    def __init__(
        self,
        n_features,
        type_token_list,
        hidden_channels_prenet,
        hidden_layers_prenet,
        out_dim_prenet_sep,
        hidden_channels_net,
        hidden_layers_net,
        sum_deepset=True,
        activation="gelu",
        num_groups=1,
        dropout_prob=None,
    ):
        """
        Parameters
        ----------
        type_token_list : List[int]
            List of particles in the process, with an integer representing the particle type
            Example: [0,0,1,2,2] for q q > Z g g
        hidden_channels_prenet : int
        hidden_layers_prenet: int
        out_dim_prenet_sep : int
            Size of the latent space extract from the deep set
        hidden_channels_net : int
        hidden_layers_net : int
        sum_deepset : bool
            whether to sum the deep set embeddings or concatenate them
            Permutation invariance is broken anyways if use_invariants=True,
            so one can also decide to break it at an earlier stage
        dropout_prob : float
        """
        super().__init__()
        self.sum_deepset = sum_deepset

        self.n_particles = len(type_token_list)
        self.n_particle_types = len(np.unique(type_token_list))
        self.type_token_list = type_token_list
        fvs_features = 4 * self.n_particles
        self.out_shape = 1

        self.prenets = nn.ModuleList(
            [
                MLP(
                    n_features=4,
                    type_token_list=type_token_list,
                    out_shape=out_dim_prenet_sep,
                    hidden_channels=hidden_channels_prenet,
                    hidden_layers=hidden_layers_prenet,
                    dropout_prob=dropout_prob,
                )
                for _ in range(max(type_token_list) + 1)
            ]
        )

        if self.sum_deepset:
            n_features_net = (
                n_features - fvs_features + out_dim_prenet_sep * self.n_particle_types
            )
        else:
            n_features_net = (
                n_features - fvs_features + out_dim_prenet_sep * self.n_particle
            )
        self.net = MLP(
            n_features=n_features_net,
            type_token_list=type_token_list,
            out_shape=1,
            hidden_channels=hidden_channels_net,
            hidden_layers=hidden_layers_net,
            dropout_prob=dropout_prob,
            activation=activation,
            num_groups=num_groups,
        )

    def forward(self, x, type_token):
        FVinputs = x[:, :, -4 * self.n_particles :]
        nonFVinputs = x[:, :, : -4 * self.n_particles]

        # deep set preprocessing
        assert len(type_token) == 1
        type_token = type_token[0]
        assert type_token.cpu().numpy().tolist() == self.type_token_list
        preprocessing = []
        for i in range(max(type_token) + 1):
            ps = FVinputs.reshape(
                FVinputs.shape[0], FVinputs.shape[1], FVinputs.shape[2] // 4, 4
            )
            identical_particles = ps[..., type_token == i, :]
            embedding = self.prenets[i](identical_particles)
            embedding = (
                embedding.sum(dim=-2, keepdim=True) if self.sum_deepset else embedding
            )
            preprocessing.append(embedding)
        preprocessing = torch.cat(preprocessing, dim=-2)
        preprocessing = preprocessing.view(*FVinputs.shape[:-1], -1)

        # combine everything
        latent_full = torch.cat((preprocessing, nonFVinputs), dim=-1)
        result = self.net(latent_full)
        return result
