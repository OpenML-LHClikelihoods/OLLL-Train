import torch.nn as nn
#from katrational import KAT_Group


def switchable_activation(activation: str = 'gelu', num_groups: int = 1):
    match activation:
        case 'relu':
            return nn.ReLU()
        case 'gelu':
            return nn.GELU()
        case 'swish':
            return nn.SiLU()
        case 'mish':
            return nn.Mish()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'tanh':
            return nn.Tanh()
        case 'elu':
            return nn.ELU()
        case 'KAN':
            return KAT_Group(mode="gelu", num_groups=num_groups)
        case _:
            raise ValueError(
                f"Activation function {activation} not supported")
