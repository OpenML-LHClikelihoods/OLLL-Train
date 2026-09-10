import functools
from typing import Optional, Tuple, Union, List, Sequence

import torch

from torch import Tensor
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

import opt_einsum

# from xformers.ops import AttentionBias

from misc import to_nd

from .activation import switchable_activation


def custom_einsum(
    equation: str, *operands: torch.Tensor, path: List[int]
) -> torch.Tensor:
    """Computes einsum with a custom contraction order."""

    # Justification: For the sake of performance, we need direct access to torch's private methods.

    # pylint:disable-next=protected-access
    return torch._VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]

@functools.lru_cache(maxsize=None)
def _get_cached_path_for_equation_and_shapes(
    equation: str, op_shape: Sequence[torch.Tensor]
) -> List[int]:
    """Provides caching of optimal path."""
    tupled_path = opt_einsum.contract_path(
        equation, *op_shape, optimize="optimal", shapes=True
    )[0]

    return [item for pair in tupled_path for item in pair]

def cached_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Computes einsum with a cached optimal contraction.

    Inspired by upstream
    https://github.com/pytorch/pytorch/blob/v1.13.0/torch/functional.py#L381.
    """
    op_shape = tuple(op.shape for op in operands)
    path = _get_cached_path_for_equation_and_shapes(
        equation=equation, op_shape=op_shape
    )

    return custom_einsum(equation, *operands, path=path)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal=False,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Dynamically dispatch to xFormers if attn_mask is an instance of xformers.ops.AttentionBias
    or FORCE_XFORMERS is set, use torch otherwise.

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Union[AttentionBias, Tensor]]
        Attention mask
    is_causal: bool

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    # if FORCE_XFORMERS or isinstance(attn_mask, AttentionBias):
    #     assert (
    #         not is_causal
    #     ), "is_causal=True not implemented yet for xformers attention"
    #     if key.shape[1] != query.shape[1]:  # required to make multi_query work
    #         key = key.expand(key.shape[0], query.shape[1], *key.shape[2:])
    #         value = value.expand(value.shape[0], query.shape[1], *value.shape[2:])
    #     query = query.transpose(
    #         1, 2
    #     )  # [batch, head, item, d] -> [batch, item, head, d]
    #     key = key.transpose(1, 2)
    #     value = value.transpose(1, 2)
    #     out = memory_efficient_attention(
    #         query.contiguous(),
    #         key.contiguous(),
    #         value,
    #         attn_bias=attn_mask,
    #     )
    #     out = out.transpose(1, 2)  # [batch, item, head, d] -> [batch, head, item, d]
    #     return out
    return torch_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)


class ApplyRotaryPositionalEncoding(torch.nn.Module):
    """Applies rotary position encodings (RoPE) to scalar tensors.

    References
    ----------
    Jianlin Su et al, "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        arXiv:2104.09864

    Parameters
    ----------
    num_channels : int
        Number of channels (key and query size).
    item_dim : int
        Embedding dimension. Should be even.
    base : int
        Determines the frequencies.
    """

    def __init__(self, num_channels, item_dim, base=4096):
        super().__init__()

        assert (
            num_channels % 2 == 0
        ), "Number of channels needs to be even for rotary position embeddings"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, num_channels, 2).float() / num_channels)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.device_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.item_dim = item_dim
        self.num_channels = num_channels

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """Computes rotary embeddings along `self.item_dim` and applies them to inputs.

        The inputs are usually scalar queries and keys.

        Assumes that the last dimension is the feature dimension (and is thus not suited
        for multivector data!).

        Parameters
        ----------
        scalars : torch.Tensor of shape (..., num_channels)
            Input data. The last dimension is assumed to be the channel / feature dimension
            (NOT the 16 dimensions of a multivector).

        Returns
        -------
        outputs : torch.Tensor of shape (..., num_channels)
            Output data. Rotary positional embeddings applied to the input tensor.
        """

        # Check inputs
        assert scalars.shape[-1] == self.num_channels

        # Compute embeddings, if not already cached
        self._compute_embeddings(scalars)

        # Apply embeddings
        outputs = (
            scalars * self.cos_cached + self._rotate_half(scalars) * self.sin_cached
        )

        return outputs

    def _compute_embeddings(self, inputs):
        """Computes position embeddings and stores them.

        The position embedding is computed along dimension `item_dim` of tensor `inputs`
        and is stored in `self.sin_cached` and `self.cos_cached`.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data.
        """
        seq_len = inputs.shape[self.item_dim]
        if seq_len != self.seq_len_cached or inputs.device != self.device_cached:
            self.seq_len_cached = seq_len
            self.device_cached = inputs.device
            t = torch.arange(inputs.shape[self.item_dim], device=inputs.device).type_as(
                self.inv_freq
            )
            freqs = cached_einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(inputs.device)

            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

            # Insert appropriate amount of dimensions such that the embedding correctly enumerates
            # along the item dim
            item_dim = (
                self.item_dim if self.item_dim >= 0 else inputs.ndim + self.item_dim
            )  # Deal with item_dim < 0
            for _ in range(item_dim + 1, inputs.ndim - 1):
                self.cos_cached = self.cos_cached.unsqueeze(1)
                self.sin_cached = self.sin_cached.unsqueeze(1)

    @staticmethod
    def _rotate_half(inputs):
        """Utility function that "rotates" a tensor, as required for rotary embeddings."""
        x1, x2 = (
            inputs[..., : inputs.shape[-1] // 2],
            inputs[..., inputs.shape[-1] // 2 :],
        )
        return torch.cat((-x2, x1), dim=-1)
    
class BaselineLayerNorm(nn.Module):
    """Baseline layer norm over all dimensions except the first."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        outputs : Tensor
            Normalized inputs.
        """
        return torch.nn.functional.layer_norm(
            inputs, normalized_shape=inputs.shape[-1:]
        )


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_channels, 3 * hidden_channels * num_heads)

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels * num_heads)
        q, k, v = rearrange(
            qkv,
            "... items (qkv hidden_channels num_heads) -> qkv ... num_heads items hidden_channels",
            num_heads=self.num_heads,
            qkv=3,
        )
        return q, k, v


class MultiQueryQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-query attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_linear = nn.Linear(in_channels, hidden_channels)
        self.v_linear = nn.Linear(in_channels, hidden_channels)

    def forward(self, inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        q = rearrange(
            self.q_linear(inputs),
            "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
            num_heads=self.num_heads,
        )
        k = self.k_linear(inputs)[
            ..., None, :, :
        ]  # (..., head=1, item, hidden_channels)
        v = self.v_linear(inputs)[..., None, :, :]
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_enc_base: int = 4096,
        multi_query: bool = True,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

        # Optional positional encoding
        if pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                hidden_channels, item_dim=-2, base=pos_enc_base
            )
        else:
            self.pos_encoding = None

        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(
            inputs
        )  # each: (..., num_heads, num_items, num_channels, 16)

        # Rotary positional encoding
        if self.pos_encoding is not None:
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        # Attention layer
        h = self._attend(q, k, v, attention_mask, is_causal=is_causal)

        # Concatenate heads and transform linearly
        h = rearrange(
            h,
            "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)",
        )
        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs

    @staticmethod
    def _attend(q, k, v, attention_mask=None, is_causal=False):
        """Scaled dot-product attention."""

        # Add batch dimension if needed
        bh_shape = q.shape[:-2]
        q = to_nd(q, 4)
        k = to_nd(k, 4)
        v = to_nd(v, 4)

        # SDPA
        outputs = scaled_dot_product_attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q),
            attn_mask=attention_mask,
            is_causal=is_causal,
        )

        # Return batch dimensions to inputs
        outputs = outputs.view(*bh_shape, *outputs.shape[-2:])

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another LayerNorm, an
    item-wise two-layer MLP with GeLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        channels,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        increase_hidden_channels=1,
        multi_query: bool = True,
        activation="gelu",
        num_groups=1,
        dropout_prob=None,
    ) -> None:
        super().__init__()

        self.norm = BaselineLayerNorm()

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        hidden_channels = channels // num_heads * increase_hidden_channels
        if pos_encoding:
            hidden_channels = (hidden_channels + 1) // 2 * 2
            hidden_channels = max(hidden_channels, 16)

        self.attention = BaselineSelfAttention(
            channels,
            channels,
            hidden_channels,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            pos_enc_base=pos_encoding_base,
            multi_query=multi_query,
            dropout_prob=dropout_prob,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
            switchable_activation(activation=activation, num_groups=1),
            nn.Linear(2 * channels, channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask=None, is_causal=False
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm(inputs)
        h = self.attention(h, attention_mask=attention_mask, is_causal=is_causal)
        outputs = inputs + h

        # Residual MLP
        h = self.norm(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs


class Transformer(nn.Module):
    """Baseline transformer.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        n_features: int,
        hidden_channels: int,
        out_channels: int = 1,
        num_blocks: int = 10,
        num_heads: int = 8,
        pos_encoding: bool = True,
        pos_encoding_base: int = 4096,
        checkpoint_blocks: bool = False,
        increase_hidden_channels=1,
        multi_query: bool = False,
        activation="gelu",
        num_groups=1,
        dropout_prob=None,
        type_token_list=None,
    ) -> None:
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.linear_in = nn.Linear(n_features, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encoding,
                    pos_encoding_base=pos_encoding_base,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query,
                    activation=activation,
                    num_groups=num_groups,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self, inputs: torch.Tensor, attention_mask=None, is_causal=False
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items, num_channels)
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask
        is_causal: bool

        Returns
        -------
        outputs : Tensor with shape (..., num_items, num_channels)
            Outputs
        """
        h = self.linear_in(inputs)
        for block in self.blocks:
            if self.checkpoint_blocks:
                fn = partial(block, attention_mask=attention_mask, is_causal=is_causal)
                h = checkpoint(fn, h)
            else:
                h = block(h, attention_mask=attention_mask, is_causal=is_causal)
        outputs = self.linear_out(h)
        return outputs


class AxialTransformer(nn.Module):
    """Baseline axial transformer for data with two token dimensions.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Assumes input data with shape `(..., num_items_1, num_items_2, num_channels, [16])`.

    The first, third, fifth, ... block computes attention over the `items_2` axis. The other blocks
    compute attention over the `items_1` axis. Positional encoding can be specified separately for
    both axes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    pos_encodings : tuple of bool
        Whether to apply rotary positional embeddings along the item dimensions to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int = 20,
        num_heads: int = 8,
        pos_encodings: Tuple[bool, bool] = (False, False),
        pos_encoding_base: int = 4096,
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encodings[(block + 1) % 2],
                    pos_encoding_base=pos_encoding_base,
                )
                for block in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items1, num_items2, num_channels)
            Input data

        Returns
        -------
        outputs : Tensor with shape (..., num_items1, num_items2, num_channels)
            Outputs
        """

        rearrange_pattern = "... i j c -> ... j i c"

        h = self.linear_in(inputs)

        for i, block in enumerate(self.blocks):
            # For first, third, ... block, we want to perform attention over the first token
            # dimension. We implement this by transposing the two item dimensions.
            if i % 2 == 1:
                h = rearrange(h, rearrange_pattern)

            h = block(h)

            # Transposing back to standard axis order
            if i % 2 == 1:
                h = rearrange(h, rearrange_pattern)

        outputs = self.linear_out(h)

        return outputs