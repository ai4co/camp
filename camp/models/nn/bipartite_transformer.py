from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv

log = get_pylogger(__name__)


class MultiHeadAttention_with_graph(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.attn = MultiHeadAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
        self.graph_layers = nn.ModuleList(
            [
                TransformerConv(
                    (embed_dim, embed_dim), embed_dim, dropout=0.05, edge_dim=embed_dim
                )
                for _ in range(1)
            ]
        )
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, node_emb, edge_emb, pyg_data, attn_mask=None):
        """x: (batch, num_agent + num_client, hidden_dim) (where hidden_dim = num heads * head dim)
        edge_emb: (batch, num_agent, num_client, hidden_dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        b, m, mn, d = edge_emb.size()

        n = mn - m

        agent_emb_init = node_emb[:, :m, :]
        customer_emb_init = node_emb[:, m:, :]

        agent_emb = agent_emb_init.clone()
        customer_emb = customer_emb_init.clone()

        agent_emb = agent_emb.reshape(-1, agent_emb.size(-1))
        customer_emb = customer_emb.reshape(-1, customer_emb.size(-1))

        edge_emb = rearrange(edge_emb, "b m mn d -> (b m) mn d")

        edge_emb = self.attn(edge_emb)
        edge_emb = rearrange(edge_emb, "(b m) mn d -> b m mn d", m=m)
        edge_emb_init = edge_emb.clone()

        pyg_data.edge_attr = edge_emb[:, :, m:, :].reshape(-1, edge_emb.size(-1))
        edge_index_reverse = torch.stack(
            [pyg_data.edge_index[1], pyg_data.edge_index[0]]
        )

        # Bi-partite graph layer
        for graph_layer in self.graph_layers:
            customer_emb_next = graph_layer(
                (agent_emb, customer_emb),
                edge_index=pyg_data.edge_index,
                edge_attr=pyg_data.edge_attr,
            )
            pyg_data.edge_attr = (
                edge_emb[:, :, m:, :].transpose(1, 2).reshape(-1, edge_emb.size(-1))
            )
            agent_emb = graph_layer(
                (customer_emb, agent_emb),
                edge_index=edge_index_reverse,
                edge_attr=pyg_data.edge_attr,
            )
            customer_emb = customer_emb_next

        agent_emb = agent_emb.reshape(agent_emb_init.size())
        customer_emb = customer_emb.reshape(customer_emb_init.size())
        agent_emb = agent_emb + agent_emb_init
        customer_emb = customer_emb + customer_emb_init

        out = torch.cat([agent_emb, customer_emb], dim=1)

        expanded_agent_emb = agent_emb.unsqueeze(2).expand(-1, -1, n, -1)
        customer_emb = customer_emb.unsqueeze(1).expand(-1, m, -1, -1)
        customer_emb = expanded_agent_emb + customer_emb
        customer_emb = self.out_proj(customer_emb)
        edge_emb = torch.cat(
            [agent_emb.unsqueeze(1).expand(-1, m, -1, -1), customer_emb], dim=2
        )
        edge_emb = edge_emb + edge_emb_init

        return out, edge_emb


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": RMSNorm,
            }.get(normalization, None)
            self.normalizer = (
                normalizer_class(embed_dim, affine=True)
                if normalizer_class is not None
                else None
            )
        else:
            self.normalizer = "layer"
        if self.normalizer is None:
            log.error(
                "Normalization type {} not found. Skipping normalization.".format(
                    normalization
                )
            )
        self.prefer_normalizer = nn.InstanceNorm2d(embed_dim, affine=True)

    def forward(self, x):
        if x.ndim == 4:
            return self.prefer_normalizer(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif isinstance(self.normalizer, RMSNorm):
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x


class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        norm_after: bool = False,  # if True, perform same as Kool et al.
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(TransformerBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        elif parallel_gated_kwargs is not None:
            ffn = ParallelGatedMLP(embed_dim, **parallel_gated_kwargs)
        else:
            ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="ReLU",
            )

        self.norm_attn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_attn_p = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )

        self.attention = MultiHeadAttention_with_graph(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm_ffn_p = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.norm_after = norm_after

    def forward(
        self,
        x: Tensor,
        edge_emb: Tensor,
        pyg_data: Batch,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.norm_after:
            # normal transformer structure
            edge_emb_x = self.norm_attn_p(edge_emb.view(-1, x.size(1), x.size(2)))
            edge_emb_x = edge_emb_x.view(x.size(0), -1, x.size(1), x.size(2))
            h, edge_emb = self.attention(self.norm_attn(x), edge_emb_x, pyg_data, mask)
            h = x + h
            edge_emb = edge_emb_x + edge_emb
            edge_emb = edge_emb.view(-1, x.size(1), x.size(2))
            h = h + self.ffn(self.norm_ffn(h))
            edge_emb = self.norm_ffn_p(edge_emb)
            edge_emb = edge_emb.view(x.size(0), -1, x.size(1), x.size(2))
        else:
            # from Kool et al. (2019)
            edge_emb_x = edge_emb.clone()
            h, edge_emb = self.attention(x, edge_emb, pyg_data, mask)
            h = self.norm_attn(x + h)
            edge_emb = self.norm_attn_p(
                (edge_emb_x + edge_emb).view(x.size(0), -1, x.size(1), x.size(2))
            )
            h = self.norm_ffn(h + self.ffn(h))
            edge_emb = self.norm_attn_p(edge_emb)

        return h, edge_emb
