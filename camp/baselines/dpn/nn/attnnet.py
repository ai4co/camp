from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

log = get_pylogger(__name__)


class MultiHeadAttention_with_preference(nn.Module):
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

        self.Wqkv_navi = nn.Linear(
            embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs
        )

        self.Wq_part1 = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk_part1 = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv_part1 = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.Wq_part2 = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk_part2 = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv_part2 = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.out_proj_navi = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )
        self.out_proj_part1 = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )
        self.out_proj_part2 = nn.Linear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        self.addAndNormalization1 = Add_And_Normalization_Module(embed_dim)
        self.feedForward1 = Feed_Forward_Module(embed_dim, embed_dim * 3)

        self.addAndNormalization2 = Add_And_Normalization_Module(embed_dim)
        self.feedForward2 = Feed_Forward_Module(embed_dim, embed_dim * 3)

        self.addAndNormalization3 = Add_And_Normalization_Module(embed_dim)
        self.feedForward3 = Feed_Forward_Module(embed_dim, embed_dim * 3)

        self.wq_p = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.wk_p = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, node_emb, prefer_emb, attn_mask=None):
        """x: (batch, num_agent + num_client, hidden_dim) (where hidden_dim = num heads * head dim)
        prefer_emb: (batch, num_agent, num_client, hidden_dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """

        node_emb, prefer_emb = node_emb, prefer_emb

        agent_num = prefer_emb.size(1)
        agent_embed, customer_embed = (
            node_emb[:, :agent_num, :],
            node_emb[:, agent_num:, :],
        )

        # Project query, key, value
        customer_q, customer_k, customer_v = rearrange(
            self.Wqkv_navi(customer_embed), "b s (three d) -> three b s d", three=3
        ).unbind(dim=0)

        customer_q = rearrange(customer_q, "b s (h d) -> b h s d", h=self.num_heads)
        customer_k = rearrange(customer_k, "b s (h d) -> b h s d", h=self.num_heads)
        customer_v = rearrange(customer_v, "b s (h d) -> b h s d", h=self.num_heads)

        customer_out = self.sdpa_fn(
            customer_q,
            customer_k,
            customer_v,
            attn_mask,
            self.attention_dropout,
            self.causal,
        )
        customer_out = rearrange(customer_out, "b h s d -> b s (h d)")
        customer_out = self.out_proj_navi(customer_out)
        customer_out = self.addAndNormalization1(customer_embed, customer_out)
        customer_out = self.feedForward1(customer_out)

        agent_q = self.Wq_part1(agent_embed)
        customer_k = self.Wk_part1(customer_out)
        customer_v = self.Wv_part1(customer_out)

        prefer_q = self.wq_p(prefer_emb)
        prefer_k = self.wk_p(prefer_emb)

        agent_q = agent_q.unsqueeze(2)
        customer_k = customer_k.unsqueeze(1)

        agent_q = (
            agent_q.expand(-1, -1, prefer_emb.size(2) - agent_embed.size(1), -1)
            + prefer_q[:, :, agent_embed.size(1) :, :]
        )
        customer_k = (
            customer_k.expand(-1, prefer_emb.size(1), -1, -1)
            + prefer_k[:, :, agent_embed.size(1) :, :]
        )

        agent_q = rearrange(agent_q, "b m n (h d) -> b h m n d", h=self.num_heads)
        customer_k = rearrange(customer_k, "b m n (h d) -> b h m n d", h=self.num_heads)
        customer_v = rearrange(customer_v, "b n (h d) -> b h n d", h=self.num_heads)

        scores = torch.matmul(agent_q, customer_k.transpose(-2, -1)) / (
            customer_k.size(-1) ** 0.5
        )
        attn_weights = (F.softmax(scores, dim=3)).sum(dim=3)
        agent_out = torch.matmul(attn_weights, customer_v)

        agent_out = rearrange(agent_out, "b h s d -> b s (h d)")
        agent_out = self.out_proj_part1(agent_out)
        agent_out = self.addAndNormalization2(agent_embed, agent_out)
        agent_out = self.feedForward2(agent_out)

        customer_q2 = rearrange(
            self.Wq_part2(customer_out), "b s (h d) -> b h s d", h=self.num_heads
        )
        agent_k = rearrange(
            self.Wk_part2(agent_out), "b s (h d) -> b h s d", h=self.num_heads
        )
        agent_v = rearrange(
            self.Wv_part2(agent_out), "b s (h d) -> b h s d", h=self.num_heads
        )
        customer_out2 = self.sdpa_fn(
            customer_q2,
            agent_k,
            agent_v,
            attn_mask,
            self.attention_dropout,
            self.causal,
        )
        customer_out2 = rearrange(customer_out2, "b h s d -> b s (h d)")

        customer_out2 = self.out_proj_part2(customer_out2)
        customer_out2 = self.addAndNormalization3(customer_out, customer_out2)
        customer_out2 = self.feedForward3(customer_out2)

        embed = torch.cat([agent_out, customer_out2], dim=1)

        return embed


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "rezero",
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super(MultiHeadAttentionLayer, self).__init__()

        # Create feed-forward network
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            self.ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            self.ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="ReLU",
            )

        # Set up attention module based on provided preference mechanism
        self.mha = MultiHeadAttention_with_preference(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )

        # Set up normalization and residual connections
        # if normalization == "rezero":
        self.w1 = nn.Parameter(torch.Tensor([0]))
        self.w2 = nn.Parameter(torch.Tensor([0]))

    def forward(self, node_embedding, preference_embedding):
        """Forward pass of the MHA layer, now with two inputs: node_embedding and preference_matrix."""

        # Apply the attention mechanism with both inputs
        attn_output = self.mha(node_embedding, preference_embedding)
        # Apply rezero normalization
        attn_output = attn_output * self.w1 + node_embedding
        # Apply feed-forward network to the attention output
        ffn_output = self.ffn(attn_output)
        # Apply rezero normalization
        ffn_output = ffn_output * self.w2 + attn_output

        return ffn_output


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization, feed-forward layer, and normalization. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        num_layers: int,
        normalization: str = "rezero",
        feedforward_hidden: int = 512,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
    ):
        super(GraphAttentionNetwork, self).__init__()

        self.layers = nn.ModuleList(
            MultiHeadAttentionLayer(
                embed_dim,
                num_heads,
                feedforward_hidden=feedforward_hidden,
                normalization=normalization,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )
            for _ in range(num_layers)
        )

    def forward(
        self, x: Tensor, preference_emb: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, num_agents+num_clients, embed_dim] initial embeddings to process
            preference_emb: [batch_size, num_agents, num_clients, embed_dim] preference embeddings to process
            mask: [batch_size, num_agents+num_clients, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"

        h = x

        for layer in self.layers:
            h = layer(h, preference_emb)

        return h


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0]))

    def forward(self, input1, input2):
        return input1 + input2 * self.alpha


class Feed_Forward_Module(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        embedding_dim = emb_dim
        ff_hidden_dim = ff_dim

        self.alpha = nn.Parameter(torch.Tensor([0]))
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return input1 + self.W2(F.relu(self.W1(input1))) * self.alpha
