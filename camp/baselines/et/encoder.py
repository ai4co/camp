import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from tensordict import TensorDict

from camp.baselines.et.env_embeddings import env_init_embedding
from camp.baselines.et.nn.attnnet import GraphAttentionNetwork


class EquityTransformerEncoder(AutoregressiveEncoder):
    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "rezero",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn=None,
        moe_kwargs: dict = None,
    ):
        super(EquityTransformerEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        init_embedding_kwargs["embed_dim"] = embed_dim

        self.init_embedding = (
            init_embedding
            if init_embedding is not None
            else env_init_embedding(self.env_name, init_embedding_kwargs)
        )

        self.layers = GraphAttentionNetwork(
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            sdpa_fn=sdpa_fn,
            moe_kwargs=moe_kwargs,
        )

    def forward(
        self,
        td: TensorDict,
    ):
        node_embed, preference_embed = self.init_embedding(td)

        h = self.layers(node_embed, preference_embed)

        return h
