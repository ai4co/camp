from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from camp.models.env_embeddings import env_init_embedding
from camp.models.nn.bipartite_transformer import Normalization, TransformerBlock
from camp.models.nn.matnet import HAMEncoderLayer, MatNetLayer
from camp.models.utils import create_batched_pyg_data


class CAMPEncoder(nn.Module):
    def __init__(
        self,
        env_name: str = "hcvrp",
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 3,
        normalization: str = "instance",
        use_final_norm: bool = False,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        norm_after: bool = False,
        **transformer_kwargs,
    ):
        super(CAMPEncoder, self).__init__()

        self.env_name = env_name
        init_embedding_kwargs["embed_dim"] = embed_dim
        self.init_embedding = (
            init_embedding
            if init_embedding is not None
            else env_init_embedding(self.env_name, init_embedding_kwargs)
        )

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    norm_after=norm_after,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.norm = Normalization(embed_dim, normalization) if use_final_norm else None

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.norm is not None:
            h = self.norm(h)

        # Return latent representation and initial embedding
        # [B, N, H]
        return h, init_h


class MatNetEncoder(nn.Module):
    def __init__(
        self,
        stage_idx: int,
        env_name: str = "ffsp",
        num_heads: int = 16,
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
        ms_hidden_dim: int = 32,
        num_layers: int = 3,
        normalization: str = "instance",
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        scale_factor: float = 1.0,
        parallel_gated_kwargs: dict = None,
        use_ham: bool = True,
        **transformer_kwargs,
    ):
        super(MatNetEncoder, self).__init__()

        self.stage_idx = stage_idx
        self.env_name = env_name
        init_embedding_kwargs["embed_dim"] = embed_dim
        self.init_embedding = (
            init_embedding
            if init_embedding is not None
            else env_init_embedding(self.env_name, init_embedding_kwargs)
        )
        if use_ham:
            LayerCls = HAMEncoderLayer
        else:
            LayerCls = MatNetLayer

        self.layers = nn.ModuleList(
            [
                LayerCls(
                    embed_dim=embed_dim,
                    head_num=num_heads,
                    ms_hidden_dim=ms_hidden_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.scale_factor = scale_factor

    def forward(self, td):
        proc_times = td["cost_matrix"]
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb, col_emb = self.init_embedding(proc_times)
        proc_times = proc_times / self.scale_factor
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, proc_times)
        return row_emb, col_emb


class PVRPEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "instance",
        use_final_norm: bool = False,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        norm_after: bool = False,
        **transformer_kwargs,
    ):
        super().__init__()

        init_embedding_kwargs["embed_dim"] = embed_dim
        self.init_embedding = (
            init_embedding
            if init_embedding is not None
            else env_init_embedding("PVRP", init_embedding_kwargs)
        )

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    norm_after=norm_after,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.norm = Normalization(embed_dim, normalization) if use_final_norm else None

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        # Transfer to embedding space
        node_emb, edge_emb = self.init_embedding(td)  # [B, M+N, H], [B, M, M+N, H]
        init_h = edge_emb

        # Batch processing of embeddings with shape [B*M, M+N, H]
        # batch size by `scaled_dot_product_attention` function. But,
        # we need to modify shit in normalization function too in this case
        batch_size, num_agents, num_nodes, hdim = edge_emb.shape

        pyg_data = create_batched_pyg_data(
            node_emb[:, :num_agents, :], edge_emb[:, :, num_agents:, :]
        )
        pyg_data = pyg_data.to(edge_emb.device)

        for layer in self.layers:
            node_emb, edge_emb = layer(node_emb, edge_emb, pyg_data, mask)
        h = edge_emb

        # normalization by each agent
        h = h.view(batch_size * num_agents, num_nodes, hdim)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.norm is not None:
            h = self.norm(h)
        h = h.view(batch_size, num_agents, num_nodes, hdim)

        # Return latent representation and initial embedding
        # [B*M, M+N, H] -> [B, M, M+N, H]
        # [B, M+N, H] -> [B, M+N, H]
        return h, init_h  # [B, M, M+N, H], #[B, M+N, H]


class PVRPHardCodeEncoder(CAMPEncoder):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 3,
        normalization: str = "instance",
        use_final_norm: bool = False,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        norm_after: bool = False,
        **transformer_kwargs,
    ):
        super(PVRPHardCodeEncoder, self).__init__(
            env_name="PVRP-hardcode",
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_layers,
            normalization=normalization,
            use_final_norm=use_final_norm,
            init_embedding=init_embedding,
            init_embedding_kwargs=init_embedding_kwargs,
            norm_after=norm_after,
            **transformer_kwargs,
        )
