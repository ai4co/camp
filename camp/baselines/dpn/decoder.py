import torch.nn as nn

from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.utils.pylogger import get_pylogger

from .env_embeddings import env_context_embedding, env_dynamic_embedding

log = get_pylogger(__name__)


class DPNDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "PVRP",
        context_embedding: nn.Module = None,
        context_embedding_kwargs: dict = {},
        dynamic_embedding: nn.Module = None,
        dynamic_embedding_kwargs: dict = {},
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        context_embedding_kwargs["embed_dim"] = embed_dim  # replace
        if context_embedding is None:
            context_embedding = env_context_embedding(
                env_name, context_embedding_kwargs
            )

        if dynamic_embedding is None:
            dynamic_embedding = env_dynamic_embedding(
                env_name, dynamic_embedding_kwargs
            )

        super(DPNDecoder, self).__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            pointer=pointer,
            moe_kwargs=moe_kwargs,
        )
