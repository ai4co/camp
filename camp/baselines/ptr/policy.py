from typing import Optional, Union

import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from .nets.attention_model import AttentionModel

log = get_pylogger(__name__)


class PtrPolicy(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        env=None,
        obj: str = "min-sum",
        n_encode_layers: int = 2,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        normalization: str = "batch",
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        multi_objective_time_weights=[0.9, 0.1],
        shrink_size: Optional[int] = None,
        max_size=None,
    ):
        super(PtrPolicy, self).__init__()

        self.embed_dim = embed_dim

        # Warning: max_size is required
        if max_size is None:
            raise ValueError("max_size is required")

        self.ptr_net = AttentionModel(
            embedding_dim=embed_dim,
            hidden_dim=embed_dim,
            obj=obj,
            problem=env,
            n_encode_layers=n_encode_layers,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            normalization=normalization,
            n_heads=n_heads,
            checkpoint_encoder=checkpoint_encoder,
            multi_objective_time_weights=multi_objective_time_weights,
            shrink_size=shrink_size,
            max_size=max_size,
        )

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """
        num_agents = td["capacity"].size(-1)
        # num_nodes = td["locs"].size(-2) - num_agents

        inputs = {
            "loc": td["locs"][:, num_agents:, :],  # [B, N, 2]
            "demand": td["demand"][:, 0, num_agents:],  # [B, N]
            "depot": td["locs"][:, 0, :],  # [B, 2]
            "capacity": td["capacity"],
            "speed": td["speed"],
        }

        if return_actions:
            cost, time, preference, ll, pi, veh = self.ptr_net(
                input=inputs, return_pi=True
            )
            outdict = {
                "reward": -cost.squeeze(),
                "time": time,
                "preference": preference,
                "log_likelihood": ll,
                "pi": pi,
                "veh": veh,
            }
        else:
            cost, time, preference, ll = self.ptr_net(input=inputs, return_pi=False)
            outdict = {
                "reward": -cost.squeeze(),
                "time": time,
                "preference": preference,
                "log_likelihood": ll,
            }

        return outdict
