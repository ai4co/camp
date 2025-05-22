from typing import Callable, List

import torch.nn as nn

from rl4co.envs import get_env
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.utils.decoding import get_decoding_strategy, get_log_likelihood
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from .decoder import DPNDecoder
from .encoder import DPNEncoder
from .utils.decoding import DecodingStrategy

log = get_pylogger(__name__)


class DPNPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: str = "PVRP",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        moe_kwargs: dict = {"encoder": None, "decoder": None},
        multi_objective_weights: List[float] = [1, 0],
        **unused_kwargs,
    ):
        if encoder is None:
            encoder = DPNEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
                moe_kwargs=moe_kwargs["encoder"],
            )

        if decoder is None:
            decoder = DPNDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                moe_kwargs=moe_kwargs["decoder"],
            )

        super(DPNPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )

        self.multi_objective_weights = multi_objective_weights

    def forward(
        self,
        td: TensorDict,
        env,
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
        # Encoder: get encoder output
        hidden = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # When decode_type is sampling, we need to know the number of samples
        num_samples = decoding_kwargs.pop("num_samples", 1)
        if "sampling" not in decode_type:
            num_samples = 1

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            num_samples=num_samples,
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            time, preference = env._get_reward(td, actions)
            reward = (
                -time * self.multi_objective_weights[0]
                + preference * self.multi_objective_weights[1]
            )
            td.set("reward", reward)
            td.set("time", time)
            td.set("preference", preference)

            outdict = {
                "reward": td["reward"],
                "log_likelihood": get_log_likelihood(
                    logprobs, actions, td.get("mask", None), return_sum_log_likelihood
                ),
                "time": td["time"],
                "preference": td["preference"],
            }

        if return_actions:
            outdict["actions"] = actions

        return outdict
