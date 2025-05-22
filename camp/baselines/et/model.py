from typing import Any, Union

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.utils.ops import gather_by_index, unbatchify

from .policy import EquityTransformerPolicy


class EquityTransformerModule(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: EquityTransformerPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "shared",
        num_augment: int = 8,
        augment_fn: Union[str, callable] = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        train_min_agents: int = 5,
        train_max_agents: int = 15,
        train_min_size: int = 50,
        train_max_size: int = 100,
        val_test_num_agents: int = 10,
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = EquityTransformerPolicy(env_name=env.name, **policy_kwargs)

        super(EquityTransformerModule, self).__init__(env, policy, baseline, **kwargs)

        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        self.train_min_agents = train_min_agents
        self.train_max_agents = train_max_agents
        self.train_min_size = train_min_size
        self.train_max_size = train_max_size
        self.val_test_num_agents = val_test_num_agents

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug = self.num_augment

        # During training, we do not augment the data

        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(td, self.env, phase=phase, return_actions=True)

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, 1))

        # Training phase
        if phase == "train":
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, 1))
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = out["actions"]
                    actions_ = unbatchify(actions_, (n_aug, 1))
                    out.update(
                        {"best_aug_actions": gather_by_index(actions_, max_idxs)}
                    )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
