from typing import Optional

import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from .generator import PVRPPGenerator
from .render import render

log = get_pylogger(__name__)


class PVRPPEnv(RL4COEnvBase):
    """Profiled Vehicle Routing Problem with Preferences (PVRP-Preference) Environment."""

    name = "PVRP"

    def __init__(
        self,
        generator: PVRPPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = PVRPPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        """
        Returns:
            A tensor dictionary containing the following keys:
                - locs: [batch_size, m+n, 2], node locations, including the depot repeated for each agent
                - demand: [batch_size, m, m+n], demand of each node, including the depot demand, repeated for each agent
                - current_length: [batch_size, m], current length of each agent
                - current_preference: [batch_size, m], current preference of each agent
                - current_node: [batch_size, m], current node of each agent
                - depot_node: [batch_size, m], depot node of each agent

        Notes:
            - `n`: number of nodes (not including the depot)
            - `m`: number of agents
        """
        device = td.device

        # Record parameters
        num_agents = td["capacity"].size(-1)
        num_loc_all = td["locs"].size(-2) + num_agents

        # Repeat the depot for each agent (i.e. each agent has its own depot, at the same place)
        depots = td["depot"]
        if depots.shape[-2] == 1 or depots.ndim == 2:
            depots = depots.unsqueeze(-2) if depots.ndim == 2 else depots
            depots = depots.repeat(1, num_agents, 1)

        # Padding depot demand as 0 to the demand
        demand_depot = torch.zeros(
            (*batch_size, num_agents), dtype=torch.float32, device=device
        )
        demand = torch.cat((demand_depot, td["demand"]), -1)

        # Repeat the demand for each agent, for convenient action mask calculation
        # Note that this will take more memory
        demand = demand.unsqueeze(-2).repeat(1, num_agents, 1)

        # Padding depot preference as 0 to the preference
        preference_depot = torch.zeros(
            (*batch_size, num_agents, num_agents), dtype=torch.float32, device=device
        )
        preference = torch.cat((preference_depot, td["preference"]), -1)

        # Init current node
        depot_node = torch.arange(num_agents, dtype=torch.int64, device=device)[
            None, ...
        ].repeat(*batch_size, 1)
        current_node = depot_node.clone()

        # Init visited
        visited = torch.zeros(
            (*batch_size, num_loc_all), dtype=torch.bool, device=device
        )

        # Init action mask
        action_mask = torch.ones(
            (*batch_size, num_agents, num_loc_all), dtype=torch.bool, device=device
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((depots, td["locs"]), -2),
                "demand": demand,
                "current_length": torch.zeros(
                    (*batch_size, num_agents), dtype=torch.float32, device=device
                ),
                "current_preference": torch.zeros(
                    (*batch_size, num_agents), dtype=torch.float32, device=device
                ),
                "current_node": current_node,
                "depot_node": depot_node,
                "used_capacity": torch.zeros((*batch_size, num_agents), device=device),
                "agents_capacity": td["capacity"],
                "agents_speed": td["speed"],
                "agents_preference": preference,  # [B, m, m+n]
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device),
                "visited": visited,
                "action_mask": action_mask,
                "done": torch.zeros((*batch_size,), dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Keys:
            - action [batch_size, num_agents]: action taken by each agent
        """
        num_agents = td["current_node"].size(-1)

        # Update the current length
        current_loc = gather_by_index(td["locs"], td["action"])
        previous_loc = gather_by_index(td["locs"], td["current_node"])
        current_length = td["current_length"] + get_distance(previous_loc, current_loc)

        # Update the current preference
        selected_preference = gather_by_index(
            td["agents_preference"], td["action"], dim=-1
        )

        # Update the used capacity
        # Increase used capacity if not visiting the depot, otherwise set to 0
        selected_demand = gather_by_index(td["demand"], td["action"], dim=-1)

        # If the agent is staying at the same node, do not add the demand and the preference the second time
        stay_flag = td["action"] == td["current_node"]

        selected_demand = selected_demand * (~stay_flag).float()
        used_capacity = (td["used_capacity"] + selected_demand) * (
            td["action"] >= num_agents
        ).float()

        selected_preference = selected_preference * (~stay_flag).float()
        current_preference = td["current_preference"] + selected_preference

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, td["action"], 1)

        # update the done and reward
        done = visited[..., num_agents:].sum(-1) == (visited.size(-1) - num_agents)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_length": current_length,
                "current_preference": current_preference,
                "current_node": td["action"],
                "used_capacity": used_capacity,
                "i": td["i"] + 1,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        batch_size = td.batch_size
        num_agents = td["current_node"].size(-1)

        # Init action mask for each agent with all not visited nodes
        action_mask = torch.repeat_interleave(
            ~td["visited"][..., None, :], dim=-2, repeats=num_agents
        )

        # Can not visit the node if the demand is more than the remaining capacity
        remain_capacity = td["agents_capacity"] - td["used_capacity"]
        within_capacity_flag = td["demand"] <= remain_capacity[..., None]  # TODO: check
        action_mask &= within_capacity_flag

        # The depot is not available if **all** the agents are at the depot and the task is not finished
        all_back_flag = torch.sum(td["current_node"] >= num_agents, dim=-1) == 0
        has_finished_early = all_back_flag & ~td["done"]

        depot_mask = ~has_finished_early[..., None]  # 1 means we can visit

        # If no available nodes outside (all visited), make the depot always available
        all_visited_flag = (
            torch.sum(~td["visited"][..., num_agents:], dim=-1, keepdim=True) == 0
        )
        depot_mask |= all_visited_flag

        # Update the depot mask in the action mask
        eye_matrix = torch.eye(num_agents, device=td.device)
        eye_matrix = eye_matrix[None, ...].repeat(*batch_size, 1, 1).bool()
        eye_matrix &= depot_mask[..., None]
        action_mask[..., :num_agents] = eye_matrix

        return action_mask

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Min-sum with preference panelty reward function.
        Returns:
            - sum_time [batch_size]: sum of the time, lower is better
            - sum_preference [batch_size]: sum of the preference, higher is better
        """
        current_length = td["current_length"]

        # Adding the final distance to the depot
        current_loc = gather_by_index(td["locs"], td["current_node"])
        depot_loc = gather_by_index(td["locs"], td["depot_node"])
        current_length = td["current_length"] + get_distance(depot_loc, current_loc)

        # Calculate the time
        current_time = current_length / td["agents_speed"]

        # Min-sum of the time
        sum_time = current_time.sum(dim=-1)

        # Min-sum of the preference
        sum_preference = td["current_preference"].sum(dim=-1)

        return sum_time, sum_preference

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check the validity of the solution.

        Args:
            - actions [batch_size, num_agents, seq_len]: actions taken by each agent

        Notes:
            - This function is implemented in a low efficiency way, only for debugging purposes.
        """
        num_agents = td["current_node"].size(-1)
        num_loc = td["locs"].size(-2) - num_agents
        batch_size = td.batch_size

        # Flatten the actions of all agents
        actions_flatten = actions.flatten(start_dim=-2)

        # Sort the actions from small to large
        actions_flatten_sort = actions_flatten.sort(dim=-1)[0]

        # Check if visited all nodes
        for batch_idx in range(*batch_size):
            actions_sort_unique = torch.unique(actions_flatten_sort[batch_idx])
            actions_sort_unique = actions_sort_unique[actions_sort_unique >= num_agents]
            assert (
                torch.arange(num_agents, num_agents + num_loc, device=td.device)
                == actions_sort_unique
            ).all(), (
                f"Invalid tour at batch {batch_idx} with tour {actions_sort_unique}"
            )

    def _make_spec(self, generator: PVRPPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
                device=self.device,
            ),
            demand=BoundedTensorSpec(
                low=-generator.min_demand,
                high=generator.max_demand,
                shape=(generator.num_loc + 1, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
            device=self.device,
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), device=self.device)
        self.done_spec = UnboundedDiscreteTensorSpec(
            shape=(1,), dtype=torch.bool, device=self.device
        )

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None, **kwargs):
        return render(td, actions, ax, **kwargs)
