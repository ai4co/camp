import torch

from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

from camp.envs.PVRP import PVRPPEnv
from camp.utils.ops import convert_seq_actions

from .render import render

log = get_pylogger(__name__)


class PVRPPSeqEnv(PVRPPEnv):
    """Profiled Vehicle Routing Problem with Preferences (PVRP-Preference) Environment (Sequence Version).
    This environment is implemented for ET, 2D-Ptr, etc sequence model. If you are using the CAMP and CAMP,
    please use the PVRPPEnv instead, which is a parallel model.

    The only difference between this environment and the parallel version is the action space. In this
    environment, each step the action is a pair of (agent, node).
    """

    name = "PVRP_seq"

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Keys:
            - action [batch_size, 2]: the first column is the agent index, the second column is the node index
        """
        num_agents = td["current_node"].size(-1)

        # Converting the action to the parallel version: i.e. only the moving agent action is updated
        # other agents stay at the same node. This can reuse the parallel version of the VRP environment.
        selected_agent = td["action"][..., 0]  # [batch_size]
        selected_node = td["action"][..., 1]  # [batch_size]

        action = td["current_node"].clone()
        action.scatter_(-1, selected_agent.unsqueeze(1), selected_node.unsqueeze(1))

        # Update the current length
        current_loc = gather_by_index(td["locs"], action)
        previous_loc = gather_by_index(td["locs"], td["current_node"])
        current_length = td["current_length"] + get_distance(previous_loc, current_loc)

        # Update the current preference
        selected_preference = gather_by_index(td["agents_preference"], action, dim=-1)

        # Update the used capacity
        # Increase used capacity if not visiting the depot, otherwise set to 0
        selected_demand = gather_by_index(td["demand"], action, dim=-1)

        # If the agent is staying at the same node, do not add the demand and preference the second time
        stay_flag = action == td["current_node"]

        selected_demand = selected_demand * (~stay_flag).float()
        used_capacity = (td["used_capacity"] + selected_demand) * (
            action >= num_agents
        ).float()

        selected_preference = selected_preference * (~stay_flag).float()
        current_preference = td["current_preference"] + selected_preference

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, action, 1)

        # update the done and reward
        done = visited[..., num_agents:].sum(-1) == (visited.size(-1) - num_agents)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_length": current_length,
                "current_preference": current_preference,
                "current_node": action,
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
        within_capacity_flag = td["demand"] <= remain_capacity[..., None]
        action_mask &= within_capacity_flag

        # The depot is not available if the agent is staying at the depot
        # Note that this is different with the parallel version since each time we only move one agent
        at_depot_flag = td["current_node"] < num_agents
        depot_mask = ~at_depot_flag  # 1 means we can visit [batch_size, num_agents]

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
    def check_solution_validity(td: TensorDict, ori_actions: torch.Tensor):
        """Check the validity of the solution for a sequence model

        Args:
            - actions <tensorDict>: note that this action is different with common parallel processed
                environment action. With two components: `veh` and `pi`, each is a tensor of shape [batch_size, seq_len]
        """
        # Convert the actions to the parallel version
        actions = convert_seq_actions(td, ori_actions["veh"], ori_actions["pi"])
        actions = actions.to(td.device)

        # Original check_solution_validity
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

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None, **kwargs):
        return render(td, actions, ax, **kwargs)
