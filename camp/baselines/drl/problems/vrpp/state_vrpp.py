import torch

from tensordict.tensordict import TensorDict

from camp.envs import PVRPPSeqEnv
from camp.envs.PVRP import PVRPPGenerator


class StateHCVRP:
    VEHICLE_CAPACITY = [20.0, 25.0, 30.0, 35.0, 40.0]
    NAME = "hcvrp"

    def __init__(
        self,
        coords,
        demand,
        ids,
        veh,
        prev_a,
        used_capacity,
        visited_,
        lengths,
        cur_coord,
        i,
    ):
        """
        Args:
            coords [B, N+1, 2]
            demand [B, N+1]
            ids [B, 1]
            veh [M, 1]
            prev_a [B, M]: previous action
            used_capacity [B, M]
            visited_ [B, 1, N+1]
            lengths [B, M]
            cur_coord [B, M, 2]
            i [1]
        """
        self.coords = coords
        self.demand = demand
        self.ids = ids
        self.veh = veh
        self.prev_a = prev_a
        self.used_capacity = used_capacity
        self.visited_ = visited_
        self.lengths = lengths
        self.cur_coord = cur_coord
        self.i = i

        # SECTION: RL4CO env wrapper
        device = coords.device
        batch_size = coords.size(0)
        num_nodes = coords.size(1) - 1
        num_agents = used_capacity.size(1)
        generator = PVRPPGenerator(
            num_loc=num_nodes,
            num_agents=num_agents,
            min_capacity=40,
            max_capacity=40,
        )
        self.env = PVRPPSeqEnv(generator=generator)
        td_init = generator._generate(batch_size=[batch_size])
        td = TensorDict(
            {
                "locs": coords[:, 1:],
                "depot": coords[:, 0],
                "num_agents": torch.full((*[batch_size],), num_agents).to(
                    device
                ),  # for compatibility
                "demand": demand[:, 1:],
                "capacity": td_init["capacity"],
                "speed": td_init["speed"],
                "preference": td_init["preference"],
                "cluster_centers": None,
            }
        ).to(device)

        self.td = self.env.reset(td, batch_size=batch_size)
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_agents = num_agents

        # self.VEHICLE_CAPACITY = [20., 25., 30., 35., 40.]  # Hardcoded
        self.vehicle_capacity = td["agents_capacity"]
        self.VEHICLE_CAPACITY = td_init["capacity"][0]

        self.vehicle_speed = td["speed"]
        self.vehicle_preference = td["preference"]

    @property
    def visited(self):
        """
        Not used
        """
        return None

    @property
    def dist(self):  # coords: []
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(
            p=2, dim=-1
        )

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(
            key, slice
        ):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                veh=self.veh[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateHCVRP, self).__getitem__(key)

    @staticmethod
    def initialize(num_agents, input, visited_dtype=torch.uint8):
        """
        Args:
            - input <dict>: with keys
                - loc <torch.Tensor> [B, N, 2]
                - demand <torch.Tensor> [B, N]
                - depot <torch.Tensor> [B, 2]
                - capacity <torch.Tensor> [B, M]
        Returns:
            Updated state of the class
        """

        depot = input["depot"]
        loc = input["loc"]
        demand = input["demand"]

        batch_size, n_loc, _ = loc.size()  # n_loc = graph_size
        return StateHCVRP(
            coords=torch.cat(
                (depot[:, None, :], loc), -2
            ),  # [batch_size, graph_size, 2]]
            demand=torch.cat(
                (torch.zeros(batch_size, 1, device=loc.device), demand), 1
            ),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[
                :, None
            ],  # Add steps dimension
            veh=torch.arange(num_agents, dtype=torch.int64, device=loc.device)[:, None],
            # prev_a is current node
            prev_a=torch.zeros(
                batch_size, num_agents, dtype=torch.long, device=loc.device
            ),
            used_capacity=demand.new_zeros(batch_size, num_agents),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(
                    batch_size,
                    1,
                    (n_loc + 63) // 64,
                    dtype=torch.int64,
                    device=loc.device,
                )  # Ceil
            ),
            lengths=torch.zeros(batch_size, num_agents, device=loc.device),
            cur_coord=input["depot"][:, None, :].expand(batch_size, num_agents, -1),
            # Add step dimension
            i=torch.zeros(
                1, dtype=torch.int64, device=loc.device
            ),  # Vector with length num_steps
        )

    def get_final_cost(self):
        assert self.all_finished()
        # coords: [batch_size, graph_size+1, 2]
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(
            p=2, dim=-1
        )

    def update(self, selected, veh):  # [batch_size, num_veh]
        """
        Args:
            - selected [B, M]
            - veh [B]
        """
        nodes = torch.gather(selected, 1, veh[:, None]).squeeze(1)  # [B]

        # Fix the next_node with the vehicle index
        next_node_ = nodes + self.num_agents - 1

        # Replace the `self.veh_num - 1` with the vel
        next_node_ = torch.where(next_node_ == self.num_agents - 1, veh, next_node_)

        actions = torch.stack([veh, next_node_], dim=1)  # [B, 2]

        # actions = torch.stack([veh, selected], dim=1)  # [B, 2]

        td = self.td
        td.batch_size = [self.batch_size]
        td.update(
            {
                "action": actions,
            }
        )
        td_updated = self.env.step(td)["next"]
        self.td = td_updated

        cur_coord = torch.gather(
            self.td["locs"], 1, self.td["current_node"][..., None].expand(-1, -1, 2)
        )  # [B, M, 2]

        self._replace(
            prev_a=selected,
            used_capacity=self.td["used_capacity"],
            visited_=None,  # Not used
            lengths=self.td["current_length"],
            cur_coord=cur_coord,
            i=self.i + 1,
        )

        return next_node_

    def _replace(
        self,
        prev_a,
        used_capacity,
        visited_,
        lengths,
        cur_coord,
        i,
    ):
        self.prev_a = prev_a
        self.used_capacity = used_capacity
        self.visited_ = visited_
        self.lengths = lengths
        self.cur_coord = cur_coord
        self.i = i

    def all_finished(self):
        # return self.visited.all()
        return self.td["done"].all()

    def get_finished(self):
        # return self.visited.sum(-1) == self.visited.size(-1)
        return self.td["done"]

    def get_current_node(self):
        current_node = self.td["current_node"]

        # Fix the node idx
        current_node_ = current_node - self.num_agents + 1
        current_node_ = torch.where(current_node_ < 0, 0, current_node_)

        return current_node_

    def get_mask(self, veh):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited

        Args:
            veh [B]: veh index

        Returns:
            [B, 1N+1]: mask of respective vehicle
        """
        mask = self.td["action_mask"]  # [B, M, M+N]

        # SECTION: compress the depot mask
        agent_idx = torch.arange(self.num_agents)
        depot_mask = mask[:, agent_idx, agent_idx]

        mask = ~torch.cat(
            [depot_mask.unsqueeze(-1), mask[..., self.num_agents :]], dim=-1
        )  # [B, M, N+1]

        mask = torch.gather(
            mask, 1, veh[:, None, None].expand(-1, 1, mask.size(-1))
        ).squeeze(1)  # [B, N+1]

        return mask[:, None, :]

    def get_reward(self, veh, pi):
        return self.env.get_reward(self.td, {"veh": veh, "pi": pi})

    def construct_solutions(self, actions):
        return actions
