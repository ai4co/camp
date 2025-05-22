from typing import Callable, Union

import torch

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

log = get_pylogger(__name__)


class PVRPZCGenerator(Generator):
    """Vehicle Routing Problem with zones (VRP-zone) Zone Constraint Data Generator.

    This environment is modified based on the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) environment in CAMP.

    Args:
        - zone_distribution: 'random' or 'pi'

    Returns:
        A tensor dictionary containing the following keys:
            - locs: [batch_size, n, 2], node locations
            - depot: [batch_size, 2], depot location
            - num_agents: [batch_size], number of agents, equals to `m`
            - demand: [batch_size, n], demand of each node
            - capacity: [batch_size, m], capacity of each agent
            - speed: [batch_size, m], speed of each agent
            - zone: [batch_size, m, n], zone matrix of each agent

    Notes:
        - By default, all agents have the same capacity and speed (for simplicity).
        - `n`: number of nodes (not including the depot)
        - `m`: number of agents
    """

    def __init__(
        self,
        num_loc: int = 50,
        num_agents: int = 3,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = None,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        min_capacity: float = 40,
        max_capacity: float = 40,
        capacity_distribution: Union[int, float, type, Callable] = Uniform,
        min_speed: float = 1.0,
        max_speed: float = 1.0,
        speed_distribution: Union[int, float, type, Callable] = Uniform,
        min_zone_constraint_rate: float = 0,
        max_zone_constraint_rate: float = 2 / 3,
        zone_distribution: str = "random",
        scale_data: bool = False,  # leave False!
        **kwargs,
    ):
        self.num_loc = num_loc
        self.num_agents = num_agents
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_zone_constraint_rate = min_zone_constraint_rate
        self.max_zone_constraint_rate = max_zone_constraint_rate
        self.zone_distribution = zone_distribution
        self.scale_data = scale_data

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = (
                get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)
                if depot_distribution is not None
                else None
            )

        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler(
                "demand", demand_distribution, min_demand - 1, max_demand - 1, **kwargs
            )

        # Capacity
        if kwargs.get("capacity_sampler", None) is not None:
            self.capacity_sampler = kwargs["capacity_sampler"]
        else:
            self.capacity_sampler = get_sampler(
                "capacity",
                capacity_distribution,
                0,
                max_capacity - min_capacity,
                **kwargs,
            )

        # Speed
        if kwargs.get("speed_sampler", None) is not None:
            self.speed_sampler = kwargs["speed_sampler"]
        else:
            self.speed_sampler = get_sampler(
                "speed", speed_distribution, min_speed, max_speed, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # If depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        # Sample demands
        demand = self.demand_sampler.sample((*batch_size, self.num_loc))
        demand = (demand.int() + 1).float()

        # Sample capacities
        capacity = self.capacity_sampler.sample((*batch_size, self.num_agents))
        capacity = (capacity.int() + self.min_capacity).float()

        # Sample zones matrix
        if self.zone_distribution == "random":
            zone = self._zone_random(depot, locs, self.num_agents)
        elif self.zone_distribution == "pi":
            zone = self._zone_pi(depot, locs, self.num_agents)
        elif self.zone_distribution == "block":
            zone = self._zone_block(depot, locs, self.num_agents)
        elif self.zone_distribution == "cluster":
            zone, cluster_centers = self._zone_cluster(depot, locs, self.num_agents)
        else:
            raise ValueError(
                f"zone distribution {self.zone_distribution} not supported."
            )

        # Sample speed
        speed = self.speed_sampler.sample((*batch_size, self.num_agents))

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "num_agents": torch.full(
                    (*batch_size,), self.num_agents
                ),  # for compatibility
                "demand": demand / self.max_capacity if self.scale_data else demand,
                "capacity": capacity / self.max_capacity
                if self.scale_data
                else capacity,
                "speed": speed / self.max_speed if self.scale_data else speed,
                "preference": zone,
                "cluster_centers": cluster_centers
                if self.zone_distribution == "cluster"
                else None,
            },
            batch_size=batch_size,
        )

    def _zone_random(self, depot, locs, num_splits):
        # """Generate zone constraint matrix randomly."""

        # SECTION: option 1
        # batch_size, num_nodes, _ = locs.shape

        # zone = torch.ones(
        #     (batch_size, num_splits, num_nodes),
        #     device=locs.device,
        #     dtype=locs.dtype,
        # )

        # for agent_idx in range(num_splits):
        #     # Randomly set some nodes to 0
        #     num_zeros = torch.randint(0, int(self.max_zone_constraint_rate * num_nodes), (batch_size,))
        #     zone[torch.arange(batch_size), agent_idx, torch.randperm(num_nodes)[:num_zeros]] = 0

        # # Check if all nodes are assigned to at least one agent
        # if (zone.sum(dim=1) == 0).any():
        #     # Fix by assigning each node to at least one agent
        #     zone[zone.sum(dim=1) == 0] = 1

        # SECTION: option 2
        # batch_size, num_nodes, _ = locs.shape
        # zone = torch.rand(
        #     (batch_size, num_splits, num_nodes),
        #     device=locs.device,
        #     dtype=locs.dtype,
        # )

        # # Lower than max_zone_constraint_rate will be 0
        # zone = zone > self.max_zone_constraint_rate

        # # If all nodes are assigned to 0, randomly assign one node to 1
        # for node_idx in range(num_nodes):
        #     if (zone[:, :, node_idx] == 0).all():
        #         zone[:, torch.randint(0, num_splits, (batch_size,)), node_idx] = 1

        # SECTION: option 3
        batch_size, num_nodes, _ = locs.shape

        zone = torch.ones(
            (batch_size, num_splits, num_nodes),
            device=locs.device,
            dtype=locs.dtype,
        )

        # Zone constraint rate is a single float random from 0 to max_zone_constraint_rate
        constraint_rate = (
            torch.rand(1)
            * (self.max_zone_constraint_rate - self.min_zone_constraint_rate)
            + self.min_zone_constraint_rate
        )

        # For each node random select min(int(constraint_rate * num_splits), num_splits-1) splits to set to 0
        for node_idx in range(num_nodes):
            # mask_indices = torch.randperm(num_splits)[:min(int(constraint_rate * num_splits), num_splits-1)]
            mask_indices = torch.randperm(num_splits)[
                : int(constraint_rate * num_splits)
            ]
            zone[:, mask_indices, node_idx] = 0

        return zone

    def _zone_pi(self, depot, locs, num_splits):
        """Generate zone matrix based on angle of nodes relative to depot.

        Args:
            - depot: [batch_size, 2]
            - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
            - num_splits: int, i.e. number of agents

        Returns:
            zone: [batch_size, num_splits, num_nodes]
        """
        batch_size, num_nodes, _ = locs.shape

        # Calculate relative positions of nodes to depot
        relative_pos = locs - depot.unsqueeze(1)

        # Calculate angles of nodes relative to depot (in radians)
        angles = torch.atan2(relative_pos[..., 1], relative_pos[..., 0])

        # Shift angles to range [0, 2π]
        angles = (angles + 2 * torch.pi) % (2 * torch.pi)

        # Calculate angle range for each split
        split_angle = 2 * torch.pi / num_splits

        # Initialize zone tensor
        zone = torch.ones(batch_size, num_splits, num_nodes, dtype=torch.int)

        for i in range(num_splits):
            # Randomly select int(max_zone_constraint_rate * num_splits) splits to set to 0
            mask_indices = torch.randperm(num_splits)[
                : int(self.max_zone_constraint_rate * num_splits)
            ]
            for idx in mask_indices:
                start_angle = idx * split_angle
                end_angle = (idx + 1) * split_angle
                mask = (angles >= start_angle) & (angles < end_angle)
                zone[:, i, :] &= ~mask.int()

        return zone

    def _zone_block(self, depot, locs, num_splits):
        """Generate zone matrix based on block assignment of nodes.

        Args:
            - depot: [batch_size, 2]
            - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
            - num_splits: int, i.e. number of agents

        Returns:
            zone: [batch_size, num_splits, num_nodes]

        Notes:
            - For the current version, please use num_splits = 4, 9 for better visualization.
        """
        batch_size, num_nodes, _ = locs.shape

        num_rows = int(num_splits**0.5)
        num_cols = num_rows

        # Initialize zone tensor
        zone = torch.zeros(batch_size, num_splits, num_nodes, dtype=torch.int)

        for i in range(num_rows):
            for j in range(num_cols):
                mask = (
                    (locs[..., 0] >= i / num_rows)
                    & (locs[..., 0] < (i + 1) / num_rows)
                    & (locs[..., 1] >= j / num_cols)
                    & (locs[..., 1] < (j + 1) / num_cols)
                )
                zone[:, i * num_cols + j, :] = mask.int()

        return zone

    def _zone_cluster(self, depot, locs, num_splits):
        """Generate zone matrix based on cluster

        Args:
            - depot: [batch_size, 2]
            - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
            - num_splits: int, i.e. number of agents

        Returns:
            zone: [batch_size, num_splits, num_nodes]
            cluster_centers: [batch_size, num_splits, 2]

        Note:
            We randomly select num_splits nodes as the cluster centers, and assign each node to the nearest cluster center.
            The zone score is the normalized distance between the node and the cluster center.
        """
        batch_size, num_nodes, _ = locs.shape

        # Randomly select cluster centers
        cluster_indices = torch.randint(0, num_nodes, (batch_size, num_splits))
        cluster_centers = torch.gather(
            locs, 1, cluster_indices.unsqueeze(-1).expand(-1, -1, 2)
        )

        # Calculate distances from each node to each cluster center
        distances = torch.norm(locs.unsqueeze(1) - cluster_centers.unsqueeze(2), dim=-1)

        # Normalize distances by sqrt(2)
        normalized_distances = distances / (2**0.5)

        # Convert distances to zones (smaller distance = higher zone)
        zones = 1 - normalized_distances

        return zones, cluster_centers
