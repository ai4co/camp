from typing import Callable, Union

import torch

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

log = get_pylogger(__name__)


class PVRPPGenerator(Generator):
    """Vehicle Routing Problem with Preferences (VRP-Preference) Data Generator.

    This environment is modified based on the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) environment in CAMP.

    Args:
        - preference_distribution: 'random' or 'pi'

    Returns:
        A tensor dictionary containing the following keys:
            - locs: [batch_size, n, 2], node locations
            - depot: [batch_size, 2], depot location
            - num_agents: [batch_size], number of agents, equals to `m`
            - demand: [batch_size, n], demand of each node
            - capacity: [batch_size, m], capacity of each agent
            - speed: [batch_size, m], speed of each agent
            - preference: [batch_size, m, n], preference matrix of each agent

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
        min_preference_factor: float = 0.0,
        max_preference_factor: float = 1.0,
        preference_distribution: float = "random",
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
        self.min_preference_factor = min_preference_factor
        self.max_preference_factor = max_preference_factor
        self.preference_distribution = preference_distribution
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

        # Sample preferences matrix
        if self.preference_distribution == "random":
            preference = torch.rand(
                (*batch_size, self.num_agents, self.num_loc),
                device=demand.device,
                dtype=demand.dtype,
            )
        elif self.preference_distribution == "pi":
            preference = self._preference_pi(depot, locs, self.num_agents)
        elif self.preference_distribution == "block":
            preference = self._preference_block(depot, locs, self.num_agents)
        elif self.preference_distribution == "cluster":
            preference, cluster_centers = self._preference_cluster(
                depot, locs, self.num_agents
            )
        else:
            raise ValueError(
                f"Preference distribution {self.preference_distribution} not supported."
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
                "preference": preference,
                "cluster_centers": cluster_centers
                if self.preference_distribution == "cluster"
                else None,
            },
            batch_size=batch_size,
        )

    def _preference_pi(self, depot, locs, num_splits):
        """Generate preference matrix based on angle of nodes relative to depot.

        Args:
            - depot: [batch_size, 2]
            - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
            - num_splits: int, i.e. number of agents

        Returns:
            preference: [batch_size, num_splits, num_nodes]
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

        # Initialize preference tensor
        preference = torch.zeros(batch_size, num_splits, num_nodes, dtype=torch.int)

        # Assign nodes to splits based on their angles
        for i in range(num_splits):
            start_angle = i * split_angle
            end_angle = (i + 1) * split_angle
            mask = (angles >= start_angle) & (angles < end_angle)
            preference[:, i, :] = mask.int()

        return preference

    def _preference_block(self, depot, locs, num_splits):
        """Generate preference matrix based on block assignment of nodes.

        Args:
            - depot: [batch_size, 2]
            - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
            - num_splits: int, i.e. number of agents

        Returns:
            preference: [batch_size, num_splits, num_nodes]

        Notes:
            - For the current version, please use num_splits = 4, 9 for better visualization.
        """
        batch_size, num_nodes, _ = locs.shape

        num_rows = int(num_splits**0.5)
        num_cols = num_rows

        # Initialize preference tensor
        preference = torch.zeros(batch_size, num_splits, num_nodes, dtype=torch.int)

        for i in range(num_rows):
            for j in range(num_cols):
                mask = (
                    (locs[..., 0] >= i / num_rows)
                    & (locs[..., 0] < (i + 1) / num_rows)
                    & (locs[..., 1] >= j / num_cols)
                    & (locs[..., 1] < (j + 1) / num_cols)
                )
                preference[:, i * num_cols + j, :] = mask.int()

        return preference

    def _preference_cluster(self, depot, locs, num_splits):
        """Generate preference matrix based on cluster

        Args:
            - depot: [batch_size, 2]
            - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
            - num_splits: int, i.e. number of agents

        Returns:
            preference: [batch_size, num_splits, num_nodes]
            cluster_centers: [batch_size, num_splits, 2]

        Note:
            We randomly select num_splits nodes as the cluster centers, and assign each node to the nearest cluster center.
            The preference score is the normalized distance between the node and the cluster center.
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

        # Convert distances to preferences (smaller distance = higher preference)
        preferences = 1 - normalized_distances

        return preferences, cluster_centers
