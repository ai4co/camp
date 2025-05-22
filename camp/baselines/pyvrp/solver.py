import numpy as np
import pyvrp as pyvrp
import torch

from pyvrp import Client, Depot, ProblemData, Profile, VehicleType, solve as _solve
from pyvrp.constants import MAX_VALUE
from pyvrp.stop import MaxRuntime
from tensordict.tensordict import TensorDict
from torch import Tensor

PYVRP_SCALING_FACTOR = 1_000

MAX_VALUE = 1 << 42  # noqa: F811


def scale(data: Tensor, scaling_factor: int):
    """Scales ands rounds data to integers so PyVRP can handle it."""
    array = (data * scaling_factor).numpy().round()
    array = np.where(array == np.inf, np.iinfo(np.int32).max, array)
    array = array.astype(int)
    if array.size == 1:
        return array.item()
    return array


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    data = instance2data(instance, PYVRP_SCALING_FACTOR)
    stop = MaxRuntime(max_runtime)
    result = _solve(data, stop)
    solution = result.best
    action = solution2action(solution)
    cost = -result.cost() / PYVRP_SCALING_FACTOR
    return action, cost, solution


def instance2data(
    instance: TensorDict,
    scaling_factor: int,
    preference_weight: float = 0.1,
) -> ProblemData:
    num_loc = instance["demand"].size()[-1]
    num_vehicles = instance["preference"].size()[0]

    # Preprocessing
    # first num_vehicles are the preferences of depots which are 0
    preferences = instance["preference"]
    preferences = torch.cat(
        [torch.zeros_like(preferences[:, :num_vehicles]), preferences], dim=-1
    )
    cost_mat = torch.cdist(instance["locs"], instance["locs"])  # note: normalized after

    # Scaling, move to numpy
    capacity = scale(
        instance["capacity"], scaling_factor
    )  # no scaling, already integers
    demand = scale(instance["demand"], scaling_factor)  # no scaling, already integers
    coords = scale(instance["locs"], scaling_factor)

    profiles = []
    depots = []
    vehicle_types = []
    cost_matrices = []
    for idx in range(num_vehicles):
        profile = Profile()
        # NOTE: important: multi-depot by default from env!
        depot = Depot(
            x=coords[idx][0],  # if single depot, they will be the same from the env
            y=coords[idx][1],
        )
        vehicle_type = VehicleType(
            num_available=(
                num_loc - num_vehicles
            ),  # max num routes = num_loc - 1 // num_vehicles
            start_depot=idx,
            end_depot=idx,  # same, not flexible
            capacity=capacity[idx],
            profile=idx,
        )

        # Create cost matrix as sum of duration - preference
        # preference is basically a bonus
        # NOTE: normalization occurs after
        cost_mat_veh = cost_mat.clone() / instance["speed"][idx]  # duration
        # if we visit a node j by going any i--> j, we get a bonus, i.e. reduce the cost
        preference_veh = preferences[idx]
        cost_mat_veh = (
            cost_mat_veh - preference_weight * preference_veh[None, :]
        )  # broadcasting

        # normalize to make cost positive
        cost_mat_veh = cost_mat_veh + torch.max(
            preference_veh
        )  # goofy ahh "normalization"
        cost_mat_veh[torch.arange(num_loc), torch.arange(num_loc)] = (
            0  # avoid PyVRP complaining about diagonal
        )
        cost_mat_veh = scale(cost_mat_veh, scaling_factor)  # scale only at the end

        depots.append(depot)
        profiles.append(profile)
        vehicle_types.append(vehicle_type)
        cost_matrices.append(cost_mat_veh)

    clients = [
        Client(
            x=coords[idx][0],
            y=coords[idx][1],
            delivery=demand[0, idx],  # dummy demand?
        )
        for idx in range(num_vehicles, num_loc)  # eliminate the depots
    ]

    return ProblemData(clients, depots, vehicle_types, cost_matrices, cost_matrices)


def solution2action(solution: pyvrp.Solution) -> list[int]:
    """
    Converts a PyVRP solution to the action representation, i.e., a giant tour.
    """
    actions = []
    for route in solution.routes():
        veh_idx = route.vehicle_type()
        actions.extend([veh_idx] + [visit for visit in route.visits() + [veh_idx]])
    return actions
