import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from camp.baselines.ortools.utils import create_data_model
from camp.baselines.utils import capacity_check, get_preference_cost
from camp.data.data_loader import numpy_data_loader
from camp.data.enums import PreferenceRegionType


def main(data, max_trip=1):
    # Create the routing index manager
    n_true_vehicles = data["num_vehicles"]
    manager = pywrapcp.RoutingIndexManager(
        len(data["demands"]), n_true_vehicles * max_trip, data["depot"]
    )

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc
    def distance_callback(from_index, to_index, vehicle_id):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return (
            data["base_cost_matrix"][from_node][to_node]
            + data["preference_matrix"][vehicle_id % n_true_vehicles][from_node]
        )

    transit_callback_indices = []
    for vehicle_id in range(data["num_vehicles"] * max_trip):
        transit_callback_index = routing.RegisterTransitCallback(
            lambda from_index, to_index, v=vehicle_id: distance_callback(
                from_index, to_index, v
            )
        )
        transit_callback_indices.append(transit_callback_index)
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, vehicle_id)

    # Add Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"] * max_trip,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.time_limit.FromSeconds(data["time_limit"])

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution
    if solution:
        return print_solution(data, manager, routing, solution, max_trip)
    else:
        print("No solution found!")


def print_solution(data, manager, routing, solution, max_trip):
    all_routes = []
    for vehicle_id in range(data["num_vehicles"] * max_trip):
        route = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        all_routes.append(route)

    # Merge routes
    merged_routes = merge_routes(all_routes, data["num_vehicles"])

    # Format the output to match [[29, 10, 5, 30, 13, 18, 7], ...]
    return merged_routes


def merge_routes(all_routes, num_vehicles):
    merged_routes = [[] for _ in range(num_vehicles)]

    for i, route in enumerate(all_routes):
        if len(route) == 2:
            continue
        vehicle_group = i % num_vehicles
        if merged_routes[vehicle_group]:
            # Remove the ending depot (last element) of the current merged route and starting depot (first element) of the next route
            if merged_routes[vehicle_group][-1] == route[0] == 0:
                merged_routes[vehicle_group] = merged_routes[vehicle_group] + route[1:]
        else:
            merged_routes[vehicle_group] = route

    return merged_routes


if __name__ == "__main__":
    full_data = numpy_data_loader("data/PVRP/", 40, 3, PreferenceRegionType.PI)
    for i in range(full_data["depot"].shape[0]):
        data = create_data_model(
            full_data["depot"][i],
            full_data["locs"][i],
            full_data["demand"][i],
            full_data["capacity"][i],
            full_data["speed"][i],
            full_data["preference"][i],
            5,
            1000000,
            0,
            1,
        )
        result = main(data, 3)
        capacity_check(result, full_data["capacity"][i], data["demands"])
        matrix = full_data["preference"][i].max() - full_data["preference"][i]
        padded_matrix = np.pad(
            matrix, pad_width=((0, 0), (1, 1)), mode="constant", constant_values=0
        )
        dist, pref = get_preference_cost(
            result, data["base_cost_matrix"], padded_matrix
        )
        print(dist)
        print(pref)
        print(result)
        break
