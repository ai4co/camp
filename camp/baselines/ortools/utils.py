import numpy as np

from camp.baselines.utils import euclidean_distance_matrix


# Define data for the CVRP
# Capacitated Vehicle Routing Problem (CVRP) solver using Google OR-Tools
def create_data_model(
    depot_coord: np.ndarray,
    locs: np.ndarray,
    demands: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    preferences: np.ndarray,
    time_limit: int,
    multiplier: int,
    duration_coefficient: float,
    preference_coefficient: float,
):
    base_cost_matrix = (
        euclidean_distance_matrix(np.vstack((depot_coord, locs)))
        * multiplier
        * duration_coefficient
        / int(speeds[0])
    ).astype(int)
    data = {}
    data["base_cost_matrix"] = base_cost_matrix
    data["preference_matrix"] = []
    for vehicle_idx in range(len(capacities)):
        # Pad one zero at the beginning of the preferences array
        preferences_padded = np.pad(
            preferences.max() - preferences[vehicle_idx], (1, 0), "constant"
        )

        # Add preferences to all rows of the base_cost_matrix
        data["preference_matrix"].append(
            (
                (preferences_padded * multiplier * preference_coefficient).astype(int)
            ).tolist()
        )
    data["demands"] = [0] + demands.astype(int).tolist()
    data["vehicle_capacities"] = capacities.astype(
        int
    ).tolist()  # Each vehicle has its own capacity
    data["num_vehicles"] = len(data["vehicle_capacities"])
    data["depot"] = 0
    data["time_limit"] = time_limit
    return data
