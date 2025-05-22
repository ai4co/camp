import numpy as np


def euclidean_distance_matrix(arr):
    """
    Computes the Euclidean distance matrix for a 2D array arr.

    Parameters:
    arr (numpy array): A 2D array of shape (n_samples, n_features)

    Returns:
    D (numpy array): A matrix of shape (n_samples, n_samples) where D[i, j] is the Euclidean distance between arr[i] and arr[j].
    """
    # Calculate the squared Euclidean distance
    squared_diff = np.sum((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2, axis=2)

    # Take the square root to get the Euclidean distance
    distance_matrix = np.sqrt(squared_diff)

    return distance_matrix


def restore_indices(matrix):
    restored_indices = []
    for row in matrix:
        indices = (np.where(row == 1)[0] + 1).tolist()
        restored_indices.append(indices)
    return restored_indices


def capacity_check(routes, capacities, demands):
    for route_idx, route in enumerate(routes):
        default_capacity = capacities[route_idx % len(capacities)]
        capacity = default_capacity
        for client in route:
            if client == 0:
                capacity = default_capacity
            else:
                capacity -= demands[client]
            assert capacity >= 0


def get_preference_cost(routes, distance_matrix, preference_matrix):
    distance_cost = 0
    preference_cost = 0
    for route in routes:
        for prev, next in zip(route, route[1:]):
            distance_cost += distance_matrix[prev][next]
    for route_idx in range(len(routes)):
        for client in routes[route_idx]:
            preference_cost += preference_matrix[route_idx % len(preference_matrix)][
                client
            ]
    return distance_cost, preference_cost
