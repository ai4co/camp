import os

import numpy as np

from camp.data.enums import PreferenceRegionType


def numpy_data_loader(
    base_path: str,
    n_clients: int,
    n_agents: int,
    preference_type: PreferenceRegionType,
    seed: int = 1234,
) -> np.lib.npyio.NpzFile:
    full_path = os.path.join(
        base_path, f"n{n_clients}_m{n_agents}_seed{seed}_{preference_type}.npz"
    )

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {full_path} does not exist.")

    return np.load(full_path)
