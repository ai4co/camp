import numpy as np
import torch

from tensordict import TensorDict


def scatter_at_index(src, idx):
    """Scatter elements from camp at index idx along specified dim

    Now this function is specific for the multi agent masking, you may
    want to create a general function.

    Example:
    >>> src: shape [64, 3, 20] # [batch_size, num_agents, num_nodes]
    >>> idx: shape [64, 3] # [batch_size, num_agents]
    >>> Returns: [64, 3, 20]
    """
    idx_ = torch.repeat_interleave(idx.unsqueeze(-2), dim=-2, repeats=src.shape[-2])
    return src.scatter(-1, idx_, 0)


def pad_tours(data):
    # Determine the maximum length from all the lists
    max_len = max(len(lst) for lst in data.values())

    # Pad each list to match the maximum length
    for key, value in data.items():
        data[key] = value + [0] * (max_len - len(value))

    # Make tensor
    tours = torch.tensor(list(data.values()))
    return tours


def pad_actions(tensors):
    # Determine the maximum length from all tensors
    max_len = max(t.size(1) for t in tensors)
    # Pad each tensor to match the maximum length
    padded_tensors = []
    for t in tensors:
        if t.size(1) < max_len:
            pad_size = max_len - t.size(1)
            pad = torch.zeros(t.size(0), pad_size).long()
            padded_t = torch.cat([t, pad], dim=-1)
        else:
            padded_t = t
        padded_tensors.append(padded_t)
    return torch.stack(padded_tensors)


def rollout(instances, actions, env, num_agents, preprocess_actions=True, verbose=True):
    assert env is not None, "Environment must be provided"

    if env.name == "mpdp":
        depots = instances["depots"]
        locs = instances["locs"]
        diff = num_agents - 1
    else:
        depots = instances[:, 0:1]
        locs = instances[:, 1:]
        diff = num_agents - 2

    td = TensorDict(
        {
            "depots": depots,
            "locs": locs,
            "num_agents": torch.tensor([num_agents] * locs.shape[0]),
        },
        batch_size=[locs.shape[0]],
    )

    td_init = env.reset(td)

    if preprocess_actions:
        # Make actions as follows: add to all numbers except 0 the number of agents
        actions_md = torch.where(actions > 0, actions + diff, actions)
    else:
        actions_md = actions

    # Rollout through the environment
    td_init_test = td_init.clone()
    next_td = td_init_test
    with torch.no_grad():
        # take actions from the last dimension
        for i in range(actions_md.shape[-1]):
            cur_a = actions_md[:, :, i]
            next_td.set("action", cur_a)
            next_td = env.step(next_td)["next"]

    # Plotting
    # env.render(td_init_test, actions_md, plot_number=True)
    reward = env.get_reward(next_td, actions_md)
    if verbose:
        print(f"Average reward: {reward.mean()}")
    # return instances, actions, actions_md, reward, next_td
    return {
        "instances": instances,
        "actions": actions,
        "actions_md": actions_md,
        "reward": reward,
        "td": next_td,
        "td_init": td_init,
    }


def generate_random_preference_matrix(batch_size, locs, num_splits):
    """Generate preference matrix based on angle of nodes relative to depot.
    Args:
        - depot: [batch_size, 2]
        - num_splits: int, i.e. number of agents
        - num_nodes: int, i.e. number of nodes
    Returns:
        preference: [batch_size, num_splits, num_nodes]
    """
    batch_size, num_nodes, _ = locs.shape
    preference = np.random.uniform(0, 1, size=(batch_size, num_splits, num_nodes))
    return preference


def generate_pi_split_preference_matrix(depot, locs, num_splits):
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
    relative_pos = locs - np.expand_dims(depot, 1)

    # Calculate angles of nodes relative to depot (in radians)
    angles = np.arctan2(relative_pos[..., 1], relative_pos[..., 0])

    # Shift angles to range [0, 2π]
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    # Calculate angle range for each split
    split_angle = 2 * np.pi / num_splits

    # Initialize preference array
    preference = np.zeros((batch_size, num_splits, num_nodes), dtype=int)

    # Assign nodes to splits based on their angles
    for i in range(num_splits):
        start_angle = i * split_angle
        end_angle = (i + 1) * split_angle
        mask = (angles >= start_angle) & (angles < end_angle)
        preference[:, i, :] = mask.astype(int)

    return preference


def generate_cluster_preference_matrix(depot, locs, num_splits):
    """Generate preference matrix based on angle of nodes relative to depot.
    Args:
        - depot: [batch_size, 2]
        - locs: [batch_size, num_nodes, 2], note here the locs does not include the depot
        - num_splits: int, i.e. number of agents
    Returns:
        preference: [batch_size, num_splits, num_nodes]
    """
    batch_size, num_nodes, _ = locs.shape

    # Randomly select cluster centers
    cluster_indices = np.random.randint(0, num_nodes, size=(batch_size, num_splits))
    cluster_centers = np.take_along_axis(
        locs, cluster_indices[:, :, np.newaxis], axis=1
    )

    # Calculate distances from each node to each cluster center
    distances = np.linalg.norm(
        locs[:, np.newaxis] - cluster_centers[:, :, np.newaxis], axis=-1
    )

    # Normalize distances by sqrt(2)
    normalized_distances = distances / np.sqrt(2)

    # Convert distances to preferences (smaller distance = higher preference)
    preferences = 1 - normalized_distances

    return preferences


def convert_seq_actions(td, veh, pi):
    """Convert the sequence actions to the parallel version

    Args:
        - td <TensorDict>: the tensor dictionary
        - veh <torch.Tensor>: [batch_size, seq_len], the vehicle index
        - pi <torch.Tensor>: [batch_size, seq_len], the node index

    Returns:
        - actions <torch.Tensor>: [batch_size, num_agents, seq_len], the parallel version of the actions
    """
    batch_size = veh.size(0)
    seq_length = veh.size(-1)
    num_agents = td["capacity"].size(-1)

    actions = torch.zeros(batch_size, num_agents, seq_length, dtype=torch.int64)

    for batch_idx in range(batch_size):
        for seq_idx in range(seq_length):
            actions[batch_idx, :, seq_idx] = actions[batch_idx, :, seq_idx - 1]
            actions[batch_idx, veh[batch_idx, seq_idx], seq_idx] = pi[
                batch_idx, seq_idx
            ]

    return actions
