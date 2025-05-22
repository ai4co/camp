import torch

from rl4co.utils.ops import gather_by_index
from torch_geometric.data import Batch, Data


def replace_key_td(td, key, replacement):
    td.pop(key)
    td[key] = replacement
    return td


def resample_batch(td, num_agents, num_locs):
    # Remove depots until num_agents
    td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))
    if "depots" in td.keys():
        # note that if we have "depot" instead, this will automatically
        # be repeated inside the environment
        td = replace_key_td(td, "depots", td["depots"][..., :num_agents, :])

    if "pickup_et" in td.keys():
        # Ensure num_locs is even for omdcpdp
        num_locs = num_locs - 1 if num_locs % 2 == 0 else num_locs
        # also, set the "num_agents" key to the new number of agents
        td.set_(
            "num_agents", torch.full((*td.batch_size,), num_agents, device=td.device)
        )

    td = replace_key_td(td, "locs", td["locs"][..., :num_locs, :])

    # For early time windows
    if "pickup_et" in td.keys():
        td = replace_key_td(td, "pickup_et", td["pickup_et"][..., : num_locs // 2])
    if "delivery_et" in td.keys():
        td = replace_key_td(td, "delivery_et", td["delivery_et"][..., : num_locs // 2])

    # Capacities
    if "capacity" in td.keys():
        td = replace_key_td(td, "capacity", td["capacity"][..., :num_agents])

    if "speed" in td.keys():
        td = replace_key_td(td, "speed", td["speed"][..., :num_agents])

    if "demand" in td.keys():
        td = replace_key_td(td, "demand", td["demand"][..., :num_locs])

    # Preference Matrix
    if "preference" in td.keys():
        td = replace_key_td(
            td, "preference", td["preference"][..., :num_agents, :num_locs]
        )

    return td


def get_log_likelihood(log_p, actions=None, mask=None, return_sum: bool = False):
    """Get log likelihood of selected actions

    Args:
        log_p: [batch, n_agents, (decode_len), n_nodes]
        actions: [batch, n_agents, (decode_len)]
        mask: [batch, n_agents, (decode_len)]
    """

    # NOTE: we do not use this since it is more inefficient, we do it in the decoder
    if actions is not None:
        if log_p.dim() > 3:
            log_p = gather_by_index(log_p, actions, dim=-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[mask] = 0

    assert (
        log_p > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return log_p.sum(-1)  # [batch, num_agents]
    else:
        return log_p  # [batch, num_agents, (decode_len)]


def create_batched_pyg_data(node_emb, edge_emb):
    """
    Create batched pyg data object for GNN training.

    Parameters:
    - node_emb: Tensor of shape (B, m, h) where:
        B = batch size, m = number of source nodes per graph, h = node feature size
    - edge_emb: Tensor of shape (B, m, n, h) where:
        B = batch size, m = number of source nodes, n = number of destination nodes, h = edge feature size

    Returns:
    - batched_data: A batched pyg Data object containing all graphs in the batch
    """
    B, m, h_node = node_emb.shape
    _, _, n, h_edge = edge_emb.shape

    # Flatten node features to shape (B*m, h)
    node_features = node_emb.reshape(B * m, h_node)  # Shape (B*m, h_node)

    # Global node indexing for batching (each graph's nodes are shifted by m for src nodes)
    src_batch_offsets = torch.arange(B, device=node_emb.device) * m  # Shape (B,)

    # For destination nodes, the batch offset should use `n` if m != n
    dest_batch_offsets = torch.arange(B, device=node_emb.device) * n  # Shape (B,)

    # Create local edge indices (src, dest) for a single graph
    src_indices = torch.arange(m, device=node_emb.device).repeat_interleave(
        n
    )  # Shape (m*n,)
    dest_indices = torch.arange(n, device=node_emb.device).repeat(m)  # Shape (m*n,)

    # Adjust for global indexing across the batch
    # For src_indices, shift by `m`
    src_indices = (
        src_indices.unsqueeze(0) + src_batch_offsets.view(-1, 1)
    ).flatten()  # Shape (B*m*n,)

    # For dest_indices, shift by `n`
    dest_indices = (
        dest_indices.unsqueeze(0) + dest_batch_offsets.view(-1, 1)
    ).flatten()  # Shape (B*m*n,)

    # Stack src and dest indices to form edge_index of shape (2, B*m*n)
    edge_index = torch.stack([src_indices, dest_indices], dim=0)

    # Flatten edge features (edge_emb) to shape (B*m*n, h_edge)
    edge_attr = edge_emb.reshape(B * m * n, h_edge)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    # Create batched data object (single batch with multiple graphs)
    batched_data = Batch.from_data_list([data])

    return batched_data
