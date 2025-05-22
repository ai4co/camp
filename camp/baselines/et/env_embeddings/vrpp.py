import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from camp.baselines.et.nn.positional_encoder import PositionalEncoder


class PVRPSeqInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        linear_bias: bool = False,
        demand_scaler: float = 40.0,
        speed_scaler: float = 1.0,
        preference_scaler: float = 1.0,
        use_polar_feats: bool = True,
    ):
        super(PVRPSeqInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.pos_embedding_proj = nn.Linear(embed_dim, embed_dim, linear_bias)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.init_embed_agents = nn.Linear(4, embed_dim, linear_bias)
        self.init_embed_depot_agents = nn.Linear(2 * embed_dim, embed_dim, linear_bias)

        client_feats_dim = 5 if use_polar_feats else 3
        self.init_embed_clients = nn.Linear(client_feats_dim, embed_dim, linear_bias)

        self.preference_projection = nn.Linear(1, embed_dim, linear_bias)

        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.preference_scaler = preference_scaler
        self.use_polar_feats = use_polar_feats

    def forward(self, td):
        num_agents = td["speed"].shape[-1]
        depot_locs = td["locs"][..., :num_agents, :]
        agents_locs = td["locs"][..., :num_agents, :]
        clients_locs = td["locs"][..., num_agents:, :]

        # Depots embedding with positional encoding
        depots_embedding = self.init_embed_depot(depot_locs)
        pos_embedding = self.pos_encoder(depots_embedding, add=False)
        pos_embedding = self.alpha * self.pos_embedding_proj(pos_embedding)
        depot_embedding = depots_embedding + pos_embedding

        # Agents embedding
        agents_feats = torch.cat(
            [
                agents_locs,
                td["capacity"][..., None] / self.demand_scaler,
                td["speed"][..., None] / self.speed_scaler,
            ],
            dim=-1,
        )
        agents_embedding = self.init_embed_agents(agents_feats)

        # Combine depot and agents embeddings
        depot_agents_feats = torch.cat([depot_embedding, agents_embedding], dim=-1)
        depot_agents_embedding = self.init_embed_depot_agents(depot_agents_feats)

        # Clients embedding
        demands = td["demand"][..., 0, num_agents:]
        clients_feats = torch.cat(
            [clients_locs, demands[..., None] / self.demand_scaler], dim=-1
        )

        if self.use_polar_feats:
            depot = depot_locs[..., 0:1, :]
            client_locs_centered = clients_locs - depot
            dist_to_depot = torch.norm(client_locs_centered, p=2, dim=-1, keepdim=True)
            angle_to_depot = torch.atan2(
                client_locs_centered[..., 1:], client_locs_centered[..., :1]
            )
            clients_feats = torch.cat(
                [clients_feats, dist_to_depot, angle_to_depot], dim=-1
            )

        clients_embedding = self.init_embed_clients(clients_feats)

        # # Preference modeling
        preferences = td["agents_preference"] / self.preference_scaler  # [B, m, m+N]
        preference_embed = self.preference_projection(
            preferences.unsqueeze(-1)
        )  # [B, m, m+N, embed_dim]

        # Prepare embeddings for combination
        B, m = agents_embedding.shape[:2]
        N = clients_embedding.shape[1]

        # Combine agent and client embeddings
        nodes_embedding = torch.cat(
            [depot_agents_embedding, clients_embedding], dim=1
        )  # [B, m+N, embed_dim]

        return nodes_embedding, preference_embed  # [B, M+N, H], [B, M, M+N, H]


class PVRPSeqContextEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        demand_scaler=40.0,
        speed_scaler=1.0,
        use_time_to_depot=True,
    ):
        super(PVRPSeqContextEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # Feature projection
        self.proj_agent_feats = nn.Linear(4, embed_dim)
        self.proj_global_feats = nn.Linear(2, embed_dim)
        self.project_context = nn.Linear(embed_dim * 4, embed_dim)

        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.use_time_to_depot = use_time_to_depot

    def _agent_state_embedding(self, embeddings, td):
        context_feats = torch.stack(
            [
                td["current_length"]
                / (td["agents_speed"] / self.speed_scaler),  # current time
                (td["agents_capacity"] - td["used_capacity"])
                / self.demand_scaler,  # remaining capacity
            ],
            dim=-1,
        )

        depot = td["locs"][..., 0:1, :]
        dist_to_depot = torch.norm(td["locs"] - depot, p=2, dim=-1, keepdim=True)
        max_dist_to_depot = dist_to_depot.max(dim=-2, keepdim=True).values
        remain_max_dist_to_depot = (
            (dist_to_depot * (~td["visited"][..., None]))
            .max(dim=-2, keepdim=True)
            .values
        )

        max_time_to_depot = max_dist_to_depot / (
            td["agents_speed"][..., None] / self.speed_scaler
        )
        remain_max_time_to_depot = remain_max_dist_to_depot / (
            td["agents_speed"][..., None] / self.speed_scaler
        )
        context_feats = torch.cat(
            [context_feats, max_time_to_depot, remain_max_time_to_depot], dim=-1
        )

        context_feats = gather_by_index(context_feats, td["current_agent_idx"], dim=1)
        return self.proj_agent_feats(context_feats)

    def _global_state_embedding(self, embeddings, td, num_agents, num_clients):
        global_feats = torch.cat(
            [
                td["visited"][..., num_agents:].sum(-1)[..., None]
                / num_clients,  # number of visited cities / total
                td["current_agent_idx"].float() + 1 / num_agents,
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)

    def forward(self, embeddings, td):
        # Collect embeddings
        b, mn, hdim = embeddings.shape

        num_agents = td["agents_speed"].shape[-1]
        num_clients = mn - num_agents

        current_agent_pos = gather_by_index(td["current_node"], td["current_agent_idx"])

        cur_node_embedding = gather_by_index(embeddings, current_agent_pos, dim=-2)

        cur_agent_embedding = gather_by_index(
            embeddings, td["current_agent_idx"], dim=-2
        )  # [B, 1, hdim]

        agent_state_embed = self._agent_state_embedding(embeddings, td)  # [B, 1, hdim]
        global_embed = self._global_state_embedding(
            embeddings, td, num_agents=num_agents, num_clients=num_clients
        )  # [B, 1, hdim]

        context_embed = torch.cat(
            [cur_node_embedding, cur_agent_embedding, agent_state_embed, global_embed],
            dim=-1,
        )
        # [B, M, 4 * hdim, 4]
        context_embed = self.project_context(context_embed).squeeze(1)

        return context_embed
