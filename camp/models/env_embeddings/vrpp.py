import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from camp.models.nn.positional_encoder import PositionalEncoder
from camp.models.nn.transformer import (
    Normalization,
    TransformerBlock as CommunicationLayer,
)

from .communication import BaseMultiAgentContextEmbedding


class PVRPInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        linear_bias: bool = False,
        demand_scaler: float = 40.0,
        speed_scaler: float = 1.0,
        preference_scaler: float = 1.0,
        use_polar_feats: bool = True,
    ):
        super(PVRPInitEmbedding, self).__init__()
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

        # New layer for final embedding combination
        self.final_projection = nn.Linear(3 * embed_dim, embed_dim, linear_bias)

        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.preference_scaler = preference_scaler
        self.use_polar_feats = use_polar_feats

    def forward(self, td):
        num_agents = td["action_mask"].shape[-2]
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

        # Preference modeling
        preferences = td["agents_preference"] / self.preference_scaler  # [B, m, m+N]
        preference_embed = self.preference_projection(
            preferences.unsqueeze(-1)
        )  # [B, m, m+N, embed_dim]

        # Prepare embeddings for combination
        B, m = agents_embedding.shape[:2]
        N = clients_embedding.shape[1]

        # Combine agent and client embeddings
        nodes_embedding = torch.cat(
            [agents_embedding, clients_embedding], dim=1
        )  # [B, m+N, embed_dim]
        nodes_expanded = nodes_embedding.unsqueeze(1).expand(B, m, m + N, -1)

        # Expand depot_agents_embedding
        depot_agents_expanded = depot_agents_embedding.unsqueeze(2).expand(
            B, m, m + N, -1
        )

        # Combine embeddings using the final projection
        combined_input = torch.cat(
            [nodes_expanded, preference_embed, depot_agents_expanded], dim=-1
        )
        combined_embedding = self.final_projection(combined_input)

        return nodes_embedding, combined_embedding  # [B, M, M+N, H]


class PVRPContextEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        use_communication=True,
        use_final_norm=False,
        num_communication_layers=1,
        demand_scaler=40.0,
        speed_scaler=1.0,
        use_time_to_depot=True,
        **communication_kwargs,  # note: see TransformerBlock
    ):
        super(PVRPContextEmbedding, self).__init__()
        self.embed_dim = embed_dim

        # Feature projection
        self.proj_agent_feats = nn.Linear(3, embed_dim)
        self.proj_global_feats = nn.Linear(1, embed_dim)
        self.project_context = nn.Linear(embed_dim * 4, embed_dim)

        if use_communication:
            self.communication_layers = nn.Sequential(
                *(
                    CommunicationLayer(
                        embed_dim=embed_dim,
                        **communication_kwargs,
                    )
                    for _ in range(num_communication_layers)
                )
            )
        else:
            self.communication_layers = nn.Identity()

        self.norm = (
            Normalization(embed_dim, communication_kwargs.get("normalization", "rms"))
            if use_final_norm
            else None
        )

        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.use_time_to_depot = use_time_to_depot

    def _agent_state_embedding(self, embeddings, td, num_agents, num_clients):
        context_feats = torch.stack(
            [
                td["current_length"]
                / (td["agents_speed"] / self.speed_scaler),  # current time
                (td["agents_capacity"] - td["used_capacity"])
                / self.demand_scaler,  # remaining capacity
            ],
            dim=-1,
        )
        if self.use_time_to_depot:
            depot = td["locs"][..., 0:1, :]
            cur_loc = gather_by_index(td["locs"], td["current_node"], dim=-2)
            dist_to_depot = torch.norm(cur_loc - depot, p=2, dim=-1, keepdim=True)
            time_to_depot = dist_to_depot / (
                td["agents_speed"][..., None] / self.speed_scaler
            )
            context_feats = torch.cat([context_feats, time_to_depot], dim=-1)
        return self.proj_agent_feats(context_feats)

    def _global_state_embedding(self, embeddings, td, num_agents, num_clients):
        global_feats = torch.cat(
            [
                td["visited"][..., num_agents:].sum(-1)[..., None]
                / num_clients,  # number of visited cities / total
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)[..., None, :].repeat(
            1, num_agents, 1
        )

    def forward(self, embeddings, td):
        # Collect embeddings
        b, m, mn, hdim = embeddings.shape
        n = mn - m  # number of clients

        cur_node_embedding = gather_by_index(embeddings, td["current_node"], dim=-2)

        depot_embedding = gather_by_index(embeddings, td["depot_node"], dim=-2)

        agent_state_embed = self._agent_state_embedding(
            embeddings, td, num_agents=m, num_clients=n
        )  # [B, M, hdim]
        global_embed = self._global_state_embedding(
            embeddings, td, num_agents=m, num_clients=n
        )  # [B, M, hdim]
        context_embed = torch.cat(
            [cur_node_embedding, depot_embedding, agent_state_embed, global_embed],
            dim=-1,
        )
        # [B, M, hdim, 4] -> [B, M, hdim]
        context_embed = self.project_context(context_embed)
        h_comm = self.communication_layers(context_embed)
        if self.norm is not None:
            h_comm = self.norm(h_comm)
        return h_comm


class PVRPSeqInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        linear_bias: bool = False,
        demand_scaler: float = 40.0,
        speed_scaler: float = 1.0,
        preference_scaler: float = 1.0,
        use_polar_feats: bool = True,
        num_agents: int = 5,
        num_clients: int = 50,
    ):
        super(PVRPSeqInitEmbedding, self).__init__()
        # depot feats: [x0, y0]
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.pos_embedding_proj = nn.Linear(embed_dim, embed_dim, linear_bias)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        # Note: hardcode the preference embedding for now
        # agent feats: [x0, y0, capacity, speed] + [num_nodes]
        self.init_embed_agents = nn.Linear(
            4 + num_agents + num_clients, embed_dim, linear_bias
        )
        # combine depot and agent embeddings
        self.init_embed_depot_agents = nn.Linear(2 * embed_dim, embed_dim, linear_bias)
        # client feats: [x, y, demand]
        client_feats_dim = 5 if use_polar_feats else 3
        self.init_embed_clients = nn.Linear(client_feats_dim, embed_dim, linear_bias)

        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.preference_scaler = preference_scaler
        self.use_polar_feats = use_polar_feats

    def forward(self, td):
        num_agents = td["action_mask"].shape[-2]  # [B, m, m+N]
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
                td["agents_preference"] / self.preference_scaler,
            ],
            dim=-1,
        )
        agents_embedding = self.init_embed_agents(agents_feats)

        # Combine depot and agents embeddings
        depot_agents_feats = torch.cat([depot_embedding, agents_embedding], dim=-1)
        depot_agents_embedding = self.init_embed_depot_agents(depot_agents_feats)

        # Clients embedding
        demands = td["demand"][
            ..., 0, num_agents:
        ]  # [B, N] , note that demands is repeated but the same in the beginning
        clients_feats = torch.cat(
            [clients_locs, demands[..., None] / self.demand_scaler], dim=-1
        )

        if self.use_polar_feats:
            # Convert to polar coordinates
            depot = depot_locs[..., 0:1, :]
            client_locs_centered = clients_locs - depot  # centering
            dist_to_depot = torch.norm(client_locs_centered, p=2, dim=-1, keepdim=True)
            angle_to_depot = torch.atan2(
                client_locs_centered[..., 1:], client_locs_centered[..., :1]
            )
            clients_feats = torch.cat(
                [clients_feats, dist_to_depot, angle_to_depot], dim=-1
            )

        clients_embedding = self.init_embed_clients(clients_feats)

        return torch.cat(
            [depot_agents_embedding, clients_embedding], -2
        )  # [B, m+N, hdim]


class PVRPSeqContextEmbedding(BaseMultiAgentContextEmbedding):
    def __init__(
        self,
        embed_dim,
        agent_feat_dim=2,
        global_feat_dim=1,
        demand_scaler=40.0,
        speed_scaler=1.0,
        use_time_to_depot=True,
        num_agents=5,
        num_clients=50,
        **kwargs,
    ):
        if use_time_to_depot:
            agent_feat_dim += 1
        super(PVRPSeqContextEmbedding, self).__init__(
            embed_dim, agent_feat_dim, global_feat_dim, **kwargs
        )
        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.use_time_to_depot = use_time_to_depot

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        context_feats = torch.stack(
            [
                td["current_length"]
                / (td["agents_speed"] / self.speed_scaler),  # current time
                (td["agents_capacity"] - td["used_capacity"])
                / self.demand_scaler,  # remaining capacity
            ],
            dim=-1,
        )
        if self.use_time_to_depot:
            depot = td["locs"][..., 0:1, :]
            cur_loc = gather_by_index(td["locs"], td["current_node"])
            dist_to_depot = torch.norm(cur_loc - depot, p=2, dim=-1, keepdim=True)
            time_to_depot = dist_to_depot / (
                td["agents_speed"][..., None] / self.speed_scaler
            )
            context_feats = torch.cat([context_feats, time_to_depot], dim=-1)
        return self.proj_agent_feats(context_feats)

    def _global_state_embedding(self, embeddings, td, num_agents, num_cities):
        global_feats = torch.cat(
            [
                td["visited"][..., num_agents:].sum(-1)[..., None]
                / num_cities,  # number of visited cities / total
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)[..., None, :].repeat(
            1, num_agents, 1
        )
