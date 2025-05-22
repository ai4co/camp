import matplotlib.pyplot as plt
import torch

from matplotlib import cm
from matplotlib.patches import Wedge


def render(
    td,
    actions,
    ax,
    plot_depot_transition=True,
    plot_node_colors=True,
    preference_pi=False,
    **kwargs,
):
    """
    Args:
        - td: not batched tensordict
        - actions: [num_agents, seq_len] tensor, also not batched
        - ax: matplotlib axis
        - plot_depot_transition: if True, nodes will be in its preference color
        - plot_node_colors: if True, nodes will be in its preference color
            Note: this only works with integer preferences
        - preference_pi: if True, the sector for the preference will be plotted
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Variables
    num_agents = td["capacity"].size(0)
    depot = td["locs"][0]
    nodes = td["locs"][num_agents:]

    # Plot the depot
    ax.scatter(
        depot[0],
        depot[1],
        marker="s",
        facecolors=cm.Set1(0),
        edgecolors="black",
        s=50,
    )

    # Plot nodes
    if plot_node_colors:
        for node_idx in range(50):
            facecolor_idx = (
                td["agents_preference"][:, node_idx + num_agents].nonzero().item() + 1
            )
            ax.scatter(
                nodes[node_idx, 0].cpu(),
                nodes[node_idx, 1].cpu(),
                marker="o",
                facecolors=cm.Set1(facecolor_idx),
                edgecolors="black",
                s=30,
            )
    else:
        ax.scatter(
            nodes[num_agents:, 0],
            nodes[num_agents:, 1],
            marker="o",
            facecolors="gray",
            edgecolors="black",
            s=30,
        )

    # Plot preference sectors if preference_pi
    if preference_pi:
        split_angle = 360 / num_agents

        for i in range(num_agents):
            start_angle = i * split_angle
            end_angle = (i + 1) * split_angle
            wedge = Wedge(
                (depot[0], depot[1]),
                5,
                start_angle,
                end_angle,
                alpha=0.2,
                color=cm.Set1(i + 1),
                zorder=1,
            )
            ax.add_patch(wedge)

    # Plot Actions
    # add as first action of all the agents the depot (which is agent_idx)
    actions_first = torch.arange(num_agents).unsqueeze(0).expand(actions.size(0), -1)
    actions = torch.cat([actions_first, actions, actions_first], dim=-1)

    for agent_idx in range(num_agents):
        agent_action = actions[agent_idx]
        for action_idx in range(agent_action.size(0) - 1):
            from_loc = td["locs"][agent_action[action_idx]]
            to_loc = td["locs"][agent_action[action_idx + 1]]

            # if it is to or from depot, raise flag
            if (
                agent_action[action_idx] == agent_idx
                or agent_action[action_idx + 1] == agent_idx
            ):
                depot_transition = True
            else:
                depot_transition = False

            if depot_transition:
                if plot_depot_transition:
                    ax.plot(
                        [from_loc[0], to_loc[0]],
                        [from_loc[1], to_loc[1]],
                        color=cm.Set1(agent_idx + 1),
                        lw=0.3,
                        linestyle="--",
                    )

            else:
                ax.plot(
                    [from_loc[0], to_loc[0]],
                    [from_loc[1], to_loc[1]],
                    color=cm.Set1(agent_idx + 1),
                    lw=1,
                )

    # Plot Configs
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # remove axs labels
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
