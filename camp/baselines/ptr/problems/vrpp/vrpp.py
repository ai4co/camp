import torch

from tensordict.tensordict import TensorDict

from camp.envs.PVRP_seq import PVRPPSeqEnv


class PVRPEnv:
    def __init__(self, input, scale=(1, 40, 1)):
        """
        :param input:
        input:{
            'loc':  batch_size, graph_size, 2
            'demand': batch_size, graph_size
            'depot': batch_size, 2
            'capacity': batch_size, vehicle_num
            'speed': batch_size, vehicle_num
        }
        :param scale: used to output normalized state (coords,demand,speed)
        """
        self.device = input["loc"].device
        self.batch_size = input["loc"].shape[0]
        self.bs_index = torch.arange(self.batch_size, device=self.device)
        self.step = 0
        self.scale_coords, self.scale_demand, self.scale_speed = scale
        self.initial_node_state(input["loc"], input["demand"], input["depot"])
        self.initial_veh_state(input["capacity"], input["speed"])

        # SECTION: Init the inner env from camp
        graph_size = input["loc"].shape[1]
        vehicle_num = input["capacity"].shape[1]
        self.inner_env = PVRPPSeqEnv(
            generator_params={
                "num_loc": graph_size,
                "num_agents": vehicle_num,
                "preference_distribution": "pi",
            }
        )

        # Use the input from the 2D-Ptr's setting
        td = TensorDict(
            {
                "locs": input["loc"],
                "depot": input["depot"],
                "num_agents": torch.full((self.batch_size,), vehicle_num),
                "demand": self.demand[..., 1:],  # No need to include the depot demand
                "capacity": self.veh_capacity,
                "speed": self.veh_speed,
                "preference": torch.rand(
                    (self.batch_size, vehicle_num, graph_size),
                    device=self.device,
                ),
            }
        ).to(self.device)
        self.td = self.inner_env.reset(td=td, batch_size=self.batch_size)

        # Use the input from camp's setting
        # self.td = self.inner_env.reset(batch_size=self.batch_size).to(self.device)

        self.initial_node_state(
            loc=self.td["locs"][..., vehicle_num:, :],
            demand=self.td["demand"][..., 0, vehicle_num:],
            depot=self.td["locs"][..., 0, :],
        )
        self.initial_veh_state(
            capacity=self.td["capacity"],
            speed=self.td["speed"],
        )

        self.agents_preference = self.td["agents_preference"]

    def initial_node_state(self, loc, demand, depot):
        """
        :param loc:  customer coordinates [batch_size, graph_size,2]
        :param demand: customer demands [batch_size, graph_size]
        :param depot: depot coordinates [batch_size, 2]
        :return:
        """
        assert (
            loc.shape[:2] == demand.shape
        ), "The custumer's loc and demand shape do not match"
        self.customer_num = loc.shape[1]
        self.N = loc.shape[1] + 1  # Let N represent the graph size
        self.coords = torch.cat([depot.unsqueeze(1), loc], dim=1)  # batch_size, N, 2
        self.demand = torch.cat(
            [torch.zeros_like(demand[:, [0]]), demand], dim=1
        )  # batch_size, N
        self.visited = torch.zeros_like(self.demand).bool()  # batch_size, N
        self.visited[:, 0] = True  # start from depot, so depot is visited

    def all_finished(self):
        """
        :return: Are all tasks finished?
        """
        return self.td["done"].all()

    def finished(self):
        """
        :return: [bs],true or false, is each task finished?
        """
        return self.td["done"]

    def get_all_node_state(self):
        """
        This function is for the init embedding. Here 3 means three features.

        :return: [bs,N+1,3], get node initial features
        """
        return torch.cat(
            [
                self.coords / self.scale_coords,
                self.demand.unsqueeze(-1) / self.scale_demand,
            ],
            dim=-1,
        )  # batch_size, N, 3

    def initial_veh_state(self, capacity, speed):
        """
        :param capacity:  batch_size, veh_num
        :param speed: batch_size, veh_num
        :return
        """
        assert (
            capacity.size() == speed.size()
        ), "The vehicle's speed and capacity shape do not match"
        self.veh_capacity = capacity
        self.veh_speed = speed
        self.veh_num = capacity.shape[1]
        self.veh_time = torch.zeros_like(capacity)  # batch_size, veh_num
        self.veh_cur_node = torch.zeros_like(capacity).long()  # batch_size, veh_num
        self.veh_used_capacity = torch.zeros_like(capacity)
        # a util vector
        self.veh_index = torch.arange(self.veh_num, device=self.device)

    def min_max_norm(self, data):
        """
        deprecated
        :param data:
        :return:
        """
        # bs，M
        min_data = data.min(-1, keepdim=True)[0]
        max_data = data.max(-1, keepdim=True)[0]
        return (data - min_data) / (max_data - min_data)

    def get_all_veh_state(self):
        """
        :return: [bs,M,4]
        # time，capacity，usage capacity，speed
        """

        return torch.cat(
            [
                self.veh_time.unsqueeze(-1),
                self.veh_capacity.unsqueeze(-1) / self.scale_demand,
                self.veh_used_capacity.unsqueeze(-1) / self.scale_demand,
                self.veh_speed.unsqueeze(-1) / self.scale_speed,
                self.agents_preference,
            ],
            dim=-1,
        )

    def get_veh_state(self, veh):
        # deprecated
        """
        :param veh: veh_index，batch_size
        :return:
        """
        all_veh_state = self.get_all_veh_state()  # bs,veh_num,4
        return all_veh_state[self.bs_index, veh]  # bs,4

    def action_is_legal(self, veh, next_node):
        # deprecated
        return (
            self.demand[self.bs_index, next_node]
            <= (self.veh_capacity - self.veh_used_capacity)[self.bs_index, veh]
        )

    def update(self, veh, next_node, next_node_):
        """
        input action tuple and update the env
        :param veh: [batch_size,]
        :param next_node: [batch_size,]
        :return:
        """
        # Fix the batch_size
        self.td.batch_size = [self.batch_size]

        # # Fix the next_node with the vehicle index
        # next_node_ = next_node + self.veh_num - 1

        # # Replace the `self.veh_num - 1` with the vel
        # next_node_ = torch.where(next_node_ == self.veh_num - 1, veh, next_node_)

        action = torch.stack([veh, next_node_], dim=1)
        self.td.update({"action": action})
        self.td = self.inner_env.step(self.td)["next"]

        # Update vehicle time，
        self.veh_time = self.td["current_length"]

        # Update the veihcle preference
        self.veh_preference = self.td["current_preference"]

        # Update the used_capacity
        self.veh_used_capacity = self.td["used_capacity"]

        # Update the node index where the vehicle stands
        veh_cur_node = self.td["current_node"] - self.veh_num + 1
        veh_cur_node = torch.where(veh_cur_node < 0, 0, veh_cur_node)
        self.veh_cur_node = veh_cur_node

        # Update visited vector
        self.visited[self.bs_index, next_node] = True

        self.step += 1

    def all_go_depot(self):
        """
        All vehicle go back the depot
        :return:
        """
        veh_list = torch.arange(self.veh_num, device=self.device)
        depot = torch.zeros_like(self.bs_index)
        for i in veh_list:
            next_node_ = depot + self.veh_num - 1
            next_node_ = torch.where(
                next_node_ == self.veh_num - 1, i.expand(self.batch_size), next_node_
            )
            self.update(i.expand(self.batch_size), depot, next_node_)

    def get_cost(self, obj, veh, pi):
        self.all_go_depot()
        if obj == "min-max":
            return self.inner_env._get_reward(self.td, {"veh": veh, "pi": pi})
        elif obj == "min-sum":
            return self.inner_env._get_reward(self.td, {"veh": veh, "pi": pi})

    def get_action_mask(self):
        mask = self.td["action_mask"]

        # SECTION: compress the depot mask
        agent_idx = torch.arange(self.veh_num)
        depot_mask = mask[:, agent_idx, agent_idx]

        mask = ~torch.cat([depot_mask.unsqueeze(-1), mask[..., self.veh_num :]], dim=-1)
        return mask

    @staticmethod
    def caculate_cost(input, solution, obj):
        """
        :param input: equal to __init__
        :param solution: (veh,next_node): [total_step, batch_size],[total_step, batch_size]
        :param obj: 'min-max' or 'min-sum'
        :return: cost : batch_size
        """

        env = PVRPEnv(input)
        for veh, next_node in zip(*solution):
            env.update(veh, next_node)
        return env.get_cost(obj)
