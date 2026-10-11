from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np

from problems.vrp.state_cvrp import StateCVRP, StateCVRPTW
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class CVRPTW(object):

    NAME = 'cvrptw'  # Capacitated Vehicle Routing Problem with Time Windows

    VEHICLE_CAPACITY = 1.0  # 车辆容量为1，demands需要被缩放

    WORK_TIME = 8.0  # 一天工作8小时（T_w）
    TIME_WINDOW_LENGTH = 2  # 时间窗的长度为2小时，即一天被分为4个时间窗分别是9:00-11:00, 11:00-13:00, 13:00-15:00, 15:00-17:00，客户的时间窗是从这4个时间窗中选择的

    VEHICLE_SPEED = 2  # speed of the vehicle 领域边长为1.0，车辆每小时行驶2个坐标单位（S_v）
    PENALTY_LAMBDA = 3.0  # 惩罚缩放因子λ，公式(10)：β = λ·T_w，默认λ=3即β=24（每小时迟到）
    LATE_EPS = 1e-6  # 迟到超过该值（小时）才计为迟到客户，避免浮点舍入把准时到达记为迟到

    @staticmethod
    def penalty():
        """惩罚系数β = λ·T_w（每小时迟到）"""
        return CVRPTW.PENALTY_LAMBDA * CVRPTW.WORK_TIME

    @staticmethod
    def set_penalty_lambda(penalty_lambda):
        CVRPTW.PENALTY_LAMBDA = float(penalty_lambda)

    @staticmethod
    def get_costs(dataset, pi):
        """
        混合成本，公式(1)：J(Y) = Σ_j [l(Y_j) + β·t_vio(Y_j)]
        :param dataset: 字典，包含了问题的所有数据（depot、loc、demand、time windows）
        :param pi: 张量(batch_size, steps)，表示一种可能的路径，即节点的访问顺序，0表示回到仓库
        :return cost: (batch_size,) 混合成本 = 路径总长度 + β·总迟到时长（小时）
        :return mask: None
        :return lateness: (batch_size,) 总迟到时长（小时）
        :return n_late: (batch_size,) 迟到的客户数量
        :return route_length: (batch_size,) 路径总长度
        """
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        """
        验证路径是否合法，即路径中包含节点1到n
        """
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        """
        将仓库的demand设置为-VEHICLE_CAPACITY，与其他客户的demand拼接，得到[batch_size, graph_size+1]
        然后按照路径pi，将demand取出，得到[batch_size, steps]，逐步累计已用容量，验证是否超过容量
        """
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRPTW.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        exceeded = torch.zeros_like(used_cap, dtype=torch.bool)
        for i in range(pi.size(1)):
            used_cap = torch.clamp(used_cap + d[:, i], min=0)  # This will reset capacity if the depot is visited
            exceeded = exceeded | (used_cap > CVRPTW.VEHICLE_CAPACITY + 1e-5)
        assert not exceeded.any(), "Used more than capacity"

        """
        拼接仓库坐标和节点坐标，得到[batch_size, graph_size+1, 2]
        按照路径pi，将坐标取出，得到[batch_size, steps, 2]
        计算路径pi的欧几里得距离，加上从仓库出发到第一个节点，加上从最后一个节点到仓库
        """
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        pi_coords = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        first_to_last = (pi_coords[:, 1:] - pi_coords[:, :-1]).norm(p=2, dim=2).sum(1) # [batch_size] 每个节点到下一个节点的距离之和
        depot_to_first = (pi_coords[:, 0] - dataset['depot']).norm(p=2, dim=1) # [batch_size] 第一个节点到仓库
        last_to_depot = (pi_coords[:, -1] - dataset['depot']).norm(p=2, dim=1) # [batch_size] 最后一个节点到仓库
        route_length = first_to_last + depot_to_first + last_to_depot # [batch_size] 总路径长度

        lateness, n_late = CVRPTW.calculate_time_windows_violations(dataset, pi)

        cost = route_length + CVRPTW.penalty() * lateness

        return cost, None, lateness, n_late, route_length

    @staticmethod
    def calculate_time_windows_violations(dataset, pi):
        """
        按算法1的时间模型计算每个实例的总迟到时长与迟到客户数：
        每条路径由一辆车从仓库于时刻0出发；到达客户 T_a' = max(T_a + D/S_v, x^s)（早到等待）；迟到量 max(0, T_a' - x^t)；
        回到仓库时开始下一条路径，时间清零。时间以小时计，运算顺序与StateCVRPTW.update一致。
        :param dataset: 字典，包含了问题的所有数据（depot、loc、demand、tw）
        :param pi: 张量(batch_size, steps)，表示一种可能的路径，即节点的访问顺序
        :return lateness: (batch_size,) 总迟到时长（小时）
        :return n_late: (batch_size,) 迟到的客户数量
        """
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # [batch_size, graph_size+1, 2]
        pi_coords = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))  # [batch_size, steps, 2]
        tw_with_depot = torch.cat((torch.zeros_like(dataset['tw'][:, :1, :]), dataset['tw']), 1) * CVRPTW.WORK_TIME  # 仓库一行不参与计算
        pi_tw = tw_with_depot.gather(1, pi[..., None].expand(*pi.size(), 2))  # [batch_size, steps, 2] 时间窗（小时）

        prev_coord = dataset['depot']
        clock = torch.zeros_like(pi_coords[:, 0, 0])
        lateness = torch.zeros_like(clock)
        n_late = torch.zeros_like(clock)
        for i in range(pi.size(1)):
            cur_coord = pi_coords[:, i]
            arrival = clock + (cur_coord - prev_coord).norm(p=2, dim=-1) / CVRPTW.VEHICLE_SPEED
            is_depot = pi[:, i] == 0
            clock = torch.where(is_depot, torch.zeros_like(arrival), torch.max(arrival, pi_tw[:, i, 0]))
            late = torch.where(is_depot, torch.zeros_like(clock), torch.clamp(clock - pi_tw[:, i, 1], min=0))
            lateness = lateness + late
            n_late = n_late + (late > CVRPTW.LATE_EPS).float()
            prev_coord = cur_coord

        return lateness, n_late

    @staticmethod
    def count_vehicles(pi):
        """
        车辆数 = 路径条数，即序列中每一段以客户开头的连续客户串（仓库0为分隔符）
        :param pi: (batch_size, steps)
        :return: (batch_size,)
        """
        starts = (pi[:, 1:] != 0) & (pi[:, :-1] == 0)
        return (pi[:, 0] != 0).long() + starts.long().sum(1)

    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRPTW.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    # 输入dataset：字典，包含了问题的所有数据（depot、loc、demand）
    # 输入pi：张量(batch_size, steps)，表示一种可能的路径，即节点的访问顺序
    # 输出cost：路径的总长度
    # 输出Mask：None
    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }

def make_instance_cvrptw(args):
    depot, loc, demand, capacity, tw, worktime, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'tw': torch.tensor(tw, dtype=torch.float) / worktime
    }

CVRPTW_CAPACITIES = {
    10: 20.,
    20: 30.,
    30: 35.,
    50: 40.,
    70: 45.,
    100: 50.
}


class CVRPTWDataset(Dataset):
    """
    实例以堆叠张量保存：depot (N, 2)，loc (N, n, 2)，demand (N, n)（按容量归一），tw (N, n, 2)（按T_w归一）。
    下标可以是整数，也可以是一批下标（list/张量），后者直接返回整批数据，配合BatchSampler使用可省去逐条拼接。
    """

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super().__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            data = data[offset:offset+num_samples]
            depot, loc, demand, capacity, tw, worktime = (np.array(x, dtype=np.float32) for x in zip(*data))
            self.data = {
                'depot': torch.from_numpy(depot),
                'loc': torch.from_numpy(loc),
                'demand': torch.from_numpy(demand / capacity[:, None]),
                'tw': torch.from_numpy(tw / worktime[:, None, None])
            }

        else:

            """
            生成时间窗数据：左时间窗从[0,2,4,6]中随机选择，右时间窗为左时间窗+2（小时），再按T_w归一
            需求为1–9的均匀整数，按容量归一；坐标为[0,1]²上的均匀分布（PyTorch随机数）
            """
            time_windows_left_options = torch.arange(0, CVRPTW.WORK_TIME, CVRPTW.TIME_WINDOW_LENGTH)  # [0, 2, 4, 6]
            time_windows_right_options = time_windows_left_options + CVRPTW.TIME_WINDOW_LENGTH  # [2, 4, 6, 8]
            time_windows = torch.stack((time_windows_left_options, time_windows_right_options), dim=-1)  # [[0, 2], [2, 4], [4, 6], [6, 8]]

            self.data = {
                'loc': torch.FloatTensor(num_samples, size, 2).uniform_(0, 1),
                'depot': torch.FloatTensor(num_samples, 2).uniform_(0, 1),
                # Uniform 1 - 9, scaled by capacities
                'demand': (torch.FloatTensor(num_samples, size).uniform_(0, 9).int() + 1).float() / CVRPTW_CAPACITIES[size],
                'tw': time_windows[torch.randint(0, time_windows.shape[0], (num_samples, size))] / CVRPTW.WORK_TIME
            }

        self.size = self.data['loc'].size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = torch.as_tensor(idx, dtype=torch.long)
        return {k: v[idx] for k, v in self.data.items()}


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]