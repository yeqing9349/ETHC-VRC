"""
TPA-Reward（Zong et al., Reinforcement Learning for Solving Multiple Vehicle Routing Problem with Time Window,
ACM TIST 15(2):32, 2024）在本文 CVRPTW 上的实现，结构按原文第4节：
- 上下文嵌入（式8）：x_i = W_x [L_i, d_i, w1_i, w2_i] + b_x，L_i 取坐标嵌入；
- 上下文编码器（式9、10）：多头自注意力与前馈层，均为残差连接；图嵌入为客户嵌入的均值（式11）；
- 上下文解码器（式12–16）：指针注意力，状态向量为 [ĥ, h_{v_{t-1}}, h_{v_1}]，t=1 时用可学习占位向量，
  u_i = C·tanh(q·k_i/√d_k)，已访问客户为 -∞，只决定客户的访问顺序；
- 任务调度器（式18）：把解码出的客户依次分配给 K 辆车。训练与采样用概率调度器（原文表4中效果最好），
  贪心解码取概率最大的车辆，即"时间平均"规则；容量不足的车辆先从候选中排除，全部不足时被选车辆回仓库装货。
  调度器无参数，解的对数概率只来自解码器（式17）。
为适配本文问题所做的改动（论文第5.1.2节说明）：
- 用坐标嵌入而非原文默认的代价矩阵嵌入：后者的输入宽度随客户数变化，无法做跨规模测试，
  而本文实例的代价矩阵就是坐标间的欧氏距离；
- 车辆数 K = round(vehicle_ratio·n)，与原文 N=20/50/100 时 K=2/5/10 一致；
- 车辆回仓库后的下一趟按本文定义算作一辆新车，从时刻0出发（与算法1和混合成本的时间模型一致）；
- 奖励为本文的混合成本（同一β）。早到要等待，所以原文惩罚中的早到项恒为0。
解码输出与 AttentionModel 相同格式的节点序列 pi（0 表示回仓库），成本由 CVRPTW.get_costs 计算。
"""
import math

import torch
from torch import nn

from nets.graph_encoder import MultiHeadAttention, SkipConnection


class TPAEncoderLayer(nn.Sequential):
    """式(9)(10)：ĥ = h + MHA(h)，h' = ĥ + FF(ĥ)"""

    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512):
        super(TPAEncoderLayer, self).__init__(
            SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)),
            SkipConnection(nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim)
            ))
        )


class TPARewardModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 problem,
                 n_encode_layers=3,
                 n_heads=8,
                 tanh_clipping=10.,
                 vehicle_ratio=0.1,
                 feed_forward_hidden=512):
        super(TPARewardModel, self).__init__()
        assert problem.NAME == 'cvrptw', "TPA-Reward is implemented for the CVRPTW only"

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.tanh_clipping = tanh_clipping
        self.vehicle_ratio = vehicle_ratio
        self.decode_type = None
        self.temp = 1.0
        self.aux_loss = None

        # 式(8)：客户特征 [x, y, d, w1, w2] 线性嵌入；仓库也作为一个节点参与编码，特征为 [x, y, 0, 0, 1]（无需求，时间窗为整个工作日）
        self.init_embed = nn.Linear(5, embedding_dim)
        self.layers = nn.Sequential(*(
            TPAEncoderLayer(n_heads, embedding_dim, feed_forward_hidden) for _ in range(n_encode_layers)
        ))
        # 式(12)：t=1 时的占位向量 y_1, y_f
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)
        # 式(13)(14)
        self.W_Q = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        self.W_K = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def set_mask_mode(self, mask_mode):
        assert mask_mode == 'soft', "TPA-Reward handles time windows only through the penalized reward"

    def n_vehicles(self, graph_size):
        return max(1, int(round(self.vehicle_ratio * graph_size)))

    def encode(self, input):
        depot = torch.cat((input['depot'], input['depot'].new_tensor([0., 0., 1.]).expand(input['depot'].size(0), 3)), -1)
        customers = torch.cat((input['loc'], input['demand'][:, :, None], input['tw']), -1)
        x = self.init_embed(torch.cat((depot[:, None, :], customers), 1))  # (batch_size, n+1, embedding_dim)
        return self.layers(x)

    def forward(self, input):
        log_p, pi = self.decode(input, self.encode(input))
        cost, mask, lateness, n_late, length = self.problem.get_costs(input, pi)
        return cost, log_p.sum(1), lateness, n_late, length

    def _select(self, probs):
        if self.decode_type == "greedy":
            return probs.max(1)[1]
        assert self.decode_type == "sampling", "Unknown decode type"
        return probs.multinomial(1).squeeze(1)

    def decode(self, input, embeddings):
        """
        :return: log_p (batch_size, n) 每步所选客户的对数概率；pi (batch_size, n + 路径条数) 节点序列
        """
        batch_size, n_nodes, embed_dim = embeddings.size()
        n = n_nodes - 1
        K = self.n_vehicles(n)
        device = embeddings.device
        rows = torch.arange(batch_size, device=device)
        T_w, S_v = self.problem.WORK_TIME, self.problem.VEHICLE_SPEED

        h = embeddings[:, 1:]  # 客户嵌入 (batch_size, n, d)
        graph = h.mean(1)  # 式(11)
        keys = self.W_K(h)  # (batch_size, n, d)
        coords = torch.cat((input['depot'][:, None, :], input['loc']), 1)  # (batch_size, n+1, 2)

        visited = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        # 任务调度器的状态：每辆车所在节点、剩余容量、当前时刻（小时）、当前路径编号
        slot_pos = torch.zeros(batch_size, K, dtype=torch.long, device=device)
        slot_cap = torch.full((batch_size, K), self.problem.VEHICLE_CAPACITY, device=device)
        slot_time = torch.zeros(batch_size, K, device=device)
        route_id = torch.arange(K, device=device).repeat(batch_size, 1)
        next_route = torch.full((batch_size,), K, dtype=torch.long, device=device)

        log_ps, customers, routes = [], [], []
        first = prev = None
        for t in range(n):
            # 式(12)–(16)：指针注意力选择下一个客户
            if t == 0:
                context = torch.cat((graph, self.W_placeholder[None, :].expand(batch_size, -1)), -1)
            else:
                context = torch.cat((graph, h[rows, prev], h[rows, first]), -1)
            query = self.W_Q(context)
            u = torch.matmul(keys, query[:, :, None]).squeeze(-1) / math.sqrt(embed_dim)
            u = self.tanh_clipping * torch.tanh(u)
            u = u.masked_fill(visited, -math.inf)
            log_p = torch.log_softmax(u / self.temp, dim=-1)
            c = self._select(log_p.exp())
            log_ps.append(log_p.gather(1, c[:, None]).squeeze(1))
            visited = visited.scatter(1, c[:, None], True)
            if t == 0:
                first = c
            prev = c

            # 式(18)：任务调度器为客户 c 选择车辆
            demand = input['demand'][rows, c]
            enough = slot_cap >= demand[:, None] - 1e-6
            candidate = torch.where(enough.any(1, keepdim=True), enough, torch.ones_like(enough))
            busy = slot_time * candidate
            total = busy.sum(1, keepdim=True)
            k_cand = candidate.sum(1, keepdim=True).float()
            rho = torch.where((k_cand > 1) & (total > 0), (total - busy) / ((k_cand - 1).clamp(min=1) * total.clamp(min=1e-12)),
                              1. / k_cand) * candidate
            slot = self._select(rho)

            # 被选车辆容量不足（所有车辆都不足）时回仓库装货，下一趟算一辆新车，从时刻0出发
            reload = ~enough[rows, slot]
            if reload.any():
                r = rows[reload]
                s = slot[reload]
                slot_pos[r, s] = 0
                slot_cap[r, s] = self.problem.VEHICLE_CAPACITY
                slot_time[r, s] = 0.
                route_id[r, s] = next_route[r]
                next_route[r] += 1

            # 车辆驶向客户 c：早到则等待到时间窗开启
            dist = (coords[rows, c + 1] - coords[rows, slot_pos[rows, slot]]).norm(p=2, dim=-1)
            arrival = slot_time[rows, slot] + dist / S_v
            slot_time[rows, slot] = torch.max(arrival, input['tw'][rows, c, 0] * T_w)
            slot_cap[rows, slot] = slot_cap[rows, slot] - demand
            slot_pos[rows, slot] = c + 1
            customers.append(c + 1)
            routes.append(route_id[rows, slot])

        return torch.stack(log_ps, 1), self._to_sequence(torch.stack(customers, 1), torch.stack(routes, 1))

    @staticmethod
    def _to_sequence(customers, routes):
        """
        按路径分组、组内按访问先后排列客户，路径之间插入仓库0，得到节点序列 pi
        :param customers: (batch_size, n) 第 t 步分配的客户
        :param routes: (batch_size, n) 第 t 步客户所属的路径编号
        """
        batch_size, n = customers.size()
        order = torch.argsort(routes * n + torch.arange(n, device=customers.device)[None, :], dim=1)
        sorted_customers = customers.gather(1, order)
        sorted_routes = routes.gather(1, order)
        new_route = torch.cat((torch.zeros_like(sorted_routes[:, :1]), (sorted_routes[:, 1:] != sorted_routes[:, :-1]).long()), 1)
        position = torch.arange(n, device=customers.device)[None, :] + new_route.cumsum(1)
        pi = customers.new_zeros(batch_size, int(position.max().item()) + 2)
        return pi.scatter(1, position, sorted_customers)
