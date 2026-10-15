"""
MVMoE（Zhou et al., MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts, ICML 2024）在本文 CVRPTW 上的实现。
网络结构移植自官方代码 https://github.com/RoyalSkye/Routing-MVMoE 的默认配置（model_type=MOE_LIGHT）：
- 编码器：仓库与客户分别线性嵌入，6 层多头注意力（8 头），每层的前馈层替换为 MoE（4 个专家、top-2、
  带噪声的节点级 input-choice 门控），Add & InstanceNorm 放在子层之后；
- 解码器：查询为 [上一节点嵌入, 剩余容量, 当前时刻, 当前路径长度, open]，多头注意力后的输出投影使用分层门控：
  第一层门控在首个解码步按整批的平均表示给出"稠密层/MoE"的概率，此后每步按该概率抽取其一（官方实现在
  贪心解码时同样抽取），输出乘以所选分支的概率；单头注意力计算选点概率，tanh 裁剪 10；
- MoE 层的负载均衡损失（importance 与 load 的变异系数平方，乘 1e-3）累加为 aux_loss，训练时加到总损失上。
适配本文问题（论文第5.1.2节说明）：
- 只训练 CVRPTW，实例来自本文的生成器；节点特征为 (x, y, d/C, x^s/T_w, x^t/T_w)，当前时刻按 T_w 归一，
  与时间窗特征同一单位；本文仓库没有时间窗，因此去掉官方环境中"必须按时返回仓库"的掩码；
- 训练与推理都使用公式(19)的硬时间窗掩码（StateCVRPTW 的 hard 模式）；
- 按本文的统一流程训练（REINFORCE + rollout 基线，单条轨迹），推理为一次贪心解码。

Portions of this file are adapted from Routing-MVMoE, released under the MIT License:

    MIT License

    Copyright (c) 2023 Jianan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


########################################
# MoE layer (node-level, input-choice routing with noisy top-k gating)
########################################

class SparseDispatcher(object):
    """把输入按门控分发给各专家，再按门控权重合并专家输出（Shazeer et al., 2017）"""

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out):
        stitched = torch.cat(expert_out, 0).mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined


class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class MoE(nn.Module):
    """稀疏门控混合专家层，返回 (输出, 负载均衡损失)"""

    def __init__(self, input_size, output_size, num_experts, hidden_size=None, k=2, T=1.0, noisy_gating=True,
                 moe_model="MLP", loss_coef=1e-3):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.T = T
        self.loss_coef = loss_coef
        if moe_model == "MLP":
            self.experts = nn.ModuleList([MLP(input_size, output_size, hidden_size) for _ in range(num_experts)])
        else:
            assert moe_model == "Linear"
            self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(num_experts)])
        # 与官方实现一致：节点级 input-choice 门控的权重初始化为 0
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    @staticmethod
    def cv_squared(x):
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + 1e-10)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in - 1), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            noise_stddev = F.softplus(x @ self.w_noise) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_gates = torch.softmax(top_logits[..., :self.k] / self.T, dim=-1)
        gates = torch.zeros_like(logits, requires_grad=True).scatter(-1, top_indices[..., :self.k], top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = (gates > 0).sum(0)
        return gates, load

    def forward(self, x):
        output_shape = list(x.size()[:-1]) + [self.output_size]
        x = x.reshape(-1, self.input_size)
        gates, load = self.noisy_top_k_gating(x, self.training)
        loss = (self.cv_squared(gates.sum(0)) + self.cv_squared(load)) * self.loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        return dispatcher.combine(expert_outputs).reshape(output_shape), loss


########################################
# Encoder / decoder
########################################

def reshape_by_heads(qkv, head_num):
    batch_s, n = qkv.size(0), qkv.size(1)
    return qkv.reshape(batch_s, n, head_num, -1).transpose(1, 2)


def multi_head_attention(q, k, v, ninf_mask=None):
    """q: (batch, head, n, key_dim)；k, v: (batch, head, problem, key_dim)；ninf_mask: (batch, n, problem)"""
    batch_s, head_num, n, key_dim = q.size()
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(key_dim)
    if ninf_mask is not None:
        score = score + ninf_mask[:, None, :, :]
    weights = torch.softmax(score, dim=3)
    return torch.matmul(weights, v).transpose(1, 2).reshape(batch_s, n, head_num * key_dim)


class AddAndInstanceNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(AddAndInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        return self.norm((input1 + input2).transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, ff_hidden_dim, num_experts, topk):
        super(EncoderLayer, self).__init__()
        self.head_num = head_num
        qkv_dim = embedding_dim // head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_norm1 = AddAndInstanceNorm(embedding_dim)
        self.feed_forward = MoE(embedding_dim, embedding_dim, num_experts, hidden_size=ff_hidden_dim, k=topk,
                                moe_model="MLP")
        self.add_norm2 = AddAndInstanceNorm(embedding_dim)

    def forward(self, x):
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        out1 = self.add_norm1(x, self.multi_head_combine(multi_head_attention(q, k, v)))
        out2, moe_loss = self.feed_forward(out1)
        return self.add_norm2(out1, out2), moe_loss


class MVMoEModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 problem,
                 n_encode_layers=6,
                 n_heads=8,
                 tanh_clipping=10.,
                 num_experts=4,
                 topk=2,
                 ff_hidden_dim=512):
        super(MVMoEModel, self).__init__()
        assert problem.NAME == 'cvrptw', "MVMoE is implemented here for the CVRPTW only"

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.head_num = n_heads
        self.tanh_clipping = tanh_clipping
        self.decode_type = None
        self.temp = 1.0
        self.T = 1.0  # 分层门控的温度
        self.aux_loss = None

        qkv_dim = embedding_dim // n_heads
        # 编码器
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, n_heads, ff_hidden_dim, num_experts, topk) for _ in range(n_encode_layers)
        ])
        # 解码器
        self.Wq_last = nn.Linear(embedding_dim + 4, n_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, n_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, n_heads * qkv_dim, bias=False)
        self.dense_or_moe = nn.Linear(n_heads * qkv_dim, 2, bias=False)
        self.multi_head_combine_moe = MoE(n_heads * qkv_dim, embedding_dim, num_experts, k=topk, moe_model="Linear")
        self.multi_head_combine_dense = nn.Linear(n_heads * qkv_dim, embedding_dim)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def set_mask_mode(self, mask_mode):
        assert mask_mode == 'hard', "MVMoE enforces hard time windows in training and inference"

    def encode(self, input):
        out = torch.cat((self.embedding_depot(input['depot'][:, None, :]),
                         self.embedding_node(torch.cat((input['loc'], input['demand'][:, :, None], input['tw']), -1))), 1)
        moe_loss = 0
        for layer in self.layers:
            out, loss = layer(out)
            moe_loss = moe_loss + loss
        self._encoder_loss = moe_loss
        return out

    def forward(self, input):
        log_p, pi = self.decode(input, self.encode(input))
        self.aux_loss = (self._encoder_loss + self._decoder_loss).sum()
        cost, mask, lateness, n_late, length = self.problem.get_costs(input, pi)
        return cost, log_p.sum(1), lateness, n_late, length

    def decode(self, input, encoded_nodes):
        """
        单条轨迹解码，掩码为公式(19)的硬时间窗掩码
        :return: log_p (batch_size, steps) 每步所选节点的对数概率；pi (batch_size, steps) 节点序列
        """
        batch_size, n_nodes, embedding_dim = encoded_nodes.size()
        rows = torch.arange(batch_size, device=encoded_nodes.device)
        k = reshape_by_heads(self.Wk(encoded_nodes), self.head_num)
        v = reshape_by_heads(self.Wv(encoded_nodes), self.head_num)
        single_head_key = encoded_nodes.transpose(1, 2)

        state = self.problem.make_state(input, hard=True)
        route_length = torch.zeros(batch_size, device=encoded_nodes.device)
        gate_probs, decoder_loss = None, 0
        log_ps, sequences = [], []
        while not state.all_finished():
            mask = state.get_mask()[:, 0, :]  # (batch_size, n_nodes)
            ninf_mask = torch.zeros(mask.size(), device=mask.device).masked_fill(mask, -math.inf)[:, None, :]
            last = state.get_current_node()[:, 0]
            attr = torch.stack((self.problem.VEHICLE_CAPACITY - state.used_capacity[:, 0],
                                state.clock[:, 0] / self.problem.WORK_TIME,
                                route_length,
                                torch.zeros_like(route_length)), 1)
            q = reshape_by_heads(self.Wq_last(torch.cat((encoded_nodes[rows, last], attr), 1))[:, None, :], self.head_num)
            out_concat = multi_head_attention(q, k, v, ninf_mask=ninf_mask)  # (batch_size, 1, head*qkv)

            # 分层门控：首个解码步给出"稠密层/MoE"的概率，此后每步按该概率选择一个分支
            if gate_probs is None:
                gate_probs = torch.softmax(self.dense_or_moe(out_concat.mean(0).mean(0).unsqueeze(0)) / self.T, dim=-1)
            selected_branch = gate_probs.multinomial(1).squeeze(0)
            if selected_branch.item() == 1:
                mh_atten_out, loss = self.multi_head_combine_moe(out_concat)
                decoder_loss = decoder_loss + loss
            else:
                mh_atten_out = self.multi_head_combine_dense(out_concat)
            mh_atten_out = mh_atten_out * gate_probs.squeeze(0)[selected_branch]

            score = torch.matmul(mh_atten_out, single_head_key).squeeze(1) / math.sqrt(embedding_dim)
            score = self.tanh_clipping * torch.tanh(score)
            log_p = torch.log_softmax(score.masked_fill(mask, -math.inf) / self.temp, dim=-1)
            if self.decode_type == "greedy":
                selected = log_p.max(1)[1]
            else:
                assert self.decode_type == "sampling", "Unknown decode type"
                selected = log_p.exp().multinomial(1).squeeze(1)
            log_ps.append(log_p.gather(1, selected[:, None]).squeeze(1))
            sequences.append(selected)

            prev_length = state.lengths[:, 0]
            state = state.update(selected)
            route_length = (route_length + state.lengths[:, 0] - prev_length) * (selected != 0).float()

        self._decoder_loss = decoder_loss
        return torch.stack(log_ps, 1), torch.stack(sequences, 1)
