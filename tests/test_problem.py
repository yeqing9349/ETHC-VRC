"""
问题层的单元测试：算法1的时间状态、公式(18)(19)的掩码与仓库规则、混合成本与指标。
运行：uv run pytest -q
"""
import math

import pytest
import torch

from problems.vrp.problem_vrp import CVRPTW
from problems.vrp.state_cvrp import StateCVRPTW


def make_instance():
    """仓库在原点；客户1在(1,0)，时间窗[2,4]h；客户2在(1,1)，[0,2]h；客户3在(0,0.5)，[6,8]h；需求按容量归一"""
    return {
        'depot': torch.tensor([[0.0, 0.0]]),
        'loc': torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 0.5]]]),
        'demand': torch.tensor([[0.4, 0.4, 0.4]]),
        'tw': torch.tensor([[[2.0, 4.0], [0.0, 2.0], [6.0, 8.0]]]) / CVRPTW.WORK_TIME,
    }


def replay(input, pi, hard=False):
    """按序列 pi 逐步更新状态，返回每步之后的状态，并检查每一步都没有选到被掩码的节点"""
    state = CVRPTW.make_state(input, hard=hard)
    states = []
    for t in range(pi.size(1)):
        mask = state.get_mask()[:, 0, :]
        assert not mask.gather(1, pi[:, t:t + 1]).any(), "step {} selects a masked node".format(t)
        state = state.update(pi[:, t])
        states.append(state)
    return states


def test_algorithm1_time_state():
    input = make_instance()
    # 路径1：仓库 -> 客户1 -> 客户2 -> 仓库；路径2：仓库 -> 客户3 -> 仓库
    pi = torch.tensor([[1, 2, 0, 3, 0]])
    states = replay(input, pi)
    # 仓库到客户1距离1，0.5h 到达，早到等待至 2h
    assert states[0].cur_arrival_time.item() == pytest.approx(2.0)
    # 客户1到客户2距离1，2.5h 到达，时间窗[0,2]，迟到 0.5h
    assert states[1].cur_arrival_time.item() == pytest.approx(2.5)
    assert states[1].lateness.item() == pytest.approx(0.5)
    # 回仓库：开始新路径，时间清零，容量恢复
    assert states[2].cur_arrival_time.item() == 0.0
    assert states[2].used_capacity.item() == 0.0
    # 新车从时刻0出发，0.25h 到达客户3，等待至 6h
    assert states[3].cur_arrival_time.item() == pytest.approx(6.0)
    cost, _, lateness, n_late, length = CVRPTW.get_costs(input, pi)
    expected_length = 1 + 1 + math.sqrt(2) + 0.5 + 0.5
    assert length.item() == pytest.approx(expected_length, rel=1e-6)
    assert lateness.item() == pytest.approx(0.5)
    assert n_late.item() == 1
    assert cost.item() == pytest.approx(expected_length + CVRPTW.penalty() * 0.5, rel=1e-6)
    assert CVRPTW.count_vehicles(pi).item() == 2


def test_penalty_lambda():
    old = CVRPTW.PENALTY_LAMBDA
    try:
        CVRPTW.set_penalty_lambda(3)
        assert CVRPTW.penalty() == 24.0
        CVRPTW.set_penalty_lambda(9)
        assert CVRPTW.penalty() == 72.0
    finally:
        CVRPTW.set_penalty_lambda(old)


def test_hard_mask_excludes_late_customer():
    input = make_instance()
    state = CVRPTW.make_state(input, hard=True)
    state = state.update(torch.tensor([1]))  # 在客户1，时刻 2h
    mask = state.get_mask()[0, 0]
    # 客户2：2h + 1/2h = 2.5h > 2h，硬掩码下不可选；软约束下可选
    assert mask[2].item()
    assert not CVRPTW.make_state(input).update(torch.tensor([1])).get_mask()[0, 0, 2].item()
    # 客户3：2h + ~0.56h < 8h，可选；仓库在客户处总是可选
    assert not mask[3].item()
    assert not mask[0].item()


def test_depot_rule():
    input = make_instance()
    state = CVRPTW.make_state(input)
    # 在仓库且仍有客户可选：仓库不可选
    assert state.get_mask()[0, 0, 0].item()
    # 访问完全部客户回到仓库：仓库可选（结束）
    for a in (1, 2, 0, 3, 0):
        state = state.update(torch.tensor([a]))
    assert not state.get_mask()[0, 0, 0].item()
    assert state.all_finished()


def test_capacity_mask():
    input = make_instance()
    input['demand'] = torch.tensor([[0.6, 0.6, 0.3]])
    state = CVRPTW.make_state(input).update(torch.tensor([1]))
    mask = state.get_mask()[0, 0]
    assert mask[2].item()  # 0.6 + 0.6 > 1
    assert not mask[3].item()


def test_legacy_context_keeps_old_behaviour():
    """legacy 模式复刻旧版上下文：不在仓库清零，并与归一化时间窗混比"""
    input = make_instance()
    state = CVRPTW.make_state(input, legacy=True)
    for a in (1, 2, 0):
        state = state.update(torch.tensor([a]))
    assert state.cur_arrival_time.item() > 0  # 旧实现回仓库不清零
    assert state.clock.item() == 0.0  # 掩码与成本用的时钟按算法1清零
