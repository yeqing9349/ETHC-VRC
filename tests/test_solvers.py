"""
元启发式（OR-Tools、TS、BA）的单元测试：解覆盖全部客户且各一次、容量不超、车队不超过 12 辆，
并且用 DRL 方法的成本函数（CVRPTW.get_costs）重算的混合成本与求解器报告的一致——三者与 DRL 方法优化同一目标。
运行：uv run pytest -q
"""
import numpy as np
import pytest
import torch

from BaselineMethods.cvrptw_eval import load_instances, eval_solution, to_sequence, SPEED, WORK_TIME
from problems.vrp.problem_vrp import CVRPTW

DATA = 'data/cvrptw/cvrptw20_test_seed1234.pkl'


def torch_cost(inst, seq, penalty_lambda):
    """把求解器的解转成 DRL 方法的节点序列，用 CVRPTW.get_costs 计算（float32）"""
    CVRPTW.set_penalty_lambda(penalty_lambda)
    input = {
        'depot': torch.tensor(inst.coords[None, 0], dtype=torch.float32),
        'loc': torch.tensor(inst.coords[None, 1:], dtype=torch.float32),
        'demand': torch.tensor(inst.demand[None, 1:] / inst.capacity, dtype=torch.float32),
        'tw': torch.tensor(inst.tw[None, 1:] / WORK_TIME, dtype=torch.float32),
    }
    cost, _, lateness, n_late, length = CVRPTW.get_costs(input, torch.tensor([seq]))
    CVRPTW.set_penalty_lambda(3.0)
    return cost.item(), length.item(), lateness.item()


def check_solution(inst, routes, lens, beta, max_routes=12):
    served = np.concatenate([routes[r, :lens[r]] for r in range(len(lens)) if lens[r] > 0])
    assert sorted(served.tolist()) == list(range(1, inst.n + 1))
    for r in range(len(lens)):
        assert inst.demand[routes[r, :lens[r]]].sum() <= inst.capacity
    assert (lens > 0).sum() <= max_routes
    return eval_solution(routes, lens, inst.dist, inst.tw, SPEED, beta)


@pytest.mark.parametrize('method', ['ortools', 'ts', 'ba'])
@pytest.mark.parametrize('penalty_lambda', [1.0, 3.0, 9.0])
def test_solver_objective_matches_drl_cost(method, penalty_lambda):
    if method == 'ortools':
        from BaselineMethods.ortools_cvrptw import solve
    elif method == 'ts':
        from BaselineMethods.tabu_search import solve
    else:
        from BaselineMethods.bees_algorithm import solve
    beta = penalty_lambda * WORK_TIME
    for inst in load_instances(DATA, 0, 3):
        out = solve(inst, beta, 0.5)
        routes, lens = out[0], out[1]
        cost, length, late, n_late, vehicles = check_solution(inst, routes, lens, beta)
        t_cost, t_length, t_late = torch_cost(inst, to_sequence(routes, lens), penalty_lambda)
        assert t_cost == pytest.approx(cost, rel=1e-5)
        assert t_length == pytest.approx(length, rel=1e-5)


def test_eval_route_waits_and_penalizes():
    """手算：仓库(0,0) -> 客户(1,0)，时间窗[2,4]：0.5h 到达，等待到 2h；再到 (1,1)，时间窗[0,2]：2.5h 到达，迟到 0.5h"""
    from BaselineMethods.cvrptw_eval import Instance, eval_route
    inst = Instance([0.0, 0.0], [[1.0, 0.0], [1.0, 1.0]], [1, 1], 10, [[2.0, 4.0], [0.0, 2.0]])
    cost, length, late, n_late = eval_route(np.array([1, 2]), 2, inst.dist, inst.tw, SPEED, 24.0)
    assert late == pytest.approx(0.5)
    assert n_late == 1
    assert length == pytest.approx(2 + np.sqrt(2))
    assert cost == pytest.approx(length + 24 * 0.5)
