"""
蜂群算法（Bees Algorithm, BA），作为基于种群的元启发式的代表（论文第5.1.2节，Hussein et al., 2017 综述中的基本 BA）：
- 侦察蜂：ns 个解，按随机顺序（scout='random'）或"时间窗开启时刻 + 随机扰动"的顺序（scout='sorted'）
  逐个以最小增量成本插入客户构造，另含一个按时间窗开启时刻排序构造的解；
- 选择 m 个最优位置，其中 e 个精英位置各派 nep 只跟随蜂、其余各派 nsp 只，跟随蜂在位置附近做邻域搜索：
  复制该解后施加 1..ngh_s 次随机邻域操作（relocate / swap / 2-opt / 2-opt*），取最好的跟随蜂替换位置（若更优）；
- 邻域收缩：位置没有改进时其邻域 ngh_s 减 1（最小为 1）；连续 stlim 次没有改进则放弃该位置（site abandonment），
  换成新的侦察蜂；其余 ns - m 只蜂每轮重新随机侦察；
- 停止：达到单实例时限（与 OR-Tools、TS 相同）。
目标与约束与 OR-Tools 相同：混合成本（β = λ·T_w），容量硬约束，车队不超过 max_routes 辆。
"""
import time

import numpy as np
from numba import njit

from BaselineMethods.cvrptw_eval import eval_solution, construct, random_move, SPEED


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _scout(pop_routes, pop_lens, pop_loads, pop_costs, s, demand, capacity, dist, tw, speed, beta, max_routes, sorted_scout):
    """随机侦察：随机顺序，或按"时间窗开启时刻 + [0, 2) 小时的随机扰动"排序，再逐个以最小增量成本插入"""
    n = demand.shape[0] - 1
    if sorted_scout:
        order = np.argsort(tw[1:, 0] + 2.0 * np.random.random(n)) + 1
    else:
        order = np.random.permutation(n) + 1
    routes, lens, loads, ok = construct(order, demand, capacity, dist, tw, speed, beta, max_routes)
    if not ok:
        return False
    pop_routes[s] = routes
    pop_lens[s] = lens
    pop_loads[s] = loads
    pop_costs[s] = eval_solution(routes, lens, dist, tw, speed, beta)[0]
    return True


@njit(cache=True)
def _ba_iterations(pop_routes, pop_lens, pop_loads, pop_costs, stagnation, site_ngh, best_routes, best_lens, best_cost, n_iter,
                   m, e, nep, nsp, ngh, stlim, demand, capacity, dist, tw, speed, beta, buf, sorted_scout):
    ns, R, n = pop_routes.shape
    cand_routes = np.zeros((R, n), dtype=np.int64)
    cand_lens = np.zeros(R, dtype=np.int64)
    cand_loads = np.zeros(R, dtype=np.int64)
    loc_routes = np.zeros((R, n), dtype=np.int64)
    loc_lens = np.zeros(R, dtype=np.int64)
    loc_loads = np.zeros(R, dtype=np.int64)
    for _ in range(n_iter):
        order = np.argsort(pop_costs)
        # 选择的位置：邻域搜索
        for rank in range(m):
            s = order[rank]
            recruits = nep if rank < e else nsp
            loc_cost = np.inf
            for _r in range(recruits):
                cand_routes[:, :] = pop_routes[s]
                cand_lens[:] = pop_lens[s]
                cand_loads[:] = pop_loads[s]
                moves = np.random.randint(1, site_ngh[s] + 1)
                for _k in range(moves):
                    random_move(cand_routes, cand_lens, cand_loads, demand, capacity, buf)
                c = eval_solution(cand_routes, cand_lens, dist, tw, speed, beta)[0]
                if c < loc_cost:
                    loc_cost = c
                    loc_routes[:, :] = cand_routes
                    loc_lens[:] = cand_lens
                    loc_loads[:] = cand_loads
            if loc_cost < pop_costs[s] - 1e-9:
                pop_routes[s] = loc_routes
                pop_lens[s] = loc_lens
                pop_loads[s] = loc_loads
                pop_costs[s] = loc_cost
                stagnation[s] = 0
            else:
                stagnation[s] += 1
                site_ngh[s] = max(1, site_ngh[s] - 1)
            if pop_costs[s] < best_cost - 1e-9:
                best_cost = pop_costs[s]
                best_routes[:, :] = pop_routes[s]
                best_lens[:] = pop_lens[s]
            if stagnation[s] > stlim:  # 放弃该位置，换成新的侦察蜂
                _scout(pop_routes, pop_lens, pop_loads, pop_costs, s, demand, capacity, dist, tw, speed, beta, R,
                       sorted_scout)
                stagnation[s] = 0
                site_ngh[s] = ngh
        # 其余蜂随机侦察
        for rank in range(m, ns):
            s = order[rank]
            _scout(pop_routes, pop_lens, pop_loads, pop_costs, s, demand, capacity, dist, tw, speed, beta, R,
                   sorted_scout)
            stagnation[s] = 0
            site_ngh[s] = ngh
            if pop_costs[s] < best_cost - 1e-9:
                best_cost = pop_costs[s]
                best_routes[:, :] = pop_routes[s]
                best_lens[:] = pop_lens[s]
    return best_cost


def solve(inst, beta, time_limit, max_routes=12, seed=1234, ns=30, m=10, e=3, nep=15, nsp=5, ngh=4, stlim=500, chunk=5,
          scout='random'):
    """
    :return: routes, lens（最优解），iterations，ok（初始解能否在 max_routes 辆车内构造）
    """
    _seed(seed)
    n = inst.n
    pop_routes = np.zeros((ns, max_routes, n), dtype=np.int64)
    pop_lens = np.zeros((ns, max_routes), dtype=np.int64)
    pop_loads = np.zeros((ns, max_routes), dtype=np.int64)
    pop_costs = np.full(ns, np.inf)
    stagnation = np.zeros(ns, dtype=np.int64)
    site_ngh = np.full(ns, ngh, dtype=np.int64)

    # 第一只侦察蜂按时间窗开启时刻排序构造，其余随机构造
    order = np.argsort(inst.tw[1:, 0] + 1e-3 * inst.dist[0, 1:], kind='stable') + 1
    routes, lens, loads, ok = construct(order, inst.demand, inst.capacity, inst.dist, inst.tw, SPEED, beta, max_routes)
    if not ok:
        return routes, lens, 0, False
    pop_routes[0], pop_lens[0], pop_loads[0] = routes, lens, loads
    pop_costs[0] = eval_solution(routes, lens, inst.dist, inst.tw, SPEED, beta)[0]
    for s in range(1, ns):
        if not _scout(pop_routes, pop_lens, pop_loads, pop_costs, s, inst.demand, inst.capacity, inst.dist, inst.tw,
                      SPEED, beta, max_routes, scout == 'sorted'):
            pop_routes[s], pop_lens[s], pop_loads[s], pop_costs[s] = routes, lens, loads, pop_costs[0]
    b = int(np.argmin(pop_costs))
    best_routes, best_lens, best_cost = pop_routes[b].copy(), pop_lens[b].copy(), pop_costs[b]
    buf = np.zeros(n + 1, dtype=np.int64)

    start = time.perf_counter()
    it = 0
    while time.perf_counter() - start < time_limit:
        best_cost = _ba_iterations(pop_routes, pop_lens, pop_loads, pop_costs, stagnation, site_ngh, best_routes, best_lens,
                                   best_cost, chunk, m, e, nep, nsp, ngh, stlim, inst.demand, inst.capacity,
                                   inst.dist, inst.tw, SPEED, beta, buf, scout == 'sorted')
        it += chunk
    return best_routes, best_lens, it, True
