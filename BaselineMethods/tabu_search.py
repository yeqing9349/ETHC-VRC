"""
禁忌搜索（Tabu Search, TS），作为迭代邻域搜索类元启发式的代表（论文第5.1.2节，Aminzadegan et al., 2021）：
- 初始解：按时间窗开启时刻排序，逐个以最小增量成本插入（cvrptw_eval.construct）；
- 邻域：relocate、swap、路径内 2-opt、路径间 2-opt*；每次迭代评估全部满足容量的邻域解，取非禁忌的最优者；
- 禁忌：客户被移出某条路径后，在 tenure 次迭代内不得移回该路径；被移动过的客户在 tenure 次迭代内也不得参与
  路径内的操作（防止在局部最优处来回循环）；特赦：能得到历史最优解的操作不受禁忌限制；
- 停止：达到单实例时限（与 OR-Tools、BA 相同）。
目标与约束与 OR-Tools 相同：混合成本（β = λ·T_w），容量硬约束，车队不超过 max_routes 辆。
"""
import time

import numpy as np
from numba import njit

from BaselineMethods.cvrptw_eval import eval_route, eval_solution, route_costs, construct, \
    apply_relocate, apply_swap, apply_two_opt, apply_two_opt_star, SPEED

IMPROVE_EPS = 1e-9


@njit(cache=True)
def _tabu_iterations(routes, lens, loads, costs, cur_cost, best_routes, best_lens, best_cost, tabu, tabu_cust, it0, n_iter,
                     demand, capacity, dist, tw, speed, beta, tenure_min, tenure_max, buf, buf2):
    R = routes.shape[0]
    it = it0
    for it in range(it0, it0 + n_iter):
        best_delta = np.inf
        mv_type = -1
        mr1 = mi = mr2 = mj = -1
        # ---- relocate ----
        for r1 in range(R):
            k1 = lens[r1]
            for i in range(k1):
                c = routes[r1, i]
                m = 0
                for p in range(k1):
                    if p != i:
                        buf[m] = routes[r1, p]
                        m += 1
                rem_cost = eval_route(buf, m, dist, tw, speed, beta)[0] if m > 0 else 0.0
                empty_tried = False
                for r2 in range(R):
                    k2 = lens[r2]
                    if r2 == r1:
                        is_tabu = tabu_cust[c] > it
                        for j in range(k1):
                            if j == i:
                                continue
                            for p in range(j):
                                buf2[p] = buf[p]
                            buf2[j] = c
                            for p in range(j, m):
                                buf2[p + 1] = buf[p]
                            delta = eval_route(buf2, k1, dist, tw, speed, beta)[0] - costs[r1]
                            if is_tabu and cur_cost + delta >= best_cost - IMPROVE_EPS:
                                continue
                            if delta < best_delta - IMPROVE_EPS:
                                best_delta, mv_type, mr1, mi, mr2, mj = delta, 0, r1, i, r2, j
                        continue
                    if k2 == 0:
                        if empty_tried:
                            continue
                        empty_tried = True
                    if loads[r2] + demand[c] > capacity:
                        continue
                    is_tabu = tabu[c, r2] > it
                    for j in range(k2 + 1):
                        for p in range(j):
                            buf2[p] = routes[r2, p]
                        buf2[j] = c
                        for p in range(j, k2):
                            buf2[p + 1] = routes[r2, p]
                        delta = rem_cost - costs[r1] + eval_route(buf2, k2 + 1, dist, tw, speed, beta)[0] - costs[r2]
                        if is_tabu and cur_cost + delta >= best_cost - IMPROVE_EPS:
                            continue
                        if delta < best_delta - IMPROVE_EPS:
                            best_delta, mv_type, mr1, mi, mr2, mj = delta, 0, r1, i, r2, j
        # ---- swap ----
        for r1 in range(R):
            for i in range(lens[r1]):
                a = routes[r1, i]
                for r2 in range(r1, R):
                    for j in range(lens[r2]):
                        if r2 == r1 and j <= i:
                            continue
                        b = routes[r2, j]
                        if r1 == r2:
                            for p in range(lens[r1]):
                                buf[p] = routes[r1, p]
                            buf[i] = b
                            buf[j] = a
                            delta = eval_route(buf, lens[r1], dist, tw, speed, beta)[0] - costs[r1]
                            if (tabu_cust[a] > it or tabu_cust[b] > it) and cur_cost + delta >= best_cost - IMPROVE_EPS:
                                continue
                        else:
                            if loads[r1] - demand[a] + demand[b] > capacity or loads[r2] - demand[b] + demand[a] > capacity:
                                continue
                            is_tabu = tabu[a, r2] > it or tabu[b, r1] > it
                            for p in range(lens[r1]):
                                buf[p] = routes[r1, p]
                            buf[i] = b
                            for p in range(lens[r2]):
                                buf2[p] = routes[r2, p]
                            buf2[j] = a
                            delta = eval_route(buf, lens[r1], dist, tw, speed, beta)[0] - costs[r1] \
                                + eval_route(buf2, lens[r2], dist, tw, speed, beta)[0] - costs[r2]
                            if is_tabu and cur_cost + delta >= best_cost - IMPROVE_EPS:
                                continue
                        if delta < best_delta - IMPROVE_EPS:
                            best_delta, mv_type, mr1, mi, mr2, mj = delta, 1, r1, i, r2, j
        # ---- 2-opt（路径内反转 i..j）----
        for r in range(R):
            k = lens[r]
            for i in range(k - 1):
                for j in range(i + 1, k):
                    for p in range(k):
                        buf[p] = routes[r, p]
                    lo = i
                    hi = j
                    while lo < hi:
                        tmp = buf[lo]
                        buf[lo] = buf[hi]
                        buf[hi] = tmp
                        lo += 1
                        hi -= 1
                    delta = eval_route(buf, k, dist, tw, speed, beta)[0] - costs[r]
                    is_tabu = tabu_cust[routes[r, i]] > it or tabu_cust[routes[r, j]] > it
                    if is_tabu and cur_cost + delta >= best_cost - IMPROVE_EPS:
                        continue
                    if delta < best_delta - IMPROVE_EPS:
                        best_delta, mv_type, mr1, mi, mr2, mj = delta, 2, r, i, r, j
        # ---- 2-opt*（交换 r1 第 i 个之后与 r2 第 j 个之后的尾段）----
        for r1 in range(R):
            for r2 in range(r1 + 1, R):
                k1 = lens[r1]
                k2 = lens[r2]
                if k1 == 0 and k2 == 0:
                    continue
                for i in range(k1 + 1):
                    tail1 = 0
                    for p in range(i, k1):
                        tail1 += demand[routes[r1, p]]
                    for j in range(k2 + 1):
                        if (i == k1 and j == k2) or (i == 0 and j == 0):
                            continue
                        tail2 = 0
                        for p in range(j, k2):
                            tail2 += demand[routes[r2, p]]
                        if loads[r1] - tail1 + tail2 > capacity or loads[r2] - tail2 + tail1 > capacity:
                            continue
                        is_tabu = False
                        for p in range(i, k1):
                            if tabu[routes[r1, p], r2] > it:
                                is_tabu = True
                        for p in range(j, k2):
                            if tabu[routes[r2, p], r1] > it:
                                is_tabu = True
                        m1 = 0
                        for p in range(i):
                            buf[m1] = routes[r1, p]
                            m1 += 1
                        for p in range(j, k2):
                            buf[m1] = routes[r2, p]
                            m1 += 1
                        m2 = 0
                        for p in range(j):
                            buf2[m2] = routes[r2, p]
                            m2 += 1
                        for p in range(i, k1):
                            buf2[m2] = routes[r1, p]
                            m2 += 1
                        new1 = eval_route(buf, m1, dist, tw, speed, beta)[0] if m1 > 0 else 0.0
                        new2 = eval_route(buf2, m2, dist, tw, speed, beta)[0] if m2 > 0 else 0.0
                        delta = new1 + new2 - costs[r1] - costs[r2]
                        if is_tabu and cur_cost + delta >= best_cost - IMPROVE_EPS:
                            continue
                        if delta < best_delta - IMPROVE_EPS:
                            best_delta, mv_type, mr1, mi, mr2, mj = delta, 3, r1, i, r2, j
        if mv_type < 0:
            break

        # ---- 执行所选操作，并把移出原路径的客户设为禁忌 ----
        tenure = np.random.randint(tenure_min, tenure_max + 1)
        if mv_type == 0:
            c = routes[mr1, mi]
            apply_relocate(routes, lens, loads, demand, mr1, mi, mr2, mj)
            tabu_cust[c] = it + tenure
            if mr1 != mr2:
                tabu[c, mr1] = it + tenure
        elif mv_type == 1:
            a = routes[mr1, mi]
            b = routes[mr2, mj]
            apply_swap(routes, loads, demand, mr1, mi, mr2, mj)
            tabu_cust[a] = it + tenure
            tabu_cust[b] = it + tenure
            if mr1 != mr2:
                tabu[a, mr1] = it + tenure
                tabu[b, mr2] = it + tenure
        elif mv_type == 2:
            tabu_cust[routes[mr1, mi]] = it + tenure
            tabu_cust[routes[mr1, mj]] = it + tenure
            apply_two_opt(routes, mr1, mi, mj)
        else:
            for p in range(mi, lens[mr1]):
                tabu[routes[mr1, p], mr1] = it + tenure
            for p in range(mj, lens[mr2]):
                tabu[routes[mr2, p], mr2] = it + tenure
            apply_two_opt_star(routes, lens, loads, demand, mr1, mi, mr2, mj, buf)
        costs[mr1] = eval_route(routes[mr1], lens[mr1], dist, tw, speed, beta)[0] if lens[mr1] > 0 else 0.0
        costs[mr2] = eval_route(routes[mr2], lens[mr2], dist, tw, speed, beta)[0] if lens[mr2] > 0 else 0.0
        cur_cost = costs.sum()
        if cur_cost < best_cost - IMPROVE_EPS:
            best_cost = cur_cost
            best_routes[:, :] = routes
            best_lens[:] = lens
    return cur_cost, best_cost, it + 1


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


def solve(inst, beta, time_limit, max_routes=12, seed=1234, tenure_min=7, tenure_max=15, chunk=20):
    """
    :return: routes, lens（最优解），iterations，ok（初始解能否在 max_routes 辆车内构造）
    """
    _seed(seed)
    order = np.argsort(inst.tw[1:, 0] + 1e-3 * inst.dist[0, 1:], kind='stable') + 1
    routes, lens, loads, ok = construct(order, inst.demand, inst.capacity, inst.dist, inst.tw, SPEED, beta, max_routes)
    if not ok:
        return routes, lens, 0, False
    costs = route_costs(routes, lens, inst.dist, inst.tw, SPEED, beta)
    cur_cost = best_cost = costs.sum()
    best_routes, best_lens = routes.copy(), lens.copy()
    tabu = np.zeros((inst.n + 1, max_routes), dtype=np.int64)
    tabu_cust = np.zeros(inst.n + 1, dtype=np.int64)
    buf = np.zeros(inst.n + 1, dtype=np.int64)
    buf2 = np.zeros(inst.n + 1, dtype=np.int64)
    start = time.perf_counter()
    it = 0
    while time.perf_counter() - start < time_limit:
        cur_cost, best_cost, it_new = _tabu_iterations(
            routes, lens, loads, costs, cur_cost, best_routes, best_lens, best_cost, tabu, tabu_cust, it, chunk,
            inst.demand, inst.capacity, inst.dist, inst.tw, SPEED, beta, tenure_min, tenure_max, buf, buf2)
        if it_new < it + chunk:  # 没有可行的邻域操作
            it = it_new
            break
        it = it_new
    return best_routes, best_lens, it, True


def solution_metrics(inst, routes, lens, beta):
    return eval_solution(routes, lens, inst.dist, inst.tw, SPEED, beta)
