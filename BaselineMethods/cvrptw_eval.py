"""
三种元启发式（OR-Tools、TS、BA）共用的实例读取、混合成本评估和邻域操作（numba 加速）。
目标与 DRL 方法相同（论文公式1）：Σ 路径长度 + β·Σ 迟到时长（小时），β = λ·T_w；
早到等待；每条路径一辆车，从仓库于时刻0出发；容量为硬约束；车队上限 max_routes 辆（论文第5.1.2节为12）。
解的表示：routes (max_routes, n) 每行一条路径的客户编号（1..n），lens (max_routes,) 每条路径的客户数。
"""
import pickle

import numpy as np
from numba import njit

WORK_TIME = 8.0  # T_w（小时）
SPEED = 2.0  # S_v（坐标单位/小时）
LATE_EPS = 1e-6  # 与 problems/vrp/problem_vrp.py 中的 LATE_EPS 相同


class Instance(object):
    """原始单位的实例：坐标、整数需求、整数容量、时间窗（小时），下标0为仓库"""

    def __init__(self, depot, loc, demand, capacity, tw):
        self.coords = np.array([depot] + list(loc), dtype=np.float64)
        self.demand = np.array([0] + list(demand), dtype=np.int64)
        self.capacity = int(round(capacity))
        self.tw = np.array([[0.0, np.inf]] + [list(w) for w in tw], dtype=np.float64)
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.sqrt((diff ** 2).sum(-1))
        self.n = len(loc)


def load_instances(path, offset=0, num_instances=None):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    end = None if num_instances is None else offset + num_instances
    return [Instance(depot, loc, demand, capacity, tw) for depot, loc, demand, capacity, tw, worktime in data[offset:end]]


@njit(cache=True)
def eval_route(route, k, dist, tw, speed, beta):
    """一条路径（前 k 个客户）的混合成本、长度、迟到时长、迟到客户数"""
    length = 0.0
    late = 0.0
    n_late = 0
    t = 0.0
    prev = 0
    for i in range(k):
        c = route[i]
        d = dist[prev, c]
        length += d
        t = max(t + d / speed, tw[c, 0])
        if t > tw[c, 1]:
            late += t - tw[c, 1]
            if t - tw[c, 1] > LATE_EPS:
                n_late += 1
        prev = c
    length += dist[prev, 0]
    return length + beta * late, length, late, n_late


@njit(cache=True)
def eval_solution(routes, lens, dist, tw, speed, beta):
    """整个解的 (混合成本, 长度, 迟到时长, 迟到客户数, 车辆数)"""
    cost = 0.0
    length = 0.0
    late = 0.0
    n_late = 0
    vehicles = 0
    for r in range(routes.shape[0]):
        if lens[r] > 0:
            c, l, lt, nl = eval_route(routes[r], lens[r], dist, tw, speed, beta)
            cost += c
            length += l
            late += lt
            n_late += nl
            vehicles += 1
    return cost, length, late, n_late, vehicles


@njit(cache=True)
def route_costs(routes, lens, dist, tw, speed, beta):
    out = np.zeros(routes.shape[0])
    for r in range(routes.shape[0]):
        if lens[r] > 0:
            out[r] = eval_route(routes[r], lens[r], dist, tw, speed, beta)[0]
    return out


@njit(cache=True)
def construct(order, demand, capacity, dist, tw, speed, beta, max_routes):
    """
    按给定顺序逐个以最小增量成本插入客户（容量硬约束，路径数不超过 max_routes）。
    无法插入时（所有路径容量不足且已达上限）返回 ok=False。
    """
    n = order.shape[0]
    routes = np.zeros((max_routes, n), dtype=np.int64)
    lens = np.zeros(max_routes, dtype=np.int64)
    loads = np.zeros(max_routes, dtype=np.int64)
    costs = np.zeros(max_routes)
    buf = np.zeros(n + 1, dtype=np.int64)
    for c in order:
        best_delta = np.inf
        best_r = -1
        best_pos = -1
        for r in range(max_routes):
            if loads[r] + demand[c] > capacity:
                continue
            k = lens[r]
            for pos in range(k + 1):
                for i in range(pos):
                    buf[i] = routes[r, i]
                buf[pos] = c
                for i in range(pos, k):
                    buf[i + 1] = routes[r, i]
                delta = eval_route(buf, k + 1, dist, tw, speed, beta)[0] - costs[r]
                if delta < best_delta:
                    best_delta = delta
                    best_r = r
                    best_pos = pos
            if k == 0:
                break  # 空路径彼此等价，只试第一条
        if best_r < 0:
            return routes, lens, loads, False
        k = lens[best_r]
        for i in range(k, best_pos, -1):
            routes[best_r, i] = routes[best_r, i - 1]
        routes[best_r, best_pos] = c
        lens[best_r] = k + 1
        loads[best_r] += demand[c]
        costs[best_r] = eval_route(routes[best_r], k + 1, dist, tw, speed, beta)[0]
    return routes, lens, loads, True


@njit(cache=True)
def apply_relocate(routes, lens, loads, demand, r1, i, r2, j):
    """把 r1 的第 i 个客户移到 r2 的位置 j（j 为移出后的位置）"""
    c = routes[r1, i]
    for p in range(i, lens[r1] - 1):
        routes[r1, p] = routes[r1, p + 1]
    lens[r1] -= 1
    loads[r1] -= demand[c]
    for p in range(lens[r2], j, -1):
        routes[r2, p] = routes[r2, p - 1]
    routes[r2, j] = c
    lens[r2] += 1
    loads[r2] += demand[c]


@njit(cache=True)
def apply_swap(routes, loads, demand, r1, i, r2, j):
    a = routes[r1, i]
    b = routes[r2, j]
    routes[r1, i] = b
    routes[r2, j] = a
    loads[r1] += demand[b] - demand[a]
    loads[r2] += demand[a] - demand[b]


@njit(cache=True)
def apply_two_opt_star(routes, lens, loads, demand, r1, i, r2, j, buf):
    """交换 r1 第 i 个客户之后与 r2 第 j 个客户之后的尾段"""
    k1 = lens[r1]
    k2 = lens[r2]
    m = 0
    for p in range(i, k1):
        buf[m] = routes[r1, p]
        m += 1
    tail1 = m
    for p in range(j, k2):
        routes[r1, i + p - j] = routes[r2, p]
    for p in range(tail1):
        routes[r2, j + p] = buf[p]
    lens[r1] = i + (k2 - j)
    lens[r2] = j + tail1
    load1 = 0
    for p in range(lens[r1]):
        load1 += demand[routes[r1, p]]
    load2 = 0
    for p in range(lens[r2]):
        load2 += demand[routes[r2, p]]
    loads[r1] = load1
    loads[r2] = load2


@njit(cache=True)
def apply_two_opt(routes, r, i, j):
    """反转 r 中第 i..j 个客户"""
    while i < j:
        tmp = routes[r, i]
        routes[r, i] = routes[r, j]
        routes[r, j] = tmp
        i += 1
        j -= 1


@njit(cache=True)
def random_move(routes, lens, loads, demand, capacity, buf):
    """随机施加一个满足容量的邻域操作（relocate / swap / 2-opt / 2-opt*），用于蜂群算法的邻域搜索"""
    R = routes.shape[0]
    for attempt in range(50):
        kind = np.random.randint(4)
        r1 = np.random.randint(R)
        if lens[r1] == 0:
            continue
        i = np.random.randint(lens[r1])
        r2 = np.random.randint(R)
        if kind == 0:  # relocate
            c = routes[r1, i]
            if r2 != r1 and loads[r2] + demand[c] > capacity:
                continue
            k2 = lens[r2] - (1 if r2 == r1 else 0)
            j = np.random.randint(k2 + 1)
            apply_relocate(routes, lens, loads, demand, r1, i, r2, j)
            return True
        elif kind == 1:  # swap
            if lens[r2] == 0:
                continue
            j = np.random.randint(lens[r2])
            if r1 == r2 and i == j:
                continue
            a = routes[r1, i]
            b = routes[r2, j]
            if r1 != r2 and (loads[r1] - demand[a] + demand[b] > capacity or loads[r2] - demand[b] + demand[a] > capacity):
                continue
            apply_swap(routes, loads, demand, r1, i, r2, j)
            return True
        elif kind == 2:  # 2-opt（路径内反转）
            if lens[r1] < 3:
                continue
            j = np.random.randint(lens[r1])
            if i == j:
                continue
            apply_two_opt(routes, r1, min(i, j), max(i, j))
            return True
        else:  # 2-opt*（路径间交换尾段）
            if r1 == r2:
                continue
            j = np.random.randint(lens[r2] + 1)
            tail1 = 0
            for p in range(i, lens[r1]):
                tail1 += demand[routes[r1, p]]
            tail2 = 0
            for p in range(j, lens[r2]):
                tail2 += demand[routes[r2, p]]
            if loads[r1] - tail1 + tail2 > capacity or loads[r2] - tail2 + tail1 > capacity:
                continue
            apply_two_opt_star(routes, lens, loads, demand, r1, i, r2, j, buf)
            return True
    return False


def to_sequence(routes, lens):
    """转成与 DRL 方法相同格式的节点序列：各路径客户依次排列，路径之间为 0"""
    seq = []
    for r in range(len(lens)):
        if lens[r] > 0:
            seq.extend(int(c) for c in routes[r, :lens[r]])
            seq.append(0)
    return seq
