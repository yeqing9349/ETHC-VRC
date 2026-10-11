"""
OR-Tools 求解 CVRPTW（论文第5.1.2节）：引导局部搜索（GLS），车队固定为 max_routes(=12) 辆，
容量为硬约束；时间维度允许等待（早到等到时间窗开启，不罚）；在 x^t 处设软上界，每小时迟到罚 β = λ·T_w，
与 DRL 方法的混合成本相同。距离与时间按 SCALE 放大后取整（OR-Tools 只接受整数），
求解后用同一成本函数（cvrptw_eval.eval_solution）按原始精度重算所报告的混合成本。
"""
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from BaselineMethods.cvrptw_eval import WORK_TIME, SPEED

SCALE = 10000


def solve(inst, beta, time_limit, max_routes=12):
    """
    :return: routes (max_routes, n), lens (max_routes,), status
    """
    n = inst.n
    dist = np.rint(inst.dist * SCALE).astype(np.int64)
    travel = np.rint(inst.dist / SPEED * SCALE).astype(np.int64)  # 小时 × SCALE
    horizon = int(4 * WORK_TIME * SCALE)

    manager = pywrapcp.RoutingIndexManager(n + 1, max_routes, 0)
    routing = pywrapcp.RoutingModel(manager)

    dist_cb = routing.RegisterTransitMatrix(dist.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)

    demand_cb = routing.RegisterUnaryTransitVector(inst.demand.tolist())
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [inst.capacity] * max_routes, True, 'Capacity')

    time_cb = routing.RegisterTransitMatrix(travel.tolist())
    # 松弛量即等待时间；每辆车于时刻0从仓库出发
    routing.AddDimension(time_cb, horizon, horizon, True, 'Time')
    time_dim = routing.GetDimensionOrDie('Time')
    for c in range(1, n + 1):
        index = manager.NodeToIndex(c)
        time_dim.CumulVar(index).SetMin(int(round(inst.tw[c, 0] * SCALE)))  # 早到则等待到时间窗开启
        # 目标函数增加 β × max(0, 开始服务时刻 - x^t)，单位与距离 × SCALE 一致
        time_dim.SetCumulVarSoftUpperBound(index, int(round(inst.tw[c, 1] * SCALE)), int(round(beta)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(round(time_limit * 1000)))

    solution = routing.SolveWithParameters(params)
    routes = np.zeros((max_routes, n), dtype=np.int64)
    lens = np.zeros(max_routes, dtype=np.int64)
    if solution is None:
        return routes, lens, routing.status()
    for v in range(max_routes):
        index = solution.Value(routing.NextVar(routing.Start(v)))
        k = 0
        while not routing.IsEnd(index):
            routes[v, k] = manager.IndexToNode(index)
            k += 1
            index = solution.Value(routing.NextVar(index))
        lens[v] = k
    return routes, lens, routing.status()
