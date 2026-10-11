"""
用 OR-Tools、TS 或 BA 求解测试集，输出与 eval.py 相同格式的结果（<out>.npz 逐实例、<out>.json 汇总）。
三种方法使用相同的目标（混合成本，β = λ·T_w）、约束（容量硬约束、车队 12 辆）与单实例时限，多进程并行。
车队 12 辆无法构造可行解时自动增加车辆重试，并记录增加的次数（预期为 0）。
中间结果每 chunk 个实例保存一次，中断后重新运行会从断点继续。

用法（在仓库根目录）：
python -m BaselineMethods.solve --method ortools --dataset data/cvrptw/cvrptw20_test_seed1234.pkl \
    --penalty_lambda 3 --time_limit 2 --workers 12 --out results/paper/solvers/ortools_n20_l3
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from BaselineMethods.cvrptw_eval import load_instances, eval_solution, to_sequence, WORK_TIME, SPEED

FIELDS = ('cost', 'length', 'lateness', 'n_late', 'vehicles')


def _solve_one(args):
    method, inst, beta, time_limit, max_routes, seed, method_params = args
    if method == 'ortools':
        from BaselineMethods.ortools_cvrptw import solve
    elif method == 'ts':
        from BaselineMethods.tabu_search import solve
    else:
        assert method == 'ba'
        from BaselineMethods.bees_algorithm import solve
    extra = 0
    t0 = time.perf_counter()
    while True:
        if method == 'ortools':
            routes, lens, status = solve(inst, beta, time_limit, max_routes=max_routes + extra)
            ok, iterations = lens.sum() == inst.n, -1
        else:
            routes, lens, iterations, ok = solve(inst, beta, time_limit, max_routes=max_routes + extra, seed=seed,
                                                 **method_params)
        if ok or extra >= 4:
            break
        extra += 1
    elapsed = time.perf_counter() - t0
    assert ok, "No feasible solution even with {} vehicles".format(max_routes + extra)
    cost, length, late, n_late, vehicles = eval_solution(routes, lens, inst.dist, inst.tw, SPEED, beta)
    return (cost, length, late, n_late, vehicles), extra, iterations, elapsed, to_sequence(routes, lens)


def summarize(metrics):
    cost, length, lateness, n_late, vehicles = metrics.T
    late_customers = n_late.sum()
    return {
        'cost': float(cost.mean()),
        'length': float(length.mean()),
        'lateness_per_instance_h': float(lateness.mean()),
        'feasibility_rate': float((n_late == 0).mean()),
        'late_instance_rate': float((n_late > 0).mean()),
        'avg_lateness_per_late_customer_h': float(lateness.sum() / late_customers) if late_customers > 0 else 0.0,
        'vehicles': float(vehicles.mean()),
    }


def run(opts):
    beta = opts.penalty_lambda * WORK_TIME
    instances = load_instances(opts.dataset, opts.offset, opts.num_instances)
    n_inst = len(instances)
    method_params = json.loads(opts.method_params) if opts.method_params else {}
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    partial_path = opts.out + '.partial.npz'

    metrics = np.full((n_inst, len(FIELDS)), np.nan)
    extra = np.zeros(n_inst, dtype=np.int64)
    iterations = np.zeros(n_inst, dtype=np.int64)
    inst_time = np.zeros(n_inst)
    sequences = np.zeros((n_inst, 2 * instances[0].n + 1), dtype=np.int16)
    wall = 0.0
    done = 0
    if os.path.isfile(partial_path):
        part = np.load(partial_path)
        done = int(part['done'])
        metrics[:done], extra[:done], iterations[:done] = part['metrics'][:done], part['extra'][:done], part['iterations'][:done]
        inst_time[:done], sequences[:done], wall = part['inst_time'][:done], part['sequences'][:done], float(part['wall'])
        print('Resuming {} after {} instances'.format(opts.out, done))

    ctx = mp.get_context('spawn')
    with ctx.Pool(opts.workers) as pool:
        while done < n_inst:
            end = min(done + opts.chunk, n_inst)
            t0 = time.perf_counter()
            results = pool.map(_solve_one, [(opts.method, instances[i], beta, opts.time_limit, opts.max_routes,
                                             opts.seed + i, method_params) for i in range(done, end)], chunksize=1)
            wall += time.perf_counter() - t0
            for i, (m, ex, its, el, seq) in zip(range(done, end), results):
                metrics[i], extra[i], iterations[i], inst_time[i] = m, ex, its, el
                sequences[i, :len(seq)] = seq
            done = end
            np.savez_compressed(partial_path, done=done, metrics=metrics, extra=extra, iterations=iterations,
                                inst_time=inst_time, sequences=sequences, wall=wall)
            print('{} {}: {}/{} instances, mean cost {:.4f}, {:.0f}s'.format(
                opts.method, os.path.basename(opts.out), done, n_inst, np.nanmean(metrics[:done, 0]), wall))

    from experiments import cpu_model  # 只在主进程中导入，求解子进程不需要
    summary = {
        'method': opts.method,
        'dataset': opts.dataset,
        'n': n_inst,
        'offset': opts.offset,
        'penalty_lambda': opts.penalty_lambda,
        'penalty_beta': beta,
        'time_limit_s': opts.time_limit,
        'max_routes': opts.max_routes,
        'method_params': method_params,
        'workers': opts.workers,
        'best': summarize(metrics),
        'extra_vehicle_instances': int((extra > 0).sum()),
        'mean_iterations': float(iterations.mean()),
        'time_s': {'total': wall, 'cpu_sum': float(inst_time.sum()), 'per_instance_mean': float(inst_time.mean())},
        'device': cpu_model(),
    }
    arrays = {'best_{}'.format(f): metrics[:, j] for j, f in enumerate(FIELDS)}
    arrays.update({'extra_vehicles': extra, 'iterations': iterations, 'inst_time': inst_time, 'sequences': sequences})
    np.savez_compressed(opts.out + '.npz', **arrays)
    with open(opts.out + '.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    os.remove(partial_path)
    print('{} on {}: cost {:.4f}, feasible {:.3f}, vehicles {:.2f}, wall {:.0f}s'.format(
        opts.method, os.path.basename(opts.dataset), summary['best']['cost'], summary['best']['feasibility_rate'],
        summary['best']['vehicles'], wall))
    return summary


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Solve a CVRPTW test set with OR-Tools, tabu search or the bees algorithm")
    parser.add_argument('--method', required=True, choices=('ortools', 'ts', 'ba'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True, help='Output path without extension')
    parser.add_argument('--penalty_lambda', type=float, default=3.0, help='beta = lambda * T_w')
    parser.add_argument('--time_limit', type=float, required=True, help='Time limit per instance (s)')
    parser.add_argument('--max_routes', type=int, default=12, help='Fleet size')
    parser.add_argument('--workers', type=int, default=12, help='Parallel processes (one instance each)')
    parser.add_argument('--num_instances', type=int, default=10000)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=240, help='Instances between partial saves')
    parser.add_argument('--seed', type=int, default=1234, help='Base seed of TS/BA (instance i uses seed + i)')
    parser.add_argument('--method_params', default=None, help='JSON dict of TS/BA parameters')
    return parser.parse_args(args)


if __name__ == "__main__":
    run(get_options())
