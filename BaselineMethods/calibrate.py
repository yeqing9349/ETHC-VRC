"""
标定元启发式的单实例时限（冒烟阶段运行，结果供正式运行读取）：
对每个规模取测试集前 num_instances 个实例，用 OR-Tools 依次在时限梯度（如 1/2/5/10/30/60 秒）下求解，
取平均混合成本与最长时限结果相差不超过 tolerance（默认 0.1%）的最短时限，再乘 margin（默认 2）作为余量。
OR-Tools、TS、BA 共用标定出的时限（论文第5.1.2节：三者设置相同，时限留有余量）。

用法：python -m BaselineMethods.calibrate --sizes 20 50 100 --out results/smoke/calibration.json
"""
import argparse
import json
import os
import time

import numpy as np

from BaselineMethods import solve as solve_mod


def run(opts):
    result = {'tolerance': opts.tolerance, 'margin': opts.margin, 'limits': opts.limits,
              'num_instances': opts.num_instances, 'penalty_lambda': opts.penalty_lambda, 'sizes': {}}
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    work_dir = os.path.splitext(opts.out)[0] + '_runs'
    for size in opts.sizes:
        dataset = opts.dataset_pattern.format(size=size)
        costs, cpu = [], []
        for limit in opts.limits:
            out = os.path.join(work_dir, 'ortools_n{}_t{:g}'.format(size, limit))
            if os.path.isfile(out + '.json'):
                with open(out + '.json', encoding='utf-8') as f:
                    summary = json.load(f)
            else:
                summary = solve_mod.run(solve_mod.get_options([
                    '--method', 'ortools', '--dataset', dataset, '--out', out,
                    '--penalty_lambda', str(opts.penalty_lambda), '--time_limit', str(limit),
                    '--workers', str(opts.workers), '--num_instances', str(opts.num_instances),
                    '--chunk', str(opts.num_instances)]))
            costs.append(summary['best']['cost'])
            cpu.append(summary['time_s']['per_instance_mean'])
        ref = costs[-1]
        chosen = next(limit for limit, c in zip(opts.limits, costs) if c <= ref * (1 + opts.tolerance))
        result['sizes'][str(size)] = {
            'mean_cost': dict(zip(map(str, opts.limits), costs)),
            'cpu_per_instance_s': dict(zip(map(str, opts.limits), cpu)),
            'converged_limit_s': chosen,
            'time_limit_s': chosen * opts.margin,
        }
        print('n={}: costs {} -> converged at {}s, time limit {}s'.format(
            size, ['{:.4f}'.format(c) for c in costs], chosen, chosen * opts.margin))
    result['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(opts.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1)
    return result


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Calibrate the per-instance time limit of the metaheuristics")
    parser.add_argument('--sizes', type=int, nargs='+', default=[20, 50, 100])
    parser.add_argument('--dataset_pattern', default='data/cvrptw/cvrptw{size}_test_seed1234.pkl')
    parser.add_argument('--limits', type=float, nargs='+', default=[1, 2, 5, 10, 30, 60])
    parser.add_argument('--num_instances', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=0.001)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--penalty_lambda', type=float, default=3.0)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--out', required=True)
    return parser.parse_args(args)


if __name__ == "__main__":
    run(get_options())
