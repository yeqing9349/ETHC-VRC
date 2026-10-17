"""
某个规模的表2结果一出来，就看本方法是否领先：ETHC-VRC(G)、ETHC-VRC(S) 相对各基线的逐实例配对比较，口径同
analysis/report.py 的表2（均值差相对基线平均成本的百分比、配对 t 的 95% 置信区间、双侧 Wilcoxon 检验）。
结果可以来自几台机器：--results 依次给出各台的结果目录（results/<配置名> 或拷来的副本），同一结果取第一个找到的。

用法：python -m analysis.compare --config configs/autodl.yaml --size 50 [--results DIR ...]
"""
import argparse
import json
import os
import sys

import numpy as np

from analysis.report import paired
from experiments import load_config, eval_out, solve_out, baseline_samples, MAIN_LAMBDA

# ETHC-VRC(G) 与贪心基线、ETHC-VRC(S) 与同预算采样的基线比较，两者都与元启发式比较
COMPARISONS = [('ETHC-VRC(G)', b) for b in ('TPA-Reward(G)', 'MVMoE(G)', 'OR-Tools', 'TS', 'BA')] + \
              [('ETHC-VRC(S)', b) for b in ('TPA-Reward(S)', 'MVMoE(S)', 'OR-Tools', 'TS', 'BA')]


def result_paths(cfg, size):
    """表2各行在结果目录（cfg.paths.results）下的文件（不含扩展名）"""
    seed, k = cfg.seeds[0], baseline_samples(cfg)

    def ev(variant, strategy, modes, samples=None):
        if strategy == 'sample' and samples is None:
            samples = cfg.eval.samples
        return eval_out(cfg, variant, size, MAIN_LAMBDA, seed, size, strategy, modes, samples)

    return {'ETHC-VRC(G)': ev('ethc', 'greedy', ['soft', 'hard']), 'ETHC-VRC(S)': ev('ethc', 'sample', ['soft', 'hard']),
            'TPA-Reward(G)': ev('tpa', 'greedy', ['soft']), 'TPA-Reward(S)': ev('tpa', 'sample', ['soft'], k),
            'MVMoE(G)': ev('mvmoe', 'greedy', ['hard']), 'MVMoE(S)': ev('mvmoe', 'sample', ['hard'], k),
            'OR-Tools': solve_out(cfg, 'ortools', size, MAIN_LAMBDA), 'TS': solve_out(cfg, 'ts', size, MAIN_LAMBDA),
            'BA': solve_out(cfg, 'ba', size, MAIN_LAMBDA)}


def load(cfg, size, dirs):
    """{行: (汇总, 逐实例成本, 来源目录)}；汇总文件（任务完成的标记，最后写入）存在的结果才算数"""
    found = {}
    for d in dirs:
        cfg.paths.results = d
        for row, base in result_paths(cfg, size).items():
            if row not in found and os.path.isfile(base + '.json'):
                with open(base + '.json', encoding='utf-8') as f:
                    summary = json.load(f)
                found[row] = (summary, np.load(base + '.npz')['best_cost'], d)
    return found


def verdict(c):
    if c['p_wilcoxon'] < 0.05 and c['ci_high_pct'] < 0:
        return 'lower'
    if c['p_wilcoxon'] < 0.05 and c['ci_low_pct'] > 0:
        return 'higher'
    return 'no significant difference'


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare ETHC-VRC with the Table 2 baselines at one problem size")
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--results', nargs='+', default=None,
                        help='Result directories to search in order (default: the results directory of the config)')
    opts = parser.parse_args(argv)
    cfg = load_config(opts.config)
    dirs = opts.results or [cfg.paths.results]
    found = load(cfg, opts.size, dirs)
    print('n={}: Table 2 results found in {} director{}'.format(opts.size, len(dirs), 'y' if len(dirs) == 1 else 'ies'))
    print('{:<15} {:>10} {:>9} {:>10}  {}'.format('Method', 'Cost', 'Feasible', 'Instances', 'Source'))
    for row in result_paths(cfg, opts.size):
        if row in found:
            summary, cost, d = found[row]
            print('{:<15} {:>10.4f} {:>8.2f}% {:>10}  {}'.format(
                row, float(np.mean(cost)), 100 * summary['best']['feasibility_rate'], len(cost), d))
        else:
            print('{:<15} {:>10}'.format(row, 'missing'))
    print('\nPaired comparison (difference relative to the second method; negative: ETHC-VRC is lower):')
    for a, b in COMPARISONS:
        if a in found and b in found:
            c = paired(found[a][1], found[b][1])
            print('{:<13} vs {:<15} {:+7.2f}% [{:+.2f}, {:+.2f}]  Wilcoxon p={:.1e}  n={}  {}'.format(
                a, b, c['rel_mean_pct'], c['ci_low_pct'], c['ci_high_pct'], c['p_wilcoxon'], c['n'], verdict(c)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
