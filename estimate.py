#!/usr/bin/env python
"""
估算正式运行（configs/paper.yaml）的 GPU 与 CPU 时长，依据冒烟阶段（configs/smoke.yaml）测得的：
- 训练计时探针：各模型 × 各训练规模在 batch 512 下的每次迭代耗时、贪心评估耗时与数据生成耗时（FP32 与 TF32）；
- 评估计时探针：G 在 10,000 个实例上的耗时，S 按论文口径的每实例耗时（各测试规模）；
- 元启发式时限标定：各规模的单实例时限。
输出 results/smoke/estimate.md 与 estimate.csv：按阶段 × 方法 × 规模列出 GPU·h、CPU·h，以及本机墙钟时间。

用法：uv run python estimate.py [--paper configs/paper.yaml] [--smoke configs/smoke.yaml]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

from experiments import load_config, build_tasks, eval_out, time_limit, MAIN_LAMBDA

VARIANT_NAMES = {'ethc': 'ETHC-VRC', 'ethc_woet': 'ETHC-VRC w/o ET', 'tpa': 'TPA-Reward', 'mvmoe': 'MVMoE',
                 'ortools': 'OR-Tools', 'ortools_eq': 'OR-Tools（等时）', 'ts': 'TS', 'ba': 'BA'}


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def train_probe(smoke, variant, size, tf32):
    path = os.path.join(smoke.paths.results, 'probes', 'train_{}_n{}{}'.format(variant, size, '_tf32' if tf32 else ''),
                        'probe.json')
    return load_json(path) if os.path.isfile(path) else None


def greedy_probe(smoke, variant, size):
    """贪心评估每实例每模式的秒数；没有该规模的探针时取不小于它的最近规模（偏保守）"""
    modes = {'ethc': ['soft', 'hard'], 'tpa': ['soft'], 'mvmoe': ['hard'], 'ethc_woet': ['soft', 'hard']}[variant]
    probe_variant = 'ethc' if variant == 'ethc_woet' else variant
    probed = sorted(smoke.sizes)
    candidates = [s for s in probed if s >= size] or probed[-1:]
    path = eval_out(smoke, probe_variant, candidates[0], MAIN_LAMBDA, smoke.seeds[0], candidates[0], 'greedy', modes,
                    tag='probe') + '.json'
    if not os.path.isfile(path):
        return None
    s = load_json(path)
    return {m: s['time_s'][m] / s['n'] for m in modes}  # 每实例每模式秒数（批量处理）


def sample_probe(smoke, size, test_size):
    path = eval_out(smoke, 'ethc', size, MAIN_LAMBDA, smoke.seeds[0], test_size, 'sample', ['soft', 'hard'],
                    smoke.eval_probes.samples, tag='probe') + '.json'
    if not os.path.isfile(path):
        return None
    s = load_json(path)
    return {m: s['time_s'][m] / s['n'] for m in ('soft', 'hard')}  # 每实例每模式秒数（1280 个样本）


def estimate(paper, smoke, tf32):
    rows = []  # (stage, method, size, gpu_h, cpu_h, count, note)
    missing = set()
    tasks = build_tasks(paper)
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for t in tasks.values():
        m = t.meta
        if t.stage == 'train':
            p = train_probe(smoke, m['variant'], m['size'], tf32)
            if p is None:
                missing.add('train probe {} n={}{}'.format(m['variant'], m['size'], ' tf32' if tf32 else ''))
                continue
            epochs = paper.train.n_epochs
            epoch_s = (paper.train.epoch_size // paper.train.batch_size) * p['step_time_s'] \
                + (paper.train.epoch_size / p.get('val_size', paper.train.val_size) + 2) * p['greedy_val_time_s'] \
                + p['data_gen_epoch_s']
            key = ('train', m['variant'], m['size'])
            agg[key][0] += epochs * epoch_s / 3600
            agg[key][2] += 1
        elif t.stage in ('eval', 'timing'):
            n = m['num']
            if m['strategy'] == 'greedy':
                p = greedy_probe(smoke, m['variant'], m['test_size'])
                if p is None:
                    missing.add('greedy probe {} n={}'.format(m['variant'], m['test_size']))
                    continue
                secs = sum(p[mode] for mode in m['modes']) * n
            else:
                p = sample_probe(smoke, m['size'], m['test_size'])
                if p is None:
                    missing.add('sample probe train {} test {}'.format(m['size'], m['test_size']))
                    continue
                secs = sum(p[mode] for mode in m['modes']) * n * m['samples'] / smoke.eval_probes.samples
            key = ('eval', '{} ({})'.format(m['variant'], 'G' if m['strategy'] == 'greedy' else 'S'), m['test_size'])
            agg[key][0] += secs / 3600
            agg[key][2] += 1
        elif t.stage == 'equal_time':
            # 等时 OR-Tools：单实例时限 = ETHC-VRC(S) 的每实例耗时（用冒烟阶段的计时估计）
            p = sample_probe(smoke, m['size'], m['size'])
            if p is None:
                missing.add('sample probe train {} test {}'.format(m['size'], m['size']))
                continue
            n = paper.solvers.get('num_instances', paper.data.num_test)
            key = ('solve', m['method'], m['size'])
            agg[key][1] += sum(p.values()) * n / 3600
            agg[key][2] += 1
        elif t.stage == 'solve':
            try:
                limit = time_limit(paper, m['size'])
            except (FileNotFoundError, KeyError):
                missing.add('time-limit calibration n={}'.format(m['size']))
                continue
            n = paper.solvers.get('num_instances', paper.data.num_test)
            key = ('solve', m['method'], m['size'])
            agg[key][1] += limit * n / 3600
            agg[key][2] += 1
    for (stage, method, size), (gpu_h, cpu_h, count) in sorted(agg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        rows.append((stage, method, size, gpu_h, cpu_h, count))
    return rows, missing


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Estimate the GPU and CPU time of the full run")
    parser.add_argument('--paper', default='configs/paper.yaml')
    parser.add_argument('--smoke', default='configs/smoke.yaml')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs of the same speed as the probe GPU')
    parser.add_argument('--price', type=float, default=None, help='Rental price per GPU-hour (yuan), for a cost estimate')
    opts = parser.parse_args()
    paper, smoke = load_config(opts.paper), load_config(opts.smoke)
    workers = paper.solvers.workers
    out_dir = smoke.paths.results
    lines = ['# 正式运行时长估算', '',
             '依据冒烟阶段在本机测得的计时（GPU：{}）。GPU·h 为单卡串行时长；CPU·h 为单核时长，'
             '元启发式按 {} 个进程并行，墙钟时间 = CPU·h / {}。'.format(_probe_device(smoke), workers, workers), '']
    csv_rows = []
    totals = {}
    for tf32 in (False, True):
        rows, missing = estimate(paper, smoke, tf32)
        label = 'TF32' if tf32 else 'FP32'
        gpu_total = sum(r[3] for r in rows)
        cpu_total = sum(r[4] for r in rows)
        totals[label] = (gpu_total, cpu_total)
        lines += ['## 训练用 {}'.format(label), '',
                  '| 阶段 | 方法 | 规模 | 任务数 | GPU·h | CPU·h |', '|---|---|---|---|---|---|']
        for stage, method, size, gpu_h, cpu_h, count in rows:
            lines.append('| {} | {} | {} | {} | {:.1f} | {:.1f} |'.format(stage, VARIANT_NAMES.get(method, method), size,
                                                                       count, gpu_h, cpu_h))
            csv_rows.append([label, stage, method, size, count, round(gpu_h, 3), round(cpu_h, 3)])
        lines += ['| **合计** | | | | **{:.0f}** | **{:.0f}** |'.format(gpu_total, cpu_total), '']
        if missing:
            lines += ['缺少的计时（未计入）：' + '；'.join(sorted(missing)), '']
    gpu_fp32, cpu_total = totals['FP32']
    gpu_tf32 = totals['TF32'][0]
    lines += ['## 墙钟时间（{} 张与测速机器同型号的卡，GPU 与 CPU 任务并行）'.format(opts.gpus), '',
              '- GPU（FP32）：约 {:.0f} 小时（{:.1f} 天）'.format(gpu_fp32 / opts.gpus, gpu_fp32 / opts.gpus / 24),
              '- GPU（TF32）：约 {:.0f} 小时（{:.1f} 天）'.format(gpu_tf32 / opts.gpus, gpu_tf32 / opts.gpus / 24),
              '- CPU（{} 个进程）：约 {:.0f} 小时（{:.1f} 天）'.format(workers, cpu_total / workers, cpu_total / workers / 24),
              '- 总墙钟时间约为两者中较大者。', '']
    if opts.price is not None:
        lines += ['## 租用费用（按 {:.2f} 元/卡·时，GPU 用 FP32 估算）'.format(opts.price), '',
                  '- GPU 机时 {:.0f} 卡·时，约 {:.0f} 元；若因 CPU 求解延长租期，按实际墙钟时间 × 卡数计。'.format(
                      gpu_fp32, gpu_fp32 * opts.price), '']
    lines += [
              '换到算力中心：GPU 墙钟 ≈ GPU·h ÷（卡数 × 该卡相对本机的速度倍数）。训练任务彼此独立，可按卡数并行。']
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'estimate.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'estimate.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['precision', 'stage', 'method', 'size', 'tasks', 'gpu_h', 'cpu_h'])
        w.writerows(csv_rows)
    print('\n'.join(lines))


def _probe_device(smoke):
    probes = os.path.join(smoke.paths.results, 'probes')
    if os.path.isdir(probes):
        for d in sorted(os.listdir(probes)):
            p = os.path.join(probes, d, 'probe.json')
            if os.path.isfile(p):
                return load_json(p).get('device', '?')
    return '?'


if __name__ == "__main__":
    main()
