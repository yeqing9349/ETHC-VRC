#!/usr/bin/env python
"""
估算正式运行（configs/paper.yaml）的 GPU 与 CPU 时长，依据冒烟阶段（configs/smoke.yaml）测得的：
- 训练计时探针：各模型 × 各训练规模在 batch 512 下的每次迭代耗时、贪心评估耗时与数据生成耗时（FP32 与 TF32）；
- 评估计时探针：G 在 10,000 个实例上的耗时，S 按论文口径的每实例耗时（各测试规模）；
- 元启发式时限标定：各规模的单实例时限。
输出 results/<冒烟配置>/estimate.md 与 estimate.csv：按阶段 × 方法 × 规模列出 GPU·h、CPU·h，并按 run_all.py 的调度方式
估算不同卡数下的墙钟时间与租用费用。

用法：uv run python estimate.py [--paper configs/paper.yaml] [--smoke configs/smoke.yaml]
      python estimate.py --paper configs/autodl.yaml --smoke configs/autodl_smoke.yaml --gpus 1,2,4,8 --cpus-per-gpu 25 --price 2.78
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


def task_hours(paper, smoke, t, tf32, missing):
    """单个任务的 (GPU·h, CPU·h)；缺少计时时返回 None 并记入 missing"""
    m = t.meta
    n_solve = paper.solvers.get('num_instances', paper.data.num_test)
    if t.stage == 'train':
        p = train_probe(smoke, m['variant'], m['size'], tf32)
        if p is None:
            missing.add('train probe {} n={}{}'.format(m['variant'], m['size'], ' tf32' if tf32 else ''))
            return None
        epoch_s = (paper.train.epoch_size // paper.train.batch_size) * p['step_time_s'] \
            + (paper.train.epoch_size / p.get('val_size', paper.train.val_size) + 2) * p['greedy_val_time_s'] \
            + p['data_gen_epoch_s']
        return paper.train.n_epochs * epoch_s / 3600, 0.0
    if t.stage in ('eval', 'timing'):
        if m['strategy'] == 'greedy':
            p = greedy_probe(smoke, m['variant'], m['test_size'])
            if p is None:
                missing.add('greedy probe {} n={}'.format(m['variant'], m['test_size']))
                return None
            secs = sum(p[mode] for mode in m['modes']) * m['num']
        else:
            p = sample_probe(smoke, m['size'], m['test_size'])
            if p is None:
                missing.add('sample probe train {} test {}'.format(m['size'], m['test_size']))
                return None
            secs = sum(p[mode] for mode in m['modes']) * m['num'] * m['samples'] / smoke.eval_probes.samples
        return secs / 3600, 0.0
    if t.stage == 'equal_time':
        # 等时 OR-Tools：单实例时限 = ETHC-VRC(S) 的每实例耗时（用冒烟阶段的计时估计）
        p = sample_probe(smoke, m['size'], m['size'])
        if p is None:
            missing.add('sample probe train {} test {}'.format(m['size'], m['size']))
            return None
        return 0.0, sum(p.values()) * n_solve / 3600
    if t.stage == 'solve':
        try:
            limit = time_limit(paper, m['size'])
        except (FileNotFoundError, KeyError):
            missing.add('time-limit calibration n={}'.format(m['size']))
            return None
        return 0.0, limit * n_solve / 3600
    return 0.0, 0.0


def group_key(t):
    m = t.meta
    if t.stage == 'train':
        return 'train', m['variant'], m['size']
    if t.stage in ('eval', 'timing'):
        return 'eval', '{} ({})'.format(m['variant'], 'G' if m['strategy'] == 'greedy' else 'S'), m['test_size']
    return 'solve', m['method'], m['size']


def estimate(paper, smoke, tf32):
    """返回 (按阶段 × 方法 × 规模汇总的行, 逐任务的 (task, GPU·h, CPU·h), 缺少的计时)"""
    missing = set()
    per_task = []
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for t in build_tasks(paper).values():
        h = task_hours(paper, smoke, t, tf32, missing)
        if h is None or h == (0.0, 0.0):
            continue
        per_task.append((t, h[0], h[1]))
        a = agg[group_key(t)]
        a[0] += h[0]
        a[1] += h[1]
        a[2] += 1
    rows = [(stage, method, size, g, c, n) for (stage, method, size), (g, c, n)
            in sorted(agg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2]))]
    return rows, per_task, missing


def gain_at(gain, size):
    """每张卡同时运行多个任务时的吞吐倍数（相对单个任务），按规模线性插值，超出范围取端点"""
    if not gain:
        return 1.0
    xs = sorted(gain)
    if size <= xs[0]:
        return gain[xs[0]]
    for x0, x1 in zip(xs, xs[1:]):
        if size <= x1:
            return gain[x0] + (gain[x1] - gain[x0]) * (size - x0) / (x1 - x0)
    return gain[xs[-1]]


def wall_clock(per_task, gpus, workers, per_gpu=1, gain=None):
    """按 run_all.py 的调度估算墙钟小时数：非独占的 GPU 任务按最长优先分到 gpus × per_gpu 个槽位（每张卡同时运行 per_gpu 个
    任务时，单个任务的耗时变为 per_gpu / 吞吐倍数 倍）；元启发式任务依次运行（每个任务 workers 个进程），与 GPU 任务同时进行；
    独占的计时评估在其他任务结束后单独运行，等时比较在它之后。"""
    durations = []
    for t, g, c in per_task:
        if g > 0 and not t.exclusive:
            size = t.meta['size'] if t.stage == 'train' else t.meta.get('test_size', t.meta.get('size'))
            durations.append(g * per_gpu / gain_at(gain, size) if per_gpu > 1 else g)
    loads = [0.0] * (gpus * per_gpu)
    for h in sorted(durations, reverse=True):
        i = loads.index(min(loads))
        loads[i] += h
    gpu = max(loads)
    cpu = sum(c for t, g, c in per_task if t.stage == 'solve') / workers
    timing = sum(g for t, g, c in per_task if t.exclusive)
    equal = sum(c for t, g, c in per_task if t.stage == 'equal_time') / workers
    return {'gpu': gpu, 'cpu': cpu, 'timing': timing, 'equal_time': equal,
            'longest_task': max(durations, default=0.0), 'total': max(gpu, cpu) + timing + equal}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Estimate the GPU and CPU time of the full run")
    parser.add_argument('--paper', default='configs/paper.yaml')
    parser.add_argument('--smoke', default='configs/smoke.yaml')
    parser.add_argument('--gpus', default='1',
                        help='Comma-separated GPU counts to estimate (GPUs of the same model as the probe GPU), e.g. 1,2,4,8')
    parser.add_argument('--cpus-per-gpu', type=int, default=None,
                        help='CPU cores per rented GPU (AutoDL allocates CPUs per GPU); metaheuristic workers are then '
                             'cores - 2 x GPUs - 2 as with workers: auto. Default: the workers of the paper config')
    parser.add_argument('--price', type=float, default=None, help='Rental price per GPU-hour (yuan), for a cost estimate')
    parser.add_argument('--tasks-per-gpu', type=int, default=1, help='GPU tasks run concurrently on each GPU (run_all.py option)')
    parser.add_argument('--gain', default=None,
                        help='Measured throughput of --tasks-per-gpu concurrent tasks relative to one task, per size, '
                             'e.g. 20:2.0,50:1.6,100:1.3 (interpolated between sizes)')
    opts = parser.parse_args()
    paper, smoke = load_config(opts.paper), load_config(opts.smoke)
    gpu_counts = [int(g) for g in opts.gpus.split(',') if g]
    per_gpu = opts.tasks_per_gpu
    gain = {int(k): float(v) for k, v in (kv.split(':') for kv in opts.gain.split(','))} if opts.gain else None
    out_dir = smoke.paths.results

    def workers_for(n_gpus):
        if opts.cpus_per_gpu is None:
            return paper.solvers.workers
        return max(1, opts.cpus_per_gpu * n_gpus - 2 * n_gpus * per_gpu - 2)

    lines = ['# 正式运行时长估算', '',
             '依据冒烟阶段测得的计时（GPU：{}；元启发式时限：{}）。GPU·h 为单卡串行时长；CPU·h 为单核时长。'.format(
                 _probe_device(smoke), paper.solvers.calibration), '']
    csv_rows = []
    per_task_fp32 = None
    for tf32 in (False, True):
        rows, per_task, missing = estimate(paper, smoke, tf32)
        if not tf32:
            per_task_fp32 = per_task
        label = 'TF32' if tf32 else 'FP32'
        lines += ['## 训练用 {}'.format(label), '',
                  '| 阶段 | 方法 | 规模 | 任务数 | GPU·h | CPU·h |', '|---|---|---|---|---|---|']
        for stage, method, size, gpu_h, cpu_h, count in rows:
            lines.append('| {} | {} | {} | {} | {:.1f} | {:.1f} |'.format(stage, VARIANT_NAMES.get(method, method), size,
                                                                       count, gpu_h, cpu_h))
            csv_rows.append([label, stage, method, size, count, round(gpu_h, 3), round(cpu_h, 3)])
        lines += ['| **合计** | | | | **{:.0f}** | **{:.0f}** |'.format(sum(r[3] for r in rows), sum(r[4] for r in rows)), '']
        if missing:
            lines += ['缺少的计时（未计入）：' + '；'.join(sorted(missing)), '']

    concurrency = '' if per_gpu == 1 else '每张卡同时运行 {} 个 GPU 任务（吞吐倍数 {}），'.format(
        per_gpu, opts.gain or '未给出，按 1 计')
    lines += ['## 墙钟时间与费用（训练用 FP32）', '',
              '{}GPU 任务按最长优先分到各卡；元启发式与 GPU 任务同时运行；计时评估（表2 的时间列）在其他任务结束后独占运行，'
              '等时比较在它之后。最长的单个训练任务约 {:.0f} 小时，卡再多也不会更短。'.format(
                  concurrency, wall_clock(per_task_fp32, 1, 1, per_gpu, gain)['longest_task']), '',
              '| 卡数 | 元启发式进程 | GPU 任务（小时） | 元启发式（小时） | 计时 + 等时（小时） | 总时长 | 费用 |',
              '|---|---|---|---|---|---|---|']
    for n in gpu_counts:
        w = workers_for(n)
        wc = wall_clock(per_task_fp32, n, w, per_gpu, gain)
        cost = '{:.0f} 元'.format(wc['total'] * n * opts.price) if opts.price is not None else '—'
        lines.append('| {} | {} | {:.0f} | {:.0f} | {:.1f} | {:.0f} 小时（{:.1f} 天） | {} |'.format(
            n, w, wc['gpu'], wc['cpu'], wc['timing'] + wc['equal_time'], wc['total'], wc['total'] / 24, cost))
    lines += ['']
    if opts.price is not None:
        lines += ['费用按 {:.2f} 元/卡·时 × 卡数 × 总时长计（整机租用期间都计费，包括 GPU 任务结束后等待元启发式的时间）。'.format(
            opts.price), '']
    lines += ['计时在测速机器上测得；换用其他型号的卡，需在该卡上重新运行冒烟测试再估算。']
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
