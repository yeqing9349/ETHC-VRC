#!/usr/bin/env python
"""
一条命令跑完论文的全部实验：

    uv run python run_all.py --config configs/paper.yaml

按 experiments.py 展开的任务及其依赖依次执行：数据检查 → 训练 → 评估 → 元启发式（先标定时限）→ 汇总（表2、图3–6、
paper_numbers.json）。输出文件已存在的任务自动跳过；训练与元启发式求解中断后重新运行会从断点继续。
GPU 任务在 --gpus 列出的每张卡上各跑一个；CPU 任务（元启发式，自带多进程）同一时间只跑一个，与 GPU 任务并行。
计时任务（冒烟阶段的计时探针、表2的计时评估）独占运行，不与其他任务并行，避免 CPU 争用影响计时。

其他用法：
    --dry-run              列出全部任务及其状态，不运行
    --only train,eval      只运行这些阶段（data/probe/calibrate/train/eval/solve/timing/equal_time/analysis）
    --tasks-per-gpu K      每张卡同时运行 K 个 GPU 任务（小模型的训练受 CPU 发起计算的速度限制，单个任务用不满 GPU）
    --task ID              只运行一个任务（供 Slurm 作业数组使用）
    --emit-slurm DIR       为算力中心生成 Slurm 作业脚本（每个阶段一个作业数组，按阶段依赖提交）
"""
import argparse
import os
import re
import subprocess
import sys
import time
from collections import Counter

from experiments import load_config, build_tasks, resolve_cmd

STAGES = ('data', 'probe', 'calibrate', 'train', 'eval', 'solve', 'timing', 'equal_time', 'analysis')


# 调度用的相对耗时（RTX 4060 Ti 上实测的 100 个 epoch 训练小时数，n = 20/50/100）：先派发最长的任务，减少多卡运行末尾的空闲
TRAIN_HOURS = {'ethc': (11.2, 30.1, 84.2), 'ethc_woet': (11.0, 30.6, 89.0), 'tpa': (5.9, 13.5, 32.3),
               'mvmoe': (20.1, 40.3, 91.0)}
SAMPLE_SECONDS = (0.11, 0.31, 0.82)  # ETHC-VRC(S) 每实例耗时（两种模式各 1280 个样本）


def _interp(x, ys, xs=(20, 50, 100)):
    if x <= xs[0]:
        return ys[0] * x / xs[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1] * x / xs[-1]


def expected_hours(task):
    m = task.meta
    if task.stage == 'train':
        return _interp(m['size'], TRAIN_HOURS[m['variant']])
    if task.stage in ('eval', 'timing') and m.get('strategy') == 'sample':
        return m['num'] * m['samples'] / 1280 * len(m['modes']) / 2 * _interp(m['test_size'], SAMPLE_SECONDS) / 3600
    if task.stage in ('solve', 'equal_time'):
        return 1e-3 * m['size']  # 时限随规模增大，规模大的先运行
    return 0.0


def is_done(task):
    if not os.path.isfile(task.marker):
        return False
    dep_markers = task.meta.get('dep_markers')
    if dep_markers:  # 汇总：比全部依赖结果都新才算完成
        mtime = os.path.getmtime(task.marker)
        return all(not os.path.isfile(m) or os.path.getmtime(m) <= mtime for m in dep_markers)
    return True


def log_path(cfg, task):
    return os.path.join(cfg.paths.results, 'logs', re.sub(r'[^A-Za-z0-9_.-]+', '_', task.id) + '.log')


def stamp():
    return time.strftime('%m-%d %H:%M:%S')


def launch(cfg, task, gpu=None):
    os.makedirs(os.path.dirname(log_path(cfg, task)), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # 容器内 os.cpu_count() 是整台机器的核数（AutoDL 上为 208，而 CPU 配额只有 25），PyTorch 等库会按它开线程：
    # 限制每个任务的计算线程数，GPU 任务 2 个（与 workers: auto 为每个 GPU 任务预留的核数一致），元启发式每个进程 1 个
    threads = '2' if task.resource == 'gpu' else '1'
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env.setdefault(var, threads)
    log = open(log_path(cfg, task), 'a', encoding='utf-8')
    log.write('\n[{}] {}\n'.format(stamp(), ' '.join(map(str, resolve_cmd(cfg, task)))))
    log.flush()
    proc = subprocess.Popen(resolve_cmd(cfg, task), stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def print_status(tasks, selected):
    counts = Counter((t.stage, is_done(t)) for t in tasks.values() if t.stage in selected)
    for stage in STAGES:
        done, pending = counts[(stage, True)], counts[(stage, False)]
        if done or pending:
            print('  {:<10} done {:>4}  pending {:>4}'.format(stage, done, pending))


def run(cfg, tasks, selected, gpus):
    todo = {tid: t for tid, t in tasks.items() if t.stage in selected and not is_done(t)}
    # 汇总的依赖中有任务要运行时，汇总也要在其后重新运行
    for tid, t in tasks.items():
        if t.meta.get('dep_markers') and t.stage in selected and any(d in todo for d in t.deps):
            todo[tid] = t
    failed = set()
    running = {}  # task id -> (proc, log, slot, start)
    free_gpus = list(gpus)
    cpu_busy = False
    print('[{}] {} tasks to run'.format(stamp(), len(todo)))
    try:
        while todo or running:
            # 收回已结束的任务
            for tid, (proc, log, slot, start) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[tid]
                if slot == 'cpu':
                    cpu_busy = False
                else:
                    free_gpus.append(slot)
                ok = proc.returncode == 0 and is_done(tasks[tid])
                print('[{}] {} {} ({:.1f} min)'.format(stamp(), 'done  ' if ok else 'FAILED', tid, (time.time() - start) / 60))
                if not ok:
                    failed.add(tid)
                    print('         see {}'.format(log_path(cfg, tasks[tid])))
            # 依赖失败或未完成且不会运行的任务无法开始
            for tid, t in list(todo.items()):
                blocked = [d for d in t.deps if d in failed or (d not in todo and d not in running and not is_done(tasks[d]))]
                if blocked:
                    print('[{}] skip   {} (dependency not available: {})'.format(stamp(), tid, blocked[0]))
                    failed.add(tid)
                    del todo[tid]
            # 启动依赖已满足的任务；有就绪的独占任务时不再启动其他任务，等正在运行的任务结束后单独运行它
            ready = [tid for tid, t in todo.items() if all(is_done(tasks[d]) for d in t.deps)]
            ready.sort(key=lambda tid: -expected_hours(tasks[tid]))  # 最长的任务先运行
            exclusive_ready = [tid for tid in ready if tasks[tid].exclusive]
            if any(tasks[tid].exclusive for tid in running):
                ready = []
            elif exclusive_ready:
                ready = exclusive_ready[:1] if not running else []
            for tid in ready:
                t = tasks[tid]
                if t.exclusive and running:
                    break
                if t.resource == 'gpu':
                    if not free_gpus:
                        continue
                    slot = free_gpus.pop(0)
                else:
                    if cpu_busy:
                        continue
                    slot, cpu_busy = 'cpu', True
                proc, log = launch(cfg, t, gpu=None if slot == 'cpu' else slot)
                running[tid] = (proc, log, slot, time.time())
                del todo[tid]
                print('[{}] start  {}{}{}'.format(stamp(), tid, '' if slot == 'cpu' else ' (GPU {})'.format(slot),
                                                  ' [exclusive]' if t.exclusive else ''))
                if t.exclusive:
                    break
            time.sleep(2)
    except KeyboardInterrupt:
        print('Interrupted, stopping running tasks (they resume on the next run)')
        for proc, log, slot, start in running.values():
            proc.terminate()
            log.close()
        raise
    return failed


def emit_slurm(cfg, tasks, selected, out_dir, config_path):
    """每个阶段一个作业数组；阶段之间按 afterok 依赖提交（元启发式只依赖标定，与 GPU 阶段并行）"""
    os.makedirs(out_dir, exist_ok=True)
    stage_deps = {'probe': [], 'data': [], 'calibrate': [], 'train': ['data'], 'eval': ['train'],
                  'solve': ['calibrate'], 'timing': ['eval', 'solve'], 'equal_time': ['timing'],
                  'analysis': ['eval', 'solve', 'timing', 'equal_time']}
    # 路径都相对仓库根目录：submit.sh 先进入仓库根目录再提交，作业在提交目录（$SLURM_SUBMIT_DIR）中运行
    rel_dir = os.path.relpath(out_dir).replace(os.sep, '/')
    to_root = os.path.relpath('.', out_dir).replace(os.sep, '/')
    submit = ['#!/bin/bash', 'set -e', 'cd "$(dirname "$0")/{}"'.format(to_root)]
    written = []
    for stage in STAGES:
        ids = [tid for tid, t in tasks.items() if t.stage == stage and stage in selected and not is_done(t)]
        if not ids:
            continue
        with open(os.path.join(out_dir, 'tasks_{}.txt'.format(stage)), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(ids) + '\n')
        gpu = tasks[ids[0]].resource == 'gpu'
        lines = ['#!/bin/bash',
                 '#SBATCH --job-name=ethc_{}'.format(stage),
                 '#SBATCH --array=0-{}'.format(len(ids) - 1),
                 '#SBATCH --gres=gpu:1' if gpu else '#SBATCH --cpus-per-task={}'.format(cfg.solvers.workers),
                 '#SBATCH --time=7-00:00:00',
                 '#SBATCH --output={}/slurm_{}_%a.log'.format(rel_dir, stage),
                 'cd "$SLURM_SUBMIT_DIR"',
                 'TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "{}/tasks_{}.txt")'.format(rel_dir, stage),
                 'uv run python run_all.py --config {} --task "$TASK"'.format(config_path.replace(os.sep, '/'))]
        with open(os.path.join(out_dir, '{}.sbatch'.format(stage)), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        dep = ':'.join('${}'.format(d.upper()) for d in stage_deps[stage] if d in written)
        submit.append('{}=$(sbatch --parsable {}{}/{}.sbatch)'.format(
            stage.upper(), '--dependency=afterok:{} '.format(dep) if dep else '', rel_dir, stage))
        written.append(stage)
    with open(os.path.join(out_dir, 'submit.sh'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(submit) + '\n')
    print('Slurm scripts written to {} (adjust partitions and time limits, then run submit.sh)'.format(out_dir))


def main():
    parser = argparse.ArgumentParser(description="Run every experiment of the paper")
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--only', default=None, help='Comma-separated stages to run')
    parser.add_argument('--gpus', default='0',
                        help="Comma-separated GPU ids, one GPU task per entry (repeat an id to run two tasks on it), "
                             "or 'all' for every visible GPU")
    parser.add_argument('--tasks-per-gpu', type=int, default=1,
                        help='GPU tasks run concurrently on each GPU (small models leave the GPU partly idle)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--task', default=None, help='Run a single task id')
    parser.add_argument('--emit-slurm', default=None, metavar='DIR')
    opts = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # nohup 重定向到文件时也逐行写出进度

    if opts.gpus == 'all':
        from experiments import gpu_count
        gpu_ids = [str(i) for i in range(max(1, gpu_count()))]
    else:
        gpu_ids = [g for g in opts.gpus.split(',') if g != '']
    # GPU 槽位：每张卡 tasks-per-gpu 个，按轮次交错排列，使最先派发的长任务分散到不同的卡上
    gpus = [g for _ in range(opts.tasks_per_gpu) for g in gpu_ids]
    cfg = load_config(opts.config, gpu_slots=len(gpus))
    tasks = build_tasks(cfg)
    selected = set(opts.only.split(',')) if opts.only else set(STAGES)
    print('Config {}: {} tasks'.format(opts.config, len(tasks)))
    print_status(tasks, selected)

    if opts.task is not None:
        t = tasks[opts.task]
        if is_done(t):
            print('Already done:', t.id)
            return
        sys.exit(subprocess.call(resolve_cmd(cfg, t)))
    if opts.dry_run:
        for t in tasks.values():
            if t.stage in selected and not is_done(t):
                print('  [{}] {}'.format(t.stage, t.id))
        return
    if opts.emit_slurm:
        emit_slurm(cfg, tasks, selected, opts.emit_slurm, opts.config)
        return

    print('GPU slots: {}; metaheuristic workers: {}'.format(','.join(gpus), cfg.solvers.workers))
    failed = run(cfg, tasks, selected, gpus)
    print_status(tasks, selected)
    if failed:
        print('{} tasks failed or were skipped: {}'.format(len(failed), ', '.join(sorted(failed))))
        sys.exit(1)


if __name__ == "__main__":
    main()
