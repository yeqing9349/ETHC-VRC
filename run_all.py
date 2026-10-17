#!/usr/bin/env python
"""
一条命令跑完论文的全部实验：

    uv run python run_all.py --config configs/paper.yaml

按 experiments.py 展开的任务及其依赖依次执行：数据检查 → 训练 → 评估 → 元启发式（先标定时限）→ 汇总（表2、图3–6、
paper_numbers.json）。输出文件已存在的任务自动跳过；中断后重新运行同一条命令即从断点继续（训练从上一个 epoch 的检查点，
元启发式从最近一次保存的进度，评估从头重做）。GPU 任务按 --gpus 与 --tasks-per-gpu 分到各张卡；CPU 任务（元启发式，
自带多进程）同一时间只跑一个，与 GPU 任务并行。计时任务（冒烟阶段的计时探针、表2的计时评估）独占运行，不与其他任务并行，
避免 CPU 争用影响计时。派发按配置 schedule 段的优先级（数小的先，同一级里预计时长长的先），schedule.solo 匹配的任务
单独占一张卡。

保护：
- 同一配置同时只允许一个 run_all.py 运行（锁文件 results/<配置名>/run_all.lock，由编排进程和它启动的任务共同持有）：
  上一次启动的编排进程或任务还在运行时，新的启动会被拒绝，避免同一任务被运行两次、写坏同一批文件。
- 配置的 hardware 段（configs/autodl.yaml）给出表2计时所需的显卡与元启发式所需的 CPU：本机不符时，这些任务及依赖它们的
  任务暂不运行，其余任务照常运行。用 --gpus all 而本机看不到显卡时（例如 AutoDL 的无卡模式），不启动任何任务。

其他用法：
    --dry-run              列出全部任务及其状态，不运行
    --only train,eval      只运行这些阶段（data/probe/calibrate/train/eval/solve/timing/equal_time/analysis）
    --match REGEX          只运行 id 匹配该正则表达式的任务（例如先跑一个规模做试点）
    --tasks-per-gpu K      每张卡同时运行 K 个 GPU 任务；也可按规模给出，如 20:3,50:3,100:2（规模 ≤20 与 ≤50 的任务每卡
                           3 个，更大的 2 个；评估按训练规模与测试规模中较大的一个）。小模型的训练受 CPU 发起计算的速度限制，
                           单个任务用不满 GPU
    --task ID              只运行一个任务（供 Slurm 作业数组使用）
    --emit-slurm DIR       为算力中心生成 Slurm 作业脚本（每个阶段一个作业数组，按阶段依赖提交）

退出码：0 全部完成；1 有任务失败；2 已有运行在进行，未启动；3 其余任务已完成，有任务因硬件不符暂未运行；
4 本机没有可运行的任务（硬件不符或看不到显卡）；130 被 kill 或 Ctrl-C 中断。scripts/autodl_run.sh 据此决定是否关机。
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from math import lcm

from atomic_io import save_json
from experiments import (load_config, build_tasks, resolve_cmd, parse_tasks_per_gpu, tasks_per_gpu_for,
                         max_tasks_per_gpu, gpu_task_size, hardware_requirement, hardware_matches)

STAGES = ('data', 'probe', 'calibrate', 'train', 'eval', 'solve', 'timing', 'equal_time', 'analysis')
POLL_SECONDS = 2  # 调度循环的间隔


# 调度用的相对耗时（RTX 4060 Ti 上实测的 100 个 epoch 训练小时数，n = 20/50/100）：同一优先级里先派发最长的任务，
# 减少多卡运行末尾的空闲
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


def launch(cfg, task, gpu=None, lock_fd=None):
    os.makedirs(os.path.dirname(log_path(cfg, task)), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # 容器内 os.cpu_count() 是整台机器的核数（AutoDL 上为 208，而 CPU 配额只有 25），PyTorch 等库会按它开线程：
    # 限制每个任务的计算线程数，GPU 任务 2 个（workers: auto 为每个 GPU 任务预留一个物理核），元启发式每个进程 1 个
    threads = '2' if task.resource == 'gpu' else '1'
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env.setdefault(var, threads)
    kwargs = {}
    if os.name == 'posix':
        # 每个任务单独一个进程组，中断时连同元启发式的求解子进程一起结束；任务继承锁文件，
        # 编排进程意外退出而任务仍在运行时，新的 run_all.py 也无法启动
        kwargs['start_new_session'] = True
        if lock_fd is not None:
            kwargs['pass_fds'] = (lock_fd,)
    log = open(log_path(cfg, task), 'a', encoding='utf-8')
    log.write('\n[{}] {}\n'.format(stamp(), ' '.join(map(str, resolve_cmd(cfg, task)))))
    log.flush()
    proc = subprocess.Popen(resolve_cmd(cfg, task), stdout=log, stderr=subprocess.STDOUT, env=env, **kwargs)
    return proc, log


def stop(proc):
    if proc.poll() is not None:
        return
    try:
        if os.name == 'posix':
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------- 锁：同一配置同时只运行一个
def acquire_lock(path):
    """返回锁文件的描述符；已被其他运行（或它启动、仍在运行的任务）占用时返回 None。进程退出时锁自动释放"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            os.lseek(fd, 1 << 20, os.SEEK_SET)  # 锁住内容之外的一个字节
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        os.close(fd)
        return None
    return fd


def write_lock_info(fd, running):
    """锁文件里记下编排进程与正在运行的任务的进程号，被拒绝的启动会显示它们"""
    if os.name != 'posix':
        return
    lines = ['run_all.py pid {} (updated {})'.format(os.getpid(), stamp())]
    lines += ['{} pid {}'.format(tid, proc.pid) for tid, (proc, *_) in running.items()]
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, ('\n'.join(lines) + '\n').encode())


# ---------------------------------------------------------------- 硬件检查
def hardware_holds(cfg, tasks, pending, gpu_ids, no_gpu=False):
    """本机暂不运行的任务 {id: 原因}：硬件不符合 hardware 段的任务，以及依赖它们的任务"""
    if no_gpu:
        return {tid: 'no GPU visible' for tid in pending}
    from experiments import gpu_names, cpu_model
    names = gpu_names()
    used = [names[int(g)] for g in gpu_ids if g.isdigit() and int(g) < len(names)] or names
    cpu = [cpu_model()]
    reasons = {}
    for tid in pending:
        req = hardware_requirement(cfg, tasks[tid])
        if req:
            key, model = req
            have = used if key == 'timing_gpu' else cpu
            if not hardware_matches(model, have):
                reasons[tid] = 'needs {} (this machine: {})'.format(model, ', '.join(have) or 'no GPU')
    changed = True
    while changed:
        changed = False
        for tid in pending:
            if tid not in reasons:
                dep = next((d for d in tasks[tid].deps if d in reasons), None)
                if dep is not None:
                    reasons[tid] = 'waits for {}'.format(dep)
                    changed = True
    return reasons


def print_held(held):
    roots = Counter(r for r in held.values() if not r.startswith('waits for'))
    waiting = sum(1 for r in held.values() if r.startswith('waits for'))
    print('[{}] {} tasks held back on this machine (they run when the same command is started on matching hardware):'
          .format(stamp(), len(held)))
    for reason, n in roots.items():
        print('         {}: {} tasks'.format(reason, n))
    if waiting:
        print('         waiting for held tasks: {} tasks'.format(waiting))


def record_environment(cfg, gpu_ids, tasks_per_gpu):
    """把本次运行的硬件、配额与软件版本追加到 results/<配置名>/environment.json（论文 §5.1.3 据此书写）"""
    from experiments import environment
    path = os.path.join(cfg.paths.results, 'environment.json')
    records = []
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            records = json.load(f)
    records.append(dict(environment(len(gpu_ids)), gpu_slot_ids=list(gpu_ids), tasks_per_gpu=str(tasks_per_gpu),
                        metaheuristic_workers=cfg.solvers.workers, config=cfg.config_path))
    os.makedirs(cfg.paths.results, exist_ok=True)
    save_json(records, path, indent=1, ensure_ascii=False)


def print_status(tasks, selected):
    """selected：选中的任务 id 集合"""
    counts = Counter((t.stage, is_done(t)) for tid, t in tasks.items() if tid in selected)
    for stage in STAGES:
        done, pending = counts[(stage, True)], counts[(stage, False)]
        if done or pending:
            print('  {:<10} done {:>4}  pending {:>4}'.format(stage, done, pending))


def run(cfg, tasks, selected, gpu_ids, policy, held=(), lock_fd=None):
    """gpu_ids：卡号列表（同一卡号重复出现时容量加倍）；policy：parse_tasks_per_gpu 的结果；held：暂不运行的任务"""
    held = set(held)
    todo = {tid: t for tid, t in tasks.items() if tid in selected and tid not in held and not is_done(t)}
    # 汇总的依赖中有任务要运行时，汇总也要在其后重新运行
    for tid, t in tasks.items():
        if t.meta.get('dep_markers') and tid in selected and tid not in held and any(d in todo for d in t.deps):
            todo[tid] = t
    failed = set()
    running = {}  # task id -> (proc, log, slot, start)；slot 为 'cpu' 或 (卡号, 占用的容量)
    # 每张卡的容量 = 各规模任务数的最小公倍数：每卡 K 个的任务各占 1/K（如 20:3,50:3,100:2 时容量 6，小任务占 2，大任务占 3）
    units = lcm(*policy.values())
    free = {g: units * n for g, n in Counter(gpu_ids).items()}
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
                    free[slot[0]] += slot[1]
                ok = proc.returncode == 0 and is_done(tasks[tid])
                print('[{}] {} {} ({:.1f} min)'.format(stamp(), 'done  ' if ok else 'FAILED', tid, (time.time() - start) / 60))
                if not ok:
                    failed.add(tid)
                    print('         see {}'.format(log_path(cfg, tasks[tid])))
                if lock_fd is not None:
                    write_lock_info(lock_fd, running)
            # 依赖失败或未完成且不会运行的任务无法开始
            for tid, t in list(todo.items()):
                blocked = [d for d in t.deps if d in failed or (d not in todo and d not in running and not is_done(tasks[d]))]
                if blocked:
                    print('[{}] skip   {} (dependency not available: {})'.format(stamp(), tid, blocked[0]))
                    failed.add(tid)
                    del todo[tid]
            # 启动依赖已满足的任务；有就绪的独占任务时不再启动其他任务，等正在运行的任务结束后单独运行它
            ready = [tid for tid, t in todo.items() if all(is_done(tasks[d]) for d in t.deps)]
            ready.sort(key=lambda tid: (tasks[tid].priority, -expected_hours(tasks[tid])))  # 优先级高的、最长的先运行
            exclusive_ready = [tid for tid in ready if tasks[tid].exclusive]
            if any(tasks[tid].exclusive for tid in running):
                ready = []
            elif exclusive_ready:
                ready = exclusive_ready[:1] if not running else []
            blocked = None  # 单独占卡的任务放不下时，优先级更低的 GPU 任务不再上卡，让卡空出来给它
            for tid in ready:
                t = tasks[tid]
                if t.exclusive and running:
                    break
                if t.resource == 'gpu':
                    if blocked is not None and t.priority > blocked:
                        continue
                    need = units if (t.exclusive or t.solo) else units // tasks_per_gpu_for(policy, gpu_task_size(t))
                    fits = [g for g in free if free[g] >= need]
                    if not fits:
                        if t.solo and blocked is None:
                            blocked = t.priority
                        continue  # 放不下这个任务时，较小的任务仍可填进卡的空余
                    gpu = max(fits, key=lambda g: free[g])  # 最空的卡：最先派发的长任务分散到不同的卡上
                    free[gpu] -= need
                    slot = (gpu, need)
                else:
                    if cpu_busy:
                        continue
                    slot, cpu_busy = 'cpu', True
                proc, log = launch(cfg, t, gpu=None if slot == 'cpu' else slot[0], lock_fd=lock_fd)
                running[tid] = (proc, log, slot, time.time())
                del todo[tid]
                print('[{}] start  {}{}{}'.format(stamp(), tid, '' if slot == 'cpu' else ' (GPU {})'.format(slot[0]),
                                                  ' [exclusive]' if t.exclusive else ' [solo]' if t.solo else ''))
                if lock_fd is not None:
                    write_lock_info(lock_fd, running)
                if t.exclusive:
                    break
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print('[{}] Interrupted, stopping running tasks (they resume on the next run)'.format(stamp()))
        for proc, log, slot, start in running.values():
            stop(proc)
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
        ids = [tid for tid, t in tasks.items() if t.stage == stage and tid in selected and not is_done(t)]
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


def _interrupt(signum, frame):
    raise KeyboardInterrupt  # kill <pid>（SIGTERM）与 Ctrl-C 一样：先结束正在运行的任务再退出


def main():
    parser = argparse.ArgumentParser(description="Run every experiment of the paper")
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--only', default=None, help='Comma-separated stages to run')
    parser.add_argument('--gpus', default='0',
                        help="Comma-separated GPU ids (repeat an id to double its capacity), or 'all' for every "
                             "visible GPU")
    parser.add_argument('--tasks-per-gpu', default='1',
                        help='GPU tasks run concurrently on each GPU (small models leave the GPU partly idle): one '
                             'number, or per size, e.g. 20:3,50:3,100:2 (sizes up to 20 and 50: 3 per GPU, larger: 2)')
    parser.add_argument('--match', default=None, metavar='REGEX',
                        help='Only run tasks whose id matches this regular expression (e.g. a pilot subset)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--task', default=None, help='Run a single task id')
    parser.add_argument('--emit-slurm', default=None, metavar='DIR')
    opts = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # nohup 重定向到文件时也逐行写出进度

    policy = parse_tasks_per_gpu(opts.tasks_per_gpu)
    no_gpu = False
    if opts.gpus == 'all':
        from experiments import gpu_names
        n_gpus = len(gpu_names())
        no_gpu = n_gpus == 0
        gpu_ids = [str(i) for i in range(max(1, n_gpus))]
    else:
        gpu_ids = [g for g in opts.gpus.split(',') if g != '']
    # workers: auto 为每个可能同时运行的 GPU 任务预留一个物理核
    cfg = load_config(opts.config, gpu_slots=len(gpu_ids) * max_tasks_per_gpu(policy))
    tasks = build_tasks(cfg)
    stages = set(opts.only.split(',')) if opts.only else set(STAGES)
    match = re.compile(opts.match) if opts.match else None
    selected = {tid for tid, t in tasks.items() if t.stage in stages and (match is None or match.search(tid))}
    print('Config {}: {} tasks'.format(opts.config, len(tasks)))
    print_status(tasks, selected)

    if opts.task is not None:
        t = tasks[opts.task]
        if is_done(t):
            print('Already done:', t.id)
            return
        sys.exit(subprocess.call(resolve_cmd(cfg, t)))
    if opts.emit_slurm:
        emit_slurm(cfg, tasks, selected, opts.emit_slurm, opts.config)
        return
    pending = [tid for tid, t in tasks.items() if tid in selected and not is_done(t)]
    held = hardware_holds(cfg, tasks, pending, gpu_ids, no_gpu)
    if opts.dry_run:
        for tid in pending:
            print('  [{}] {}{}'.format(tasks[tid].stage, tid, '  (held: {})'.format(held[tid]) if tid in held else ''))
        if held:
            print_held(held)
        return
    if not any(tid not in held for tid in pending):  # 没有可运行的任务：不加锁，也不记录运行环境
        if held:
            print_held(held)
            sys.exit(4)
        print('Nothing to run')
        return

    lock_path = os.path.join(cfg.paths.results, 'run_all.lock')
    lock_fd = acquire_lock(lock_path)
    if lock_fd is None:
        print('Another run_all.py with this config, or tasks it started, is still running:')
        try:
            with open(lock_path, encoding='utf-8', errors='replace') as f:
                print(f.read().rstrip())
        except OSError:
            pass
        print('Wait for it to finish, or stop it first: kill <run_all.py pid>; leftover tasks: '
              'pkill -f "run.py|eval.py|BaselineMethods"')
        sys.exit(2)
    signal.signal(signal.SIGTERM, _interrupt)
    if held:
        print_held(held)
    print('GPUs: {}; tasks per GPU: {}; metaheuristic workers: {}'.format(
        ','.join(gpu_ids), opts.tasks_per_gpu, cfg.solvers.workers))
    record_environment(cfg, gpu_ids, opts.tasks_per_gpu)
    try:
        failed = run(cfg, tasks, selected, gpu_ids, policy, held, lock_fd)
    except KeyboardInterrupt:
        sys.exit(130)
    print_status(tasks, selected)
    if held:
        print_held(held)
    if failed:
        print('{} tasks failed or were skipped: {}'.format(len(failed), ', '.join(sorted(failed))))
        sys.exit(1)
    if held:
        sys.exit(3)


if __name__ == "__main__":
    main()
