"""
编排与续跑的测试：按规模分配每卡任务数、硬件检查、防重复启动的锁、原子写入、续训时截断日志、估算的墙钟时间。
调度测试用只睡眠、写结果文件的假任务，不需要显卡。
运行：uv run pytest -q
"""
import csv
import json
import os
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

import run_all
from atomic_io import save_json, _write
from experiments import (Task, parse_tasks_per_gpu, tasks_per_gpu_for, max_tasks_per_gpu, gpu_task_size,
                         hardware_requirement, hardware_matches, load_config, build_tasks)
from utils.log_utils import truncate_csv_log

FAKE = ('import json, os, sys, time; t0 = time.time(); time.sleep(float(sys.argv[2])); '
        'json.dump({"start": t0, "end": time.time(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}, '
        'open(sys.argv[1], "w"))')


def fake(tid, stage, resource, marker, deps=(), size=20, seconds=0.3, exclusive=False):
    return Task(id=tid, stage=stage, resource=resource, cmd=[sys.executable, '-c', FAKE, str(marker), str(seconds)],
                marker=str(marker), deps=list(deps), meta={'variant': 'ethc', 'size': size}, exclusive=exclusive)


def make_cfg(tmp_path, hardware=None):
    return OmegaConf.create({'paths': {'results': str(tmp_path)}, 'hardware': hardware or {}})


def test_parse_tasks_per_gpu():
    assert parse_tasks_per_gpu('2') == {None: 2}
    p = parse_tasks_per_gpu('20:3,50:3,100:2')
    assert p == {20: 3, 50: 3, 100: 2}
    # 图4的测试规模 30、70 归入不小于它的最近规模；超过所有键或规模未知时取最大键（偏保守）
    assert [tasks_per_gpu_for(p, s) for s in (20, 30, 50, 70, 100, 200, None)] == [3, 3, 3, 2, 2, 2, 2]
    assert max_tasks_per_gpu(p) == 3 and tasks_per_gpu_for({None: 2}, 100) == 2
    with pytest.raises(ValueError):
        parse_tasks_per_gpu('20:0')


def test_gpu_task_size_and_hardware_requirements():
    cfg = load_config('configs/autodl.yaml', gpu_slots=6)
    tasks = build_tasks(cfg)
    assert gpu_task_size(tasks['train:ethc_n50_l3_s1234']) == 50
    assert gpu_task_size(tasks['eval:ethc_n20_l3_s1234__test100__greedy_soft-hard']) == 100  # 评估取较大的规模
    assert gpu_task_size(tasks['eval:ethc_n100_l3_s1234__test30__greedy_soft-hard']) == 100
    assert hardware_requirement(cfg, tasks['solve:ortools_n50_l1']) == ('solver_cpu', 'Xeon(R) Platinum 8470Q')
    assert hardware_requirement(cfg, tasks['solve:ortools_eq_n100_l3']) == ('solver_cpu', 'Xeon(R) Platinum 8470Q')
    timing = [t for t in tasks.values() if t.stage == 'timing']
    assert len(timing) == 12 and all(hardware_requirement(cfg, t) == ('timing_gpu', 'RTX 5090') for t in timing)
    assert hardware_requirement(cfg, tasks['train:ethc_n100_l3_s1234']) is None
    assert hardware_requirement(load_config('configs/paper.yaml', gpu_slots=1), tasks['solve:ortools_n50_l1']) is None
    assert hardware_matches('RTX 5090', ['NVIDIA GeForce RTX 5090', 'NVIDIA GeForce RTX 5090'])
    assert not hardware_matches('RTX 5090', ['NVIDIA GeForce RTX 5090', 'NVIDIA GeForce RTX 4090'])
    assert not hardware_matches('RTX 5090', [])
    assert hardware_matches('Xeon(R) Platinum 8470Q', ['Intel(R) Xeon(R) Platinum 8470Q'])


def test_per_size_slots(tmp_path, monkeypatch):
    """每卡容量 6：n=20 的任务各占 2（每卡 3 个），n=100 的各占 3（每卡 2 个），任何时刻都不超过容量"""
    monkeypatch.setattr(run_all, 'POLL_SECONDS', 0.05)
    tasks = {}
    for i in range(4):
        t = fake('train:small{}'.format(i), 'train', 'gpu', tmp_path / 'small{}.json'.format(i), size=20, seconds=0.8)
        tasks[t.id] = t
    for i in range(3):
        t = fake('train:large{}'.format(i), 'train', 'gpu', tmp_path / 'large{}.json'.format(i), size=100, seconds=0.8)
        tasks[t.id] = t
    failed = run_all.run(make_cfg(tmp_path), tasks, set(tasks), ['0', '1'], parse_tasks_per_gpu('20:3,100:2'))
    assert not failed
    rec = {tid: json.load(open(t.marker)) for tid, t in tasks.items()}
    units = {20: 2, 100: 3}
    times = sorted(r['start'] for r in rec.values())
    peak = 0
    for gpu in ('0', '1'):
        for x in times:
            on = [tid for tid, r in rec.items() if r['gpu'] == gpu and r['start'] <= x < r['end']]
            assert sum(units[tasks[tid].meta['size']] for tid in on) <= 6
            peak = max(peak, len(on))
    assert peak >= 2  # 同一张卡上确实同时运行了多个任务


def test_hardware_holds_and_dependents(tmp_path, monkeypatch):
    """硬件不符的任务与依赖它们的任务暂不运行，其余照常；看不到显卡时一个任务也不运行"""
    monkeypatch.setattr(run_all, 'POLL_SECONDS', 0.05)
    cfg = make_cfg(tmp_path, {'timing_gpu': 'NoSuchGPU 9999', 'solver_cpu': 'NoSuchCPU 9999'})
    train = fake('train:a', 'train', 'gpu', tmp_path / 'a.json')
    solve = fake('solve:b', 'solve', 'cpu', tmp_path / 'b.json')
    timing = fake('eval:c', 'timing', 'gpu', tmp_path / 'c.json', deps=['train:a', 'solve:b'], exclusive=True)
    after = fake('analysis', 'analysis', 'cpu', tmp_path / 'd.json', deps=['eval:c'])
    tasks = {t.id: t for t in (train, solve, timing, after)}
    held = run_all.hardware_holds(cfg, tasks, list(tasks), ['0'])
    assert set(held) == {'solve:b', 'eval:c', 'analysis'}
    assert held['solve:b'].startswith('needs NoSuchCPU 9999') and held['analysis'] == 'waits for eval:c'
    failed = run_all.run(cfg, tasks, set(tasks), ['0'], {None: 1}, held)
    assert not failed and os.path.isfile(train.marker)
    assert not any(os.path.exists(t.marker) for t in (solve, timing, after))
    assert set(run_all.hardware_holds(cfg, tasks, list(tasks), ['0'], no_gpu=True)) == set(tasks)


def test_lock_refuses_second_run(tmp_path):
    path = str(tmp_path / 'run_all.lock')
    fd = run_all.acquire_lock(path)
    assert fd is not None
    assert run_all.acquire_lock(path) is None
    os.close(fd)  # 运行结束（或进程退出）后锁自动释放
    fd = run_all.acquire_lock(path)
    assert fd is not None
    os.close(fd)


@pytest.mark.skipif(os.name != 'posix', reason='tasks inherit the lock only on POSIX')
def test_lock_held_by_running_tasks(tmp_path):
    """编排进程已经退出、它启动的任务还在运行时，新的运行也被拒绝"""
    path = str(tmp_path / 'run_all.lock')
    fd = run_all.acquire_lock(path)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], pass_fds=(fd,),
                             start_new_session=True)
    os.close(fd)
    assert run_all.acquire_lock(path) is None
    child.wait()
    fd = run_all.acquire_lock(path)
    assert fd is not None
    os.close(fd)


def test_atomic_write_keeps_old_file(tmp_path):
    """写到一半出错（相当于断电或被杀）时，原文件保持上一次的完整内容，也不留下临时文件"""
    path = str(tmp_path / 'x.json')
    save_json({'a': 1}, path)

    def interrupted(f):
        f.write('{"partial": ')
        raise RuntimeError('killed while writing')

    with pytest.raises(RuntimeError):
        _write(path, interrupted, 'w', encoding='utf-8')
    assert json.load(open(path, encoding='utf-8')) == {'a': 1}
    assert os.listdir(tmp_path) == ['x.json']


def test_truncate_csv_log(tmp_path):
    path = str(tmp_path / 'train_log.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write('step,epoch,cost\n')
        for epoch in range(3):
            for step in range(2):
                f.write('{},{},1.0\n'.format(epoch * 2 + step, epoch))
        f.write('6,3')  # 中断时写了一半的行
    assert truncate_csv_log(path, 1) == 3
    assert open(path, encoding='utf-8').read().split() == ['step,epoch,cost', '0,0,1.0', '1,0,1.0', '2,1,1.0', '3,1,1.0']
    assert truncate_csv_log(path, 1) == 0
    assert truncate_csv_log(path, -1) == 4  # 从头训练：清空旧记录
    assert open(path, encoding='utf-8').read().split() == ['step,epoch,cost']


def test_resumed_training_rewrites_interrupted_epoch(tmp_path):
    """训练在 epoch 1 中途中断后续训：日志中删去中断前写入的行，步数不重复；验证记录每个 epoch 一行"""
    out = str(tmp_path / 'run')
    base = [sys.executable, 'run.py', '--problem', 'cvrptw', '--graph_size', '20', '--baseline', 'rollout',
            '--save_dir', out, '--resume', 'auto', '--seed', '1', '--val_dataset',
            'data/cvrptw/cvrptw20_validation_seed4321.pkl', '--val_size', '50', '--epoch_size', '512',
            '--batch_size', '256', '--log_step', '1', '--eval_batch_size', '256', '--no_progress_bar', '--no_cuda',
            '--model', 'attention', '--n_encode_layers', '1', '--embedding_dim', '16', '--hidden_dim', '16',
            '--bl_warmup_epochs', '1']
    subprocess.run(base + ['--n_epochs', '1'], check=True, capture_output=True)
    log = os.path.join(out, 'train_log.csv')
    with open(log, 'a', encoding='utf-8') as f:  # 中断前 epoch 1 已写入一行，另有写了一半的行
        f.write('2,1,0,9.9,0,0,0,0,0,\n3,1')
    subprocess.run(base + ['--n_epochs', '2'], check=True, capture_output=True)
    rows = list(csv.DictReader(open(log, encoding='utf-8')))
    assert [int(r['step']) for r in rows] == [0, 1, 2, 3]
    assert [int(r['epoch']) for r in rows] == [0, 0, 1, 1]
    assert all(r['cost'] != '9.9' for r in rows)
    val = list(csv.DictReader(open(os.path.join(out, 'val_log.csv'), encoding='utf-8')))
    assert [int(r['epoch']) for r in val] == [0, 1]
    assert json.load(open(os.path.join(out, 'done.json')))['resumed_from'] == 1


def test_wall_clock_per_size():
    from estimate import wall_clock
    t20 = Task('train:a', 'train', 'gpu', [], 'a', meta={'variant': 'ethc', 'size': 20})
    t100 = Task('train:b', 'train', 'gpu', [], 'b', meta={'variant': 'ethc', 'size': 100})
    per_task = [(t20, 9.0, 0.0)] * 6 + [(t100, 30.0, 0.0)] * 2
    assert wall_clock(per_task, 1, 1)['gpu'] == pytest.approx(6 * 9 + 2 * 30)
    packed = wall_clock(per_task, 1, 1, parse_tasks_per_gpu('20:3,100:2'), {20: 2.0, 100: 1.5})
    assert packed['gpu'] == pytest.approx(6 * 9 / 2.0 + 2 * 30 / 1.5)
    assert packed['longest_task'] == pytest.approx(30 * 2 / 1.5)
