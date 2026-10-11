"""
实验清单的测试：configs/paper.yaml 展开后的任务与论文第5节的实验设计一致，依赖关系完整。
运行：uv run pytest -q
"""
from collections import Counter

import pytest

from experiments import load_config, build_tasks, auto_workers


@pytest.fixture(scope='module')
def paper():
    cfg = load_config('configs/paper.yaml')
    return cfg, build_tasks(cfg)


def test_training_runs(paper):
    cfg, tasks = paper
    train = [t for t in tasks.values() if t.stage == 'train']
    assert len(train) == 48
    assert Counter(t.meta['variant'] for t in train) == {'ethc': 21, 'ethc_woet': 9, 'tpa': 15, 'mvmoe': 3}
    # 图3：每个 λ、每个规模各一个 ETHC-VRC 与 TPA-Reward 模型
    for size in cfg.sizes:
        for lmbda in cfg.lambdas:
            assert 'train:ethc_n{}_l{}_s1234'.format(size, lmbda) in tasks
            assert 'train:tpa_n{}_l{}_s1234'.format(size, lmbda) in tasks
    # 消融：full 与 w/o ET 各 3 个种子
    for size in cfg.sizes:
        for seed in cfg.seeds:
            assert 'train:ethc_n{}_l3_s{}'.format(size, seed) in tasks
            assert 'train:ethc_woet_n{}_l3_s{}'.format(size, seed) in tasks


def test_solvers_and_dependencies(paper):
    cfg, tasks = paper
    solve = [t for t in tasks.values() if t.stage == 'solve']
    assert Counter(t.meta['method'] for t in solve) == {'ortools': 15, 'ts': 3, 'ba': 3}
    # 等时比较：λ=3 的三个规模，各依赖本规模 ETHC-VRC(S) 的计时评估
    equal = [t for t in tasks.values() if t.stage == 'equal_time']
    assert sorted(t.meta['size'] for t in equal) == list(cfg.sizes)
    for t in equal:
        assert t.deps == ['eval:ethc_n{0}_l3_s1234__test{0}__sample_soft-hard_1280'.format(t.meta['size'])]
    for t in tasks.values():
        for d in t.deps:
            assert d in tasks, (t.id, d)
    timing = [t for t in tasks.values() if t.exclusive]
    assert len(timing) == 12  # 表2：3 个规模 × ETHC-VRC(G)(S)、TPA-Reward(G)、MVMoE(G)
    assert all(t.stage == 'timing' for t in timing)
    others = {tid for tid, t in tasks.items() if t.stage in ('train', 'eval', 'solve') and not t.exclusive}
    for t in timing:
        assert others <= set(t.deps)


def test_paper_settings(paper):
    cfg, tasks = paper
    t = tasks['train:ethc_n100_l3_s1234']
    cmd = ' '.join(t.cmd)
    for arg in ('--epoch_size 1280000', '--batch_size 512', '--n_epochs 100', '--lr_model 0.0001',
                '--bl_alpha 0.05', '--bl_warmup_epochs 0', '--penalty_lambda 3', '--seed 1234',
                '--n_encode_layers 3', '--n_heads 8', '--embedding_dim 128'):
        assert arg in cmd, arg
    s = ' '.join(tasks['eval:ethc_n100_l3_s1234__test100__sample_soft-hard_1280'].cmd)
    assert '--samples 1280' in s and '--instances_per_pass 1' in s and '--modes soft hard' in s
    g = ' '.join(tasks['eval:ethc_n100_l3_s1234__test100__greedy_soft-hard'].cmd)
    assert '--eval_batch_size 1024' in g


def test_auto_workers():
    # 按物理核计：AutoDL 单卡 25 个超线程 → 12 个物理核，1 个 GPU 任务 → 10 个求解进程
    assert auto_workers(25, 1, 2) == 10
    assert auto_workers(200, 16, 2) == 83  # 8 卡、每卡 2 个 GPU 任务
    assert auto_workers(12, 1, 1) == 10    # 无超线程
    assert auto_workers(4, 8, 2) == 1      # 至少 1 个
