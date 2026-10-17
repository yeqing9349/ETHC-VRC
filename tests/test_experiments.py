"""
实验清单的测试：configs/paper.yaml 展开后的任务与论文第5节的实验设计一致，依赖关系完整。
运行：uv run pytest -q
"""
from collections import Counter

import pytest

from experiments import load_config, build_tasks, auto_workers, environment


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
        assert t.deps == ['eval:ethc_n{0}_l3_s1234__test{0}__sample_soft-hard_1280_timing'.format(t.meta['size'])]
    for t in tasks.values():
        for d in t.deps:
            assert d in tasks, (t.id, d)
    timing = [t for t in tasks.values() if t.exclusive]
    assert len(timing) == 18  # 表2：3 个规模 × ETHC-VRC(G)(S)、TPA-Reward(G)(S)、MVMoE(G)(S)
    assert all(t.stage == 'timing' and t.id.endswith('_timing') for t in timing)
    others = {tid for tid, t in tasks.items() if t.stage in ('train', 'eval', 'solve') and not t.exclusive}
    for t in timing:
        assert others <= set(t.deps)
        # 成本取同名（去掉 _timing）的评估：它只依赖模型，模型训练完就运行
        q = tasks[t.id[:-len('_timing')]]
        assert q.stage == 'eval' and not q.exclusive and q.deps == t.deps[:1] and '--timing' not in q.cmd
        assert [a for a in q.cmd if a != q.marker[:-5]] == [a for a in t.cmd if a not in (t.marker[:-5], '--timing')]


def test_paper_settings(paper):
    cfg, tasks = paper
    t = tasks['train:ethc_n100_l3_s1234']
    cmd = ' '.join(t.cmd)
    for arg in ('--epoch_size 1280000', '--batch_size 512', '--n_epochs 100', '--lr_model 0.0001',
                '--bl_alpha 0.05', '--bl_warmup_epochs 1', '--exp_beta 0.8', '--penalty_lambda 3', '--seed 1234',
                '--n_encode_layers 3', '--n_heads 8', '--embedding_dim 128'):
        assert arg in cmd, arg
    s = ' '.join(tasks['eval:ethc_n100_l3_s1234__test100__sample_soft-hard_1280'].cmd)
    assert '--samples 1280' in s and '--instances_per_pass 1' in s and '--modes soft hard' in s
    g = ' '.join(tasks['eval:ethc_n100_l3_s1234__test100__greedy_soft-hard'].cmd)
    assert '--eval_batch_size 1024' in g
    # 同预算采样：两个 DRL 基线在各自的掩码模式下每个实例采样 2560 个解，与表2的其他评估一样计时
    for variant, mode in (('tpa', 'soft'), ('mvmoe', 'hard')):
        t = tasks['eval:{}_n100_l3_s1234__test100__sample_{}_2560'.format(variant, mode)]
        b = ' '.join(t.cmd)
        assert '--samples 2560' in b and '--instances_per_pass 1' in b and '--timing' not in b
        assert '--modes {} --num_instances'.format(mode) in b
        assert not t.exclusive and t.deps == ['train:{}_n100_l3_s1234'.format(variant)]
        timed = tasks[t.id + '_timing']
        assert timed.exclusive and '--timing' in timed.cmd and timed.deps[0] == t.deps[0]


def test_schedule(paper):
    """派发顺序：表2按规模从小到大 → 多种子消融、图4、图6 → λ 扫描 → 表2计时、等时比较与汇总；
    n=50、n=100 表2的 ETHC-VRC 与 MVMoE 训练单独占卡"""
    cfg, tasks = paper
    p = {tid: t.priority for tid, t in tasks.items()}
    t2 = [[p[tid] for tid in ('train:ethc_n{}_l3_s1234'.format(n), 'train:tpa_n{}_l3_s1234'.format(n),
                              'eval:mvmoe_n{0}_l3_s1234__test{0}__sample_hard_2560'.format(n),
                              'solve:ortools_n{}_l3'.format(n), 'solve:ba_n{}_l3'.format(n))] for n in cfg.sizes]
    assert all(len(set(row)) == 1 for row in t2) and t2[0][0] < t2[1][0] < t2[2][0]
    ablation = {p['train:ethc_n20_l3_s1235'], p['train:ethc_woet_n100_l3_s1234'],
                p['eval:ethc_n50_l3_s1234__test50__sample_soft_2560'],
                p['eval:ethc_n20_l3_s1234__test100__greedy_soft-hard'],  # 图4
                p['eval:ethc_n100_l3_s1234__test100__sample_soft-hard_1280_fig6']}
    fig3 = {p['train:ethc_n100_l9_s1234'], p['eval:tpa_n20_l1_s1234__test20__greedy_soft'], p['solve:ortools_n50_l7']}
    final = {t.priority for t in tasks.values() if t.stage in ('timing', 'equal_time', 'analysis')}
    assert len(ablation) == len(fig3) == len(final) == 1
    assert t2[2][0] < min(ablation) < min(fig3) < min(final)
    assert p['data'] == p['calibrate'] < t2[0][0]
    assert {tid for tid, t in tasks.items() if t.solo} == {
        'train:{}_n{}_l3_s1234'.format(v, n) for v in ('ethc', 'mvmoe') for n in (50, 100)}
    flat = load_config('configs/paper.yaml')
    flat.schedule = None  # 没有 schedule 段时都为 0，只按预计时长派发
    assert {(t.priority, t.solo) for t in build_tasks(flat).values()} == {(0, False)}


def test_environment_record():
    rec = environment(2)
    assert rec['cpu_model'] and rec['logical_cpus'] >= 1 and rec['gpu_slots'] == 2
    assert 'torch' in rec['packages'] and 'ortools' in rec['packages']


def test_auto_workers():
    # 按物理核计：AutoDL 单卡 25 个超线程 → 12 个物理核，1 个 GPU 任务 → 10 个求解进程
    assert auto_workers(25, 1, 2) == 10
    assert auto_workers(200, 16, 2) == 83  # 8 卡、每卡 2 个 GPU 任务
    assert auto_workers(12, 1, 1) == 10    # 无超线程
    assert auto_workers(4, 8, 2) == 1      # 至少 1 个
