"""
实验清单：把 configs/*.yaml 展开成任务（训练、评估、元启发式求解、时限标定、计时探针、汇总），
并统一各任务的输出路径与命名，run_all.py、estimate.py 和 analysis/report.py 共用。
每个任务是一条命令行；输出文件（marker）存在即视为已完成。
"""
import os
import sys
from dataclasses import dataclass, field

from omegaconf import OmegaConf

PY = sys.executable
MAIN_LAMBDA = 3


@dataclass
class Task:
    id: str
    stage: str  # data / probe / calibrate / train / eval / solve / timing / equal_time / analysis
    resource: str  # gpu / cpu
    cmd: list
    marker: str
    deps: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    exclusive: bool = False  # 计时任务：运行时不与其他任务并行，避免 CPU 争用影响计时


def _load_with_base(path):
    cfg = OmegaConf.load(path)
    if 'base' in cfg:  # smoke.yaml、autodl.yaml 等在 base 配置上覆盖部分字段（可多级）
        base = _load_with_base(os.path.join(os.path.dirname(path), cfg.base))
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def available_cpus():
    """本进程可用的 CPU 核数：优先读容器的 cgroup 配额（AutoDL 等容器中 os.cpu_count() 是整台机器的核数）"""
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()  # cgroup v2
        if quota != 'max':
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())  # cgroup v1
        period = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def gpu_count():
    import subprocess
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30).stdout
        return sum(1 for line in out.splitlines() if line.startswith('GPU '))
    except (OSError, subprocess.SubprocessError):
        return 0


def load_config(path, gpu_slots=None):
    """gpu_slots：同时运行的 GPU 任务数（run_all.py 的 GPU 槽位数），默认每张卡一个"""
    cfg = _load_with_base(path)
    cfg.config_path = path
    # 环境变量 ETHC_OUTPUTS 可把模型检查点放到别处（例如不在云同步目录里），结果文件仍按配置存放
    if os.environ.get('ETHC_OUTPUTS'):
        cfg.paths.outputs = os.path.join(os.environ['ETHC_OUTPUTS'], cfg.name)
    # 元启发式的并行进程数：auto = 可用核数 - 每个 GPU 任务 2 个核 - 2 个余量
    if str(cfg.solvers.workers) == 'auto':
        cfg.solvers.workers = max(1, available_cpus() - 2 * (gpu_slots or gpu_count()) - 2)
    return cfg


def fmt_lambda(lmbda):
    return '{:g}'.format(lmbda)


def run_name(variant, size, lmbda, seed):
    """variant: ethc / ethc_woet / tpa / mvmoe"""
    return '{}_n{}_l{}_s{}'.format(variant, size, fmt_lambda(lmbda), seed)


def run_dir(cfg, variant, size, lmbda, seed):
    return os.path.join(cfg.paths.outputs, run_name(variant, size, lmbda, seed))


def eval_name(variant, size, lmbda, seed, test_size, strategy, modes, samples=None, tag=None):
    name = '{}__test{}__{}_{}'.format(run_name(variant, size, lmbda, seed), test_size, strategy, '-'.join(modes))
    if strategy == 'sample' and samples is not None:
        name += '_{}'.format(samples)
    if tag:
        name += '_' + tag
    return name


def eval_out(cfg, *args, **kwargs):
    return os.path.join(cfg.paths.results, 'eval', eval_name(*args, **kwargs))


def solve_out(cfg, method, size, lmbda):
    return os.path.join(cfg.paths.results, 'solvers', '{}_n{}_l{}'.format(method, size, fmt_lambda(lmbda)))


def calibration_path(cfg):
    return cfg.solvers.calibration


def time_limit(cfg, size):
    """元启发式的单实例时限：配置给定，或读取冒烟阶段的标定结果"""
    if cfg.solvers.get('time_limit') is not None:
        return float(cfg.solvers.time_limit)
    import json
    with open(calibration_path(cfg), encoding='utf-8') as f:
        return float(json.load(f)['sizes'][str(size)]['time_limit_s'])


MODEL_ARGS = {
    'ethc': ['--model', 'attention'],
    'ethc_woet': ['--model', 'attention', '--no_time_state'],
    'tpa': ['--model', 'tpa'],
    'mvmoe': ['--model', 'mvmoe'],
}


def train_task(cfg, variant, size, lmbda, seed):
    t = cfg.train
    out = run_dir(cfg, variant, size, lmbda, seed)
    cmd = [PY, 'run.py', '--problem', 'cvrptw', '--graph_size', str(size), '--baseline', 'rollout',
           '--save_dir', out, '--resume', 'auto', '--seed', str(seed), '--penalty_lambda', fmt_lambda(lmbda),
           '--val_dataset', cfg.data.val.format(size=size), '--val_size', str(t.val_size),
           '--n_epochs', str(t.n_epochs), '--epoch_size', str(t.epoch_size), '--batch_size', str(t.batch_size),
           '--lr_model', str(t.lr), '--max_grad_norm', str(t.max_grad_norm), '--bl_alpha', str(t.bl_alpha),
           '--bl_warmup_epochs', str(t.bl_warmup_epochs), '--log_step', str(t.log_step),
           '--eval_batch_size', str(t.eval_batch_size), '--no_progress_bar'] + MODEL_ARGS[variant]
    for k, v in cfg.models.get(variant, {}).items():
        cmd += ['--' + k, str(v)]
    if t.tf32:
        cmd.append('--tf32')
    return Task(id='train:' + run_name(variant, size, lmbda, seed), stage='train', resource='gpu', cmd=cmd,
                marker=os.path.join(out, 'done.json'),
                meta={'variant': variant, 'size': size, 'lambda': lmbda, 'seed': seed})


def eval_task(cfg, variant, size, lmbda, seed, test_size, strategy, modes, samples=None, num=None, keep=0,
              tag=None, timing=False):
    model = os.path.join(run_dir(cfg, variant, size, lmbda, seed), 'epoch-{}.pt'.format(cfg.train.n_epochs - 1))
    samples = cfg.eval.samples if (strategy == 'sample' and samples is None) else samples
    out = eval_out(cfg, variant, size, lmbda, seed, test_size, strategy, modes, samples, tag)
    num = cfg.data.num_test if num is None else num
    cmd = [PY, 'eval.py', '--model', model, '--dataset', cfg.data.test.format(size=test_size), '--out', out,
           '--strategy', strategy, '--modes', *modes, '--num_instances', str(num),
           '--penalty_lambda', fmt_lambda(lmbda), '--eval_batch_size', str(cfg.eval.greedy_batch),
           '--seed', str(cfg.eval.seed)]
    if strategy == 'sample':
        cmd += ['--samples', str(samples), '--instances_per_pass', str(cfg.eval.instances_per_pass)]
    if keep:
        cmd += ['--keep_samples', str(keep)]
    if timing:
        cmd.append('--timing')
    if cfg.train.tf32:
        cmd.append('--tf32')
    # 计时评估单列为 timing 阶段：在其他任务之后独占运行，训练换到算力中心时可回到本机（论文所述硬件）运行
    return Task(id='eval:' + os.path.basename(out), stage='timing' if timing else 'eval', resource='gpu', cmd=cmd,
                marker=out + '.json', deps=['train:' + run_name(variant, size, lmbda, seed)], exclusive=timing,
                meta={'variant': variant, 'size': size, 'lambda': lmbda, 'seed': seed, 'test_size': test_size,
                      'strategy': strategy, 'modes': list(modes), 'samples': samples, 'num': num})


def sample_timing_id(cfg, size):
    """表2中 ETHC-VRC(S) 的计时评估任务（同规模测试集、λ=3、主种子）"""
    return 'eval:' + eval_name('ethc', size, MAIN_LAMBDA, cfg.seeds[0], size, 'sample', ['soft', 'hard'],
                               cfg.eval.samples)


def equal_time_limit(cfg, size):
    """等时比较的单实例时限：本机实测的 ETHC-VRC(S) 每实例耗时（两种模式合计）"""
    import json
    path = eval_out(cfg, 'ethc', size, MAIN_LAMBDA, cfg.seeds[0], size, 'sample', ['soft', 'hard'], cfg.eval.samples)
    with open(path + '.json', encoding='utf-8') as f:
        s = json.load(f)
    return s['time_s']['total'] / s['n']


def solve_task(cfg, method, size, lmbda, equal_time=False):
    """equal_time=True：OR-Tools 的单实例时限取 ETHC-VRC(S) 的实测每实例耗时（等时比较，表2之外另报）"""
    name = method + '_eq' if equal_time else method
    out = solve_out(cfg, name, size, lmbda)
    limit = '{EQUAL_TIME_%d}' % size if equal_time else '{TIME_LIMIT_%d}' % size
    cmd = [PY, '-m', 'BaselineMethods.solve', '--method', method, '--dataset', cfg.data.test.format(size=size),
           '--out', out, '--penalty_lambda', fmt_lambda(lmbda), '--time_limit', limit,
           '--max_routes', str(cfg.solvers.max_routes), '--workers', str(cfg.solvers.workers),
           '--num_instances', str(cfg.solvers.get('num_instances', cfg.data.num_test))]
    params = cfg.solvers.get(method)
    if params:
        import json
        cmd += ['--method_params', json.dumps(OmegaConf.to_container(params))]
    if equal_time:
        deps = [sample_timing_id(cfg, size)]
    else:
        deps = [] if cfg.solvers.get('time_limit') is not None else ['calibrate']
    return Task(id='solve:{}_n{}_l{}'.format(name, size, fmt_lambda(lmbda)),
                stage='equal_time' if equal_time else 'solve', resource='cpu', cmd=cmd, marker=out + '.json',
                deps=deps, meta={'method': name, 'size': size, 'lambda': lmbda, 'equal_time': equal_time})


def build_tasks(cfg):
    tasks = {}

    def add(task):
        tasks.setdefault(task.id, task)

    sizes, lambdas, seeds = list(cfg.sizes), list(cfg.lambdas), list(cfg.seeds)
    main_seed = seeds[0]

    # ---- 数据检查 ----
    data_check = os.path.join(cfg.paths.results, 'data_check.json')
    add(Task(id='data', stage='data', resource='cpu',
             cmd=[PY, 'generate_data.py', '--problem', 'cvrptw', '--check', data_check, '--graph_sizes',
                  *map(str, sorted(set(cfg.test_sizes) | set(sizes)))],
             marker=data_check))

    # ---- 冒烟阶段：计时探针 ----
    for probe in cfg.get('probes', []) or []:
        for size in probe.sizes:
            for variant in probe.variants:
                out = os.path.join(cfg.paths.results, 'probes', 'train_{}_n{}{}'.format(variant, size, '_tf32' if probe.get('tf32') else ''))
                cmd = [PY, 'run.py', '--problem', 'cvrptw', '--graph_size', str(size), '--baseline', 'rollout',
                       '--save_dir', out, '--probe_steps', str(probe.steps), '--batch_size', str(cfg.train.batch_size),
                       '--epoch_size', str(probe.epoch_size), '--val_size', str(probe.val_size),
                       '--eval_batch_size', str(cfg.train.eval_batch_size), '--n_epochs', str(probe.n_epochs),
                       '--no_progress_bar'] + MODEL_ARGS[variant]
                for k, v in cfg.models.get(variant, {}).items():
                    cmd += ['--' + k, str(v)]
                if probe.get('tf32'):
                    cmd.append('--tf32')
                add(Task(id='probe:' + os.path.basename(out), stage='probe', resource='gpu', cmd=cmd,
                         marker=os.path.join(out, 'probe.json'), exclusive=True,
                         meta={'variant': variant, 'size': size, 'tf32': bool(probe.get('tf32'))}))

    # ---- 元启发式时限标定（冒烟阶段运行；正式运行时若标定文件已存在则跳过） ----
    if cfg.solvers.get('run_calibration', cfg.solvers.get('time_limit') is None):
        c = cfg.solvers.calibrate
        add(Task(id='calibrate', stage='calibrate', resource='cpu',
                 cmd=[PY, '-m', 'BaselineMethods.calibrate', '--sizes', *map(str, sizes), '--limits', *map(str, c.limits),
                      '--num_instances', str(c.num_instances), '--tolerance', str(c.tolerance), '--margin', str(c.margin),
                      '--workers', str(cfg.solvers.workers), '--out', calibration_path(cfg),
                      '--dataset_pattern', cfg.data.test],
                 marker=calibration_path(cfg), meta={'sizes': sizes}))

    exp = cfg.experiments
    for size in sizes:
        # ---- 表2：主模型（λ=3，主种子），评估同规模测试集 ----
        if exp.table2 or exp.fig4:
            for variant in ('ethc', 'tpa', 'mvmoe'):
                add(train_task(cfg, variant, size, MAIN_LAMBDA, main_seed))
            test_sizes = list(cfg.test_sizes) if exp.fig4 else [size]
            for t in test_sizes:
                timing = t == size
                add(eval_task(cfg, 'ethc', size, MAIN_LAMBDA, main_seed, t, 'greedy', ['soft', 'hard'], timing=timing))
                add(eval_task(cfg, 'ethc', size, MAIN_LAMBDA, main_seed, t, 'sample', ['soft', 'hard'], timing=timing))
                add(eval_task(cfg, 'tpa', size, MAIN_LAMBDA, main_seed, t, 'greedy', ['soft'], timing=timing))
                add(eval_task(cfg, 'mvmoe', size, MAIN_LAMBDA, main_seed, t, 'greedy', ['hard'], timing=timing))
        # ---- 图3：λ 扫描（每个 λ 单独训练 ETHC 与 TPA-Reward） ----
        if exp.fig3:
            for lmbda in lambdas:
                for variant in ('ethc', 'tpa'):
                    add(train_task(cfg, variant, size, lmbda, main_seed))
                add(eval_task(cfg, 'ethc', size, lmbda, main_seed, size, 'greedy', ['soft', 'hard']))
                add(eval_task(cfg, 'ethc', size, lmbda, main_seed, size, 'sample', ['soft', 'hard']))
                add(eval_task(cfg, 'tpa', size, lmbda, main_seed, size, 'greedy', ['soft']))
        # ---- 图5与消融：full 与 w/o ET 各 3 个种子；w/o HC 用 full 的权重单模式采样 2×samples ----
        if exp.ablation:
            for seed in seeds:
                for variant in ('ethc', 'ethc_woet'):
                    add(train_task(cfg, variant, size, MAIN_LAMBDA, seed))
                    add(eval_task(cfg, variant, size, MAIN_LAMBDA, seed, size, 'greedy', ['soft', 'hard']))
                    add(eval_task(cfg, variant, size, MAIN_LAMBDA, seed, size, 'sample', ['soft', 'hard']))
                for mode in ('soft', 'hard'):
                    add(eval_task(cfg, 'ethc', size, MAIN_LAMBDA, seed, size, 'sample', [mode],
                                  samples=2 * cfg.eval.samples))
        # ---- 图6：每个规模前 9 个实例的全部采样解 ----
        if exp.fig6:
            k = cfg.eval.fig6_instances
            add(eval_task(cfg, 'ethc', size, MAIN_LAMBDA, main_seed, size, 'sample', ['soft', 'hard'],
                          num=k, keep=k, tag='fig6'))
        # ---- 冒烟阶段：评估计时探针（G 跑满 10,000 个实例，S 按论文口径测少量实例的每实例耗时） ----
        probes = cfg.get('eval_probes')
        if probes:
            for variant, modes in (('ethc', ['soft', 'hard']), ('tpa', ['soft']), ('mvmoe', ['hard'])):
                add(eval_task(cfg, variant, size, MAIN_LAMBDA, main_seed, size, 'greedy', modes,
                              num=probes.greedy_num, tag='probe', timing=True))
            for t in probes.test_sizes:
                add(eval_task(cfg, 'ethc', size, MAIN_LAMBDA, main_seed, t, 'sample', ['soft', 'hard'],
                              samples=probes.samples, num=probes.sample_num, tag='probe', timing=True))

    # ---- 元启发式：表2（λ=3）用 OR-Tools、TS、BA；图3 的 λ 扫描用 OR-Tools ----
    if exp.get('solvers', True):
        for size in sizes:
            for method in cfg.solvers.methods:
                add(solve_task(cfg, method, size, MAIN_LAMBDA))
            if exp.fig3:
                for lmbda in lambdas:
                    add(solve_task(cfg, 'ortools', size, lmbda))

    # ---- 计时评估（表2的时间列）放在其他全部训练、评估与求解任务之后独占运行，此时机器空闲，也不会让其他卡等待 ----
    others = [tid for tid, t in tasks.items() if t.stage in ('calibrate', 'solve', 'train', 'eval') and not t.exclusive]
    for t in tasks.values():
        if t.stage == 'timing':
            t.deps = t.deps + [d for d in others if d not in t.deps]

    # ---- 等时比较：OR-Tools 的单实例时限取 ETHC-VRC(S) 的实测每实例耗时（λ=3，表2的三个规模） ----
    if exp.get('solvers', True) and exp.table2 and cfg.solvers.get('equal_time', True):
        for size in sizes:
            if sample_timing_id(cfg, size) in tasks:
                add(solve_task(cfg, 'ortools', size, MAIN_LAMBDA, equal_time=True))

    # ---- 汇总：表、图与正文数字 ----
    deps = [tid for tid, t in tasks.items() if t.stage in ('eval', 'solve', 'timing', 'equal_time')]
    # 汇总只在比全部依赖结果都新时才算完成，新增或重跑的结果会触发重新汇总
    add(Task(id='analysis', stage='analysis', resource='cpu',
             cmd=[PY, '-m', 'analysis.report', '--config', cfg.config_path],
             marker=os.path.join(cfg.paths.results, 'paper_numbers.json'), deps=deps,
             meta={'dep_markers': [tasks[d].marker for d in deps]}))
    return tasks


def resolve_cmd(cfg, task):
    """把求解任务中的时限占位符替换为标定结果或 ETHC-VRC(S) 的实测耗时（这些结果出来后才能确定）"""
    out = []
    for a in task.cmd:
        if isinstance(a, str) and a.startswith('{TIME_LIMIT_'):
            a = '{:g}'.format(time_limit(cfg, int(a[len('{TIME_LIMIT_'):-1])))
        elif isinstance(a, str) and a.startswith('{EQUAL_TIME_'):
            a = '{:.4f}'.format(equal_time_limit(cfg, int(a[len('{EQUAL_TIME_'):-1])))
        out.append(a)
    return out
