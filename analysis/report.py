"""
汇总论文的全部数据：表2（CSV 与 LaTeX）、图3–6（PDF）以及 paper_numbers.json（正文引用的数字及其来源文件）。
所有数字都从 run_all.py 产生的结果文件计算，口径与论文第5.1.1节一致：
- Gap：相对本表各规模的最优值；
- 配对比较：逐实例差值的均值相对后一方法的平均成本，配对 t 分布的 95% 置信区间，双侧 Wilcoxon 符号秩检验；
- 消融：三个种子上逐实例相对差的均值、按测试实例的 95% 置信区间与种子间标准差。

用法：uv run python -m analysis.report --config configs/paper.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from experiments import load_config, eval_out, solve_out, run_dir, MAIN_LAMBDA, fmt_lambda  # noqa: E402

# ---- 图表样式：纸面印刷，白底；每个方法在所有图中颜色固定，另以标记形状区分（色觉安全的第二编码） ----
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STYLE = {
    'ETHC-VRC(G)': dict(color='#2a78d6', marker='s'),
    'ETHC-VRC(S)': dict(color='#eb6834', marker='o'),
    'TPA-Reward': dict(color='#1baf7a', marker='^'),
    'MVMoE': dict(color='#eda100', marker='D'),
    'OR-Tools': dict(color='#e87ba4', marker='v'),
    'ETHC-VRC': dict(color='#2a78d6', marker=None),
    'w/o ET': dict(color='#008300', marker=None),
}
plt.rcParams.update({
    # Linux 服务器上通常没有 Times New Roman，用字宽相同的 Liberation Serif（apt 包 fonts-liberation）代替
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'font.size': 11,
    'mathtext.fontset': 'stix',  # 公式符号（如 λ）用与 Times 风格一致的 STIX 字体，而非默认的 DejaVu Sans
    'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'grid.linestyle': '-', 'legend.frameon': False, 'legend.labelcolor': INK, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})
LINE = dict(linewidth=1.5, markersize=6, markeredgecolor='white', markeredgewidth=0.8, solid_capstyle='round')


def _json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class Results(object):
    """按实验配置读取评估与求解结果，缺失的结果返回 None（汇总时记为缺失）"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sources = set()
        self.missing = set()

    def _load(self, base):
        if not os.path.isfile(base + '.json'):
            self.missing.add(base)
            return None
        self.sources.add(base)
        return _json(base + '.json'), np.load(base + '.npz')

    def eval(self, variant, size, lmbda, seed, test_size, strategy, modes, samples=None, tag=None):
        if strategy == 'sample' and samples is None:
            samples = self.cfg.eval.samples
        return self._load(eval_out(self.cfg, variant, size, lmbda, seed, test_size, strategy, modes, samples, tag))

    def solve(self, method, size, lmbda):
        return self._load(solve_out(self.cfg, method, size, lmbda))


def paired(a, b):
    """
    a 相对 b：均值差/ b 的均值（%），配对 t 的 95% 置信区间（%），双侧 Wilcoxon 的 p 值。
    所有结果都从同一测试文件的第一个实例开始，实例数不同时取二者共同的前 n 个实例。
    """
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)
    d = a - b
    n = len(d)
    ref = float(np.mean(b))
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    p = float(stats.wilcoxon(a, b).pvalue) if n > 1 and np.any(d != 0) else 1.0
    return {'rel_mean_pct': 100 * d.mean() / ref, 'ci_low_pct': 100 * (d.mean() - half) / ref,
            'ci_high_pct': 100 * (d.mean() + half) / ref, 'ci_half_width_pp': 100 * half / ref,
            'p_wilcoxon': p, 'n': n}


def fmt_time(seconds):
    if seconds is None:
        return '--'
    if seconds < 60:
        return '{:.2f}s'.format(seconds)
    if seconds < 3600:
        return '{:.0f}m'.format(seconds / 60)
    return '{:.1f}h'.format(seconds / 3600)


def metrics_of(summary, key='best'):
    s = summary[key] if key in summary else summary['per_mode'][key]
    return {
        'cost': s['cost'], 'length': s['length'], 'feasibility_rate': s['feasibility_rate'],
        'late_instance_rate': s['late_instance_rate'],
        'avg_lateness_per_late_customer_min': 60 * s['avg_lateness_per_late_customer_h'],
        'vehicles': s['vehicles'],
    }


def full_metrics(entry):
    """输出解的指标、最多用车数；ETHC-VRC 另给每种掩码模式单独的指标与硬掩码解被选中的比例，
    元启发式另给车队上限（12 辆）被用满的实例比例与补加车辆的实例数（检验车队上限不起限制作用）"""
    summary, arrays = entry
    m = dict(metrics_of(summary), max_vehicles=int(arrays['best_vehicles'].max()))
    if len(summary.get('per_mode', {})) > 1:
        m['per_mode'] = {mode: metrics_of(summary, mode) for mode in summary['per_mode']}
        m['hard_selected_share'] = summary['mode_share'].get('hard')
    if 'max_routes' in summary:
        m['fleet_cap_share'] = float((arrays['best_vehicles'] >= summary['max_routes']).mean())
        m['extra_vehicle_instances'] = summary.get('extra_vehicle_instances', 0)
    return m


# ---------------------------------------------------------------- 表2与正文统计
# 最后一行是等时比较：OR-Tools 的单实例时限取 ETHC-VRC(S) 的实测每实例耗时
TABLE2_ROWS = ('ETHC-VRC(G)', 'ETHC-VRC(S)', 'TPA-Reward(G)', 'MVMoE(G)', 'OR-Tools', 'TS', 'BA', 'OR-Tools (equal time)')


def table2(res, out_dir):
    cfg = res.cfg
    seed = cfg.seeds[0]
    numbers, per_instance = {}, {}
    for size in cfg.sizes:
        entries = {
            'ETHC-VRC(G)': res.eval('ethc', size, MAIN_LAMBDA, seed, size, 'greedy', ['soft', 'hard']),
            'ETHC-VRC(S)': res.eval('ethc', size, MAIN_LAMBDA, seed, size, 'sample', ['soft', 'hard']),
            'TPA-Reward(G)': res.eval('tpa', size, MAIN_LAMBDA, seed, size, 'greedy', ['soft']),
            'MVMoE(G)': res.eval('mvmoe', size, MAIN_LAMBDA, seed, size, 'greedy', ['hard']),
            'OR-Tools': res.solve('ortools', size, MAIN_LAMBDA),
            'TS': res.solve('ts', size, MAIN_LAMBDA),
            'BA': res.solve('ba', size, MAIN_LAMBDA),
            'OR-Tools (equal time)': res.solve('ortools_eq', size, MAIN_LAMBDA),
        }
        for row, e in entries.items():
            if e is None:
                continue
            summary, arrays = e
            numbers.setdefault(row, {})[size] = dict(full_metrics(e), time_s=summary['time_s']['total'])
            if 'time_limit_s' in summary:  # 元启发式：单实例时限与并行进程数
                numbers[row][size].update(time_limit_s=summary['time_limit_s'], workers=summary['workers'])
            per_instance[(row, size)] = arrays['best_cost']
        g = entries['ETHC-VRC(G)']
        if g is not None:
            gs = g[0]
            numbers['ETHC-VRC(G)'][size].update({
                'soft_only_cost': gs['per_mode']['soft']['cost'], 'hard_only_cost': gs['per_mode']['hard']['cost'],
                'hard_selected_share': gs['mode_share']['hard']})
        s = entries['ETHC-VRC(S)']
        if s is not None:
            numbers['ETHC-VRC(S)'][size].update({
                'time_per_instance_s': s[0]['time_s']['total'] / s[0]['n'],
                'hard_selected_share': s[0]['mode_share']['hard']})
        if g is not None:
            numbers['ETHC-VRC(G)'][size]['time_per_instance_s'] = g[0]['time_s']['total'] / g[0]['n']
    # Gap 相对各规模的最优值
    for size in cfg.sizes:
        best = min((numbers[r][size]['cost'] for r in TABLE2_ROWS if size in numbers.get(r, {})), default=None)
        for r in TABLE2_ROWS:
            if size in numbers.get(r, {}):
                numbers[r][size]['gap_pct'] = 100 * (numbers[r][size]['cost'] - best) / best
    # 配对比较与派生数字
    derived = {}
    for size in cfg.sizes:
        d = {}
        for base_name in ('TPA-Reward(G)', 'MVMoE(G)'):
            if ('ETHC-VRC(G)', size) in per_instance and (base_name, size) in per_instance:
                d['G_vs_' + base_name] = paired(per_instance[('ETHC-VRC(G)', size)], per_instance[(base_name, size)])
        for base_name in ('OR-Tools', 'TS', 'BA', 'OR-Tools (equal time)'):
            if ('ETHC-VRC(S)', size) in per_instance and (base_name, size) in per_instance:
                d['S_vs_' + base_name] = paired(per_instance[('ETHC-VRC(S)', size)], per_instance[(base_name, size)])
            if ('ETHC-VRC(G)', size) in per_instance and (base_name, size) in per_instance:
                d['G_vs_' + base_name] = paired(per_instance[('ETHC-VRC(G)', size)], per_instance[(base_name, size)])
        if size in numbers.get('ETHC-VRC(G)', {}) and size in numbers.get('ETHC-VRC(S)', {}):
            g, s = numbers['ETHC-VRC(G)'][size]['cost'], numbers['ETHC-VRC(S)'][size]['cost']
            d['sampling_gain_pct'] = 100 * (g - s) / g
        derived[size] = d

    # CSV 与 LaTeX
    rows = []
    for r in TABLE2_ROWS:
        row = {'Method': r}
        for size in cfg.sizes:
            v = numbers.get(r, {}).get(size)
            row['n{}_obj'.format(size)] = None if v is None else round(v['cost'], 4)
            row['n{}_gap_pct'.format(size)] = None if v is None else round(v['gap_pct'], 2)
            row['n{}_time'.format(size)] = None if v is None else fmt_time(v['time_s'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'table2.csv'), index=False)
    tex = []
    for r in TABLE2_ROWS:
        cells = [r]
        for size in cfg.sizes:
            v = numbers.get(r, {}).get(size)
            if v is None:
                cells += ['--', '--', '--']
                continue
            best = abs(v['gap_pct']) < 1e-9
            obj, gap = '{:.2f}'.format(v['cost']), '{:.2f}\\%'.format(v['gap_pct'])
            cells += ['\\textbf{%s}' % obj if best else obj, '\\textbf{%s}' % gap if best else gap, fmt_time(v['time_s'])]
        tex.append(' & '.join(cells) + ' \\\\')
    with open(os.path.join(out_dir, 'table2.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    return {'rows': numbers, 'derived': derived}


# ---------------------------------------------------------------- 图3：λ 敏感性
def figure3(res, out_dir):
    cfg = res.cfg
    seed = cfg.seeds[0]
    numbers = {}
    for size in cfg.sizes:
        series = {'ETHC-VRC(G)': [], 'ETHC-VRC(S)': [], 'TPA-Reward': [], 'OR-Tools': []}
        details = {name: {} for name in series}  # 每个 λ 的路径长度、迟到与车辆数（混合成本之外的分项）
        for lmbda in cfg.lambdas:
            g = res.eval('ethc', size, lmbda, seed, size, 'greedy', ['soft', 'hard'])
            s = res.eval('ethc', size, lmbda, seed, size, 'sample', ['soft', 'hard'])
            t = res.eval('tpa', size, lmbda, seed, size, 'greedy', ['soft'])
            o = res.solve('ortools', size, lmbda)
            for name, e in (('ETHC-VRC(G)', g), ('ETHC-VRC(S)', s), ('TPA-Reward', t), ('OR-Tools', o)):
                series[name].append(None if e is None else e[0]['best']['cost'])
                if e is not None:
                    details[name][fmt_lambda(lmbda)] = full_metrics(e)
        numbers[size] = {'cost': {name: dict(zip(map(fmt_lambda, cfg.lambdas), vals)) for name, vals in series.items()},
                         'metrics': details}
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for name, vals in series.items():
            xs = [l for l, v in zip(cfg.lambdas, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if ys:
                ax.plot(xs, ys, label=name, **STYLE[name], **LINE)
        ax.set_xlabel(r'Penalty scaling factor $\lambda$')
        ax.set_ylabel('Hybrid cost')
        ax.set_xticks(list(cfg.lambdas))
        ax.legend(loc='best')
        fig.savefig(os.path.join(out_dir, 'Penalty{}.pdf'.format(size)))
        plt.close(fig)
    return numbers


# ---------------------------------------------------------------- 图4：跨规模泛化
def figure4(res, out_dir):
    cfg = res.cfg
    seed = cfg.seeds[0]
    numbers = {}
    for size in cfg.sizes:
        series = {'ETHC-VRC(G)': [], 'ETHC-VRC(S)': [], 'TPA-Reward': [], 'MVMoE': []}
        details = {name: {} for name in series}
        paired_g = {}  # ETHC-VRC(G) 对两个贪心基线的逐实例配对比较（每个训练—测试组合）
        for t in cfg.test_sizes:
            entries = dict((('ETHC-VRC(G)', res.eval('ethc', size, MAIN_LAMBDA, seed, t, 'greedy', ['soft', 'hard'])),
                            ('ETHC-VRC(S)', res.eval('ethc', size, MAIN_LAMBDA, seed, t, 'sample', ['soft', 'hard'])),
                            ('TPA-Reward', res.eval('tpa', size, MAIN_LAMBDA, seed, t, 'greedy', ['soft'])),
                            ('MVMoE', res.eval('mvmoe', size, MAIN_LAMBDA, seed, t, 'greedy', ['hard']))))
            for name, e in entries.items():
                series[name].append(None if e is None else e[0]['best']['cost'])
                if e is not None:
                    details[name][str(t)] = full_metrics(e)
            if entries['ETHC-VRC(G)'] is not None:
                paired_g[str(t)] = {base: paired(entries['ETHC-VRC(G)'][1]['best_cost'], entries[base][1]['best_cost'])
                                    for base in ('TPA-Reward', 'MVMoE') if entries[base] is not None}
        numbers[size] = {'cost': {name: dict(zip(map(str, cfg.test_sizes), vals)) for name, vals in series.items()},
                         'metrics': details, 'G_vs_baselines': paired_g}
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for name, vals in series.items():
            xs = [t for t, v in zip(cfg.test_sizes, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if ys:
                ax.plot(xs, ys, label=name, **STYLE[name], **LINE)
        ax.set_xlabel('Test problem size (nodes)')
        ax.set_ylabel('Hybrid cost')
        ax.set_xticks(list(cfg.test_sizes))
        ax.legend(loc='best')
        fig.savefig(os.path.join(out_dir, 'Generalization{}.pdf'.format(size)))
        plt.close(fig)
    return numbers


# ---------------------------------------------------------------- 图5与消融
def _train_log(cfg, variant, size, seed):
    path = os.path.join(run_dir(cfg, variant, size, MAIN_LAMBDA, seed), 'train_log.csv')
    if not os.path.isfile(path):
        return None
    # 续训会删去中断后写入的行（run.py）；这里再按步数去重（保留重新训练写入的行），保证曲线按步数单调
    log = pd.read_csv(path)
    return log.drop_duplicates('step', keep='last').sort_values('step').reset_index(drop=True)


def figure5_and_ablation(res, out_dir):
    cfg = res.cfg
    numbers = {}
    steps_per_epoch = cfg.train.epoch_size // cfg.train.batch_size
    window = max(1, steps_per_epoch // cfg.train.log_step)  # 训练曲线按一个 epoch 的滑动平均显示
    for size in cfg.sizes:
        n = {}
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for variant, name in (('ethc', 'ETHC-VRC'), ('ethc_woet', 'w/o ET')):
            logs = [_train_log(cfg, variant, size, seed) for seed in cfg.seeds]
            logs = [l for l in logs if l is not None and len(l)]
            if not logs:
                continue
            k = min(len(l) for l in logs)
            step = logs[0]['step'].values[:k]
            curves = np.stack([l['cost'].rolling(window, min_periods=1).mean().values[:k] for l in logs])
            mean, std = curves.mean(0), curves.std(0, ddof=1) if len(logs) > 1 else np.zeros(k)
            ax.fill_between(step, mean - std, mean + std, color=STYLE[name]['color'], alpha=0.18, linewidth=0)
            ax.plot(step, mean, label=name, color=STYLE[name]['color'], linewidth=1.5)
            final = [float(l['cost'].values[-window:].mean()) for l in logs]
            n[variant] = {'final_train_cost_per_seed': final,
                          'final_train_cost_std_across_seeds': float(np.std(final, ddof=1)) if len(final) > 1 else None,
                          'seeds': len(logs)}
        ax.set_xlabel('Training step')
        ax.set_ylabel('Hybrid cost')
        ax.legend(loc='upper right')
        fig.savefig(os.path.join(out_dir, 'train{}.pdf'.format(size)))
        plt.close(fig)

        # w/o ET：三个种子上的逐实例相对差
        for strategy in ('sample', 'greedy'):
            full, woet = [], []
            for seed in cfg.seeds:
                a = res.eval('ethc', size, MAIN_LAMBDA, seed, size, strategy, ['soft', 'hard'])
                b = res.eval('ethc_woet', size, MAIN_LAMBDA, seed, size, strategy, ['soft', 'hard'])
                if a is not None and b is not None:
                    full.append(a[1]['best_cost'])
                    woet.append(b[1]['best_cost'])
            if full:
                full, woet = np.stack(full), np.stack(woet)
                rel_per_seed = 100 * (woet.mean(1) - full.mean(1)) / full.mean(1)
                d = (woet - full).mean(0)
                half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                ref = full.mean()
                n['woET_vs_full_' + strategy] = {
                    'rel_mean_pct': float(rel_per_seed.mean()),
                    'ci_low_pct': float(100 * (d.mean() - half) / ref), 'ci_high_pct': float(100 * (d.mean() + half) / ref),
                    'std_across_seeds_pp': float(rel_per_seed.std(ddof=1)) if len(rel_per_seed) > 1 else None,
                    'rel_per_seed_pct': rel_per_seed.tolist(), 'seeds': len(rel_per_seed)}
        # w/o HC：同一权重下单模式采样 2×samples 个解
        for mode in ('soft', 'hard'):
            rel = []
            for seed in cfg.seeds:
                hybrid = res.eval('ethc', size, MAIN_LAMBDA, seed, size, 'sample', ['soft', 'hard'])
                single = res.eval('ethc', size, MAIN_LAMBDA, seed, size, 'sample', [mode], samples=2 * cfg.eval.samples)
                if hybrid is not None and single is not None:
                    rel.append(paired(hybrid[1]['best_cost'], single[1]['best_cost']))
            if rel:
                vals = np.array([r['rel_mean_pct'] for r in rel])
                n['hybrid_vs_' + mode + '_only'] = {'rel_mean_pct': float(vals.mean()),
                                                     'std_across_seeds_pp': float(vals.std(ddof=1)) if len(vals) > 1 else None,
                                                     'per_seed': rel}
        numbers[size] = n
    return numbers


# ---------------------------------------------------------------- 图6：采样解分布
def figure6(res, out_dir):
    cfg = res.cfg
    numbers = {}
    k = cfg.eval.fig6_instances
    for size in cfg.sizes:
        e = res.eval('ethc', size, MAIN_LAMBDA, cfg.seeds[0], size, 'sample', ['soft', 'hard'], tag='fig6')
        if e is None:
            continue
        arrays = e[1]
        data = np.concatenate((arrays['soft_kept_costs'], arrays['hard_kept_costs']), 1)  # (k, 2×samples)
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        color = STYLE['ETHC-VRC(S)']['color']
        ax.boxplot(list(data), widths=0.55, showmeans=True, whis=1.5, patch_artist=True,
                   boxprops=dict(facecolor=color + '40', edgecolor=color, linewidth=1.0),
                   medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK_2, linewidth=0.9),
                   capprops=dict(color=INK_2, linewidth=0.9),
                   meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor=INK, markersize=4),
                   flierprops=dict(marker='o', markersize=2.5, markerfacecolor=color, markeredgecolor='none', alpha=0.6))
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([str(i + 1) for i in range(len(data))])
        ax.set_xlabel('Instance')
        ax.set_ylabel('Hybrid cost')
        ax.grid(axis='x', visible=False)
        fig.savefig(os.path.join(out_dir, 'Sample{}.pdf'.format(size)))
        plt.close(fig)
        med = np.median(data, 1)
        q1, q3 = np.percentile(data, 25, 1), np.percentile(data, 75, 1)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)  # 与箱线图的须一致（whis=1.5），须外的点为离群点
        numbers[size] = {'instances': int(len(data)), 'solutions_per_instance': int(data.shape[1]),
                         'median_per_instance': med.tolist(), 'min_per_instance': data.min(1).tolist(),
                         'q1_per_instance': q1.tolist(), 'q3_per_instance': q3.tolist(),
                         'max_per_instance': data.max(1).tolist(), 'mean_per_instance': data.mean(1).tolist(),
                         'outliers_low_per_instance': (data < low[:, None]).sum(1).tolist(),
                         'outliers_high_per_instance': (data > high[:, None]).sum(1).tolist(),
                         'median_per_mode': {mode: np.median(arrays[mode + '_kept_costs'], 1).tolist()
                                             for mode in ('soft', 'hard')}}
    return numbers


def main():
    parser = argparse.ArgumentParser(description="Build Table 2, Figures 3-6 and paper_numbers.json")
    parser.add_argument('--config', default='configs/paper.yaml')
    opts = parser.parse_args()
    cfg = load_config(opts.config)
    res = Results(cfg)
    out_dir = cfg.paths.results
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    numbers = {'config': opts.config}
    if cfg.experiments.table2:
        numbers['table2'] = table2(res, out_dir)
    if cfg.experiments.fig3:
        numbers['figure3'] = figure3(res, fig_dir)
    if cfg.experiments.fig4:
        numbers['figure4'] = figure4(res, fig_dir)
    if cfg.experiments.ablation:
        numbers['ablation'] = figure5_and_ablation(res, fig_dir)
    if cfg.experiments.fig6:
        numbers['figure6'] = figure6(res, fig_dir)
    calibration = cfg.solvers.get('calibration')
    if calibration and os.path.isfile(calibration):
        numbers['solver_time_limits'] = _json(calibration)
    environment = os.path.join(out_dir, 'environment.json')  # run_all.py 记录的各次运行的硬件与软件版本
    if os.path.isfile(environment):
        numbers['environment'] = _json(environment)
    numbers['sources'] = sorted(res.sources)
    numbers['missing'] = sorted(res.missing)
    from atomic_io import save_json
    save_json(numbers, os.path.join(out_dir, 'paper_numbers.json'), indent=1, default=float)  # 最后写：汇总完成的标记
    print('Wrote {} (missing results: {})'.format(os.path.join(out_dir, 'paper_numbers.json'), len(res.missing)))


if __name__ == "__main__":
    main()
