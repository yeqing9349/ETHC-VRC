"""
汇总论文的全部数据：表2、表3（CSV 与 LaTeX）、图3–6（PDF）以及 paper_numbers.json（正文引用的数字及其来源文件）。
所有数字都从 run_all.py 产生的结果文件计算，口径与论文第5.1.1节一致：
- Gap：相对本表各规模的最优值；
- 配对比较：逐实例差值的均值相对后一方法的平均成本，配对 t 分布的 95% 置信区间，双侧 Wilcoxon 符号秩检验；
- 同预算采样：TPA-Reward(S)、MVMoE(S) 每个实例采样 2×samples 个解，与 ETHC-VRC(S) 的候选数相同；
- 表2的 DRL 行：成本取模型训练完后运行的评估，时间取其他任务都完成后独占重跑的计时评估（任务名带 _timing）；
- 消融：逐实例差值的均值相对参照（w/o ET 为完整模型，逐实例选优为单一模式解码）的平均成本，三个种子取平均；
  w/o ET 另给按测试实例的 95% 置信区间与种子间标准差；表3为三个种子的平均成本。

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
from matplotlib.ticker import MaxNLocator  # noqa: E402

from experiments import load_config, eval_out, solve_out, run_dir, MAIN_LAMBDA, fmt_lambda, baseline_samples  # noqa: E402

# ---- 图表样式：沿用论文原图（OriginLab 绘制）的画法：Times 字体，只有左轴和下轴，刻度朝外，无网格，
#      图例在图内、带细边框，数据点之间用细线相连；各方法的颜色与标记取自原图的 PDF ----
INK = '#000000'
STYLE = {
    'ETHC-VRC(G)': dict(color='#37ad6b', marker='s'),
    'ETHC-VRC(S)': dict(color='#37ad6b', marker='s', markerfacecolor='white'),  # 原图为蓝色实心圆；改为与基线一致的同色空心标记
    'TPA-Reward(G)': dict(color='#cc9900', marker='^'),
    'TPA-Reward(S)': dict(color='#cc9900', marker='^', markerfacecolor='white'),  # 采样：同色空心标记
    'MVMoE(G)': dict(color='#b177de', marker='D'),  # 原图 4 为倒三角；图 3 中 OR-Tools 已用倒三角，改为菱形
    'MVMoE(S)': dict(color='#b177de', marker='D', markerfacecolor='white'),
    'OR-Tools': dict(color='#f14040', marker='v'),
    'ETHC-VRC': dict(color='#fb6501'),
    'w/o ET': dict(color='#00cbcc'),
}
BOX_FILL = '#669999'
plt.rcParams.update({
    # Linux 服务器上通常没有 Times New Roman，用字宽相同的 Liberation Serif（apt 包 fonts-liberation）代替
    # 字号：子图在论文中按 0.48\textwidth（MDPI 版心 394 pt，即约 189 pt）排版，图宽约 215–225 pt，
    # 缩放约 0.85 倍，10 pt 的刻度与图例文字印出来约 8.5 pt（审稿意见 14 要求加大坐标轴文字，方案定为不小于 8 pt）
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'mathtext.fontset': 'stix',  # 公式符号（如 λ）用与 Times 风格一致的 STIX 字体，而非默认的 DejaVu Sans
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'axes.linewidth': 0.9, 'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.color': INK, 'ytick.color': INK,
    'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': INK, 'legend.framealpha': 1.0,
    'legend.borderpad': 0.35, 'legend.labelspacing': 0.25, 'legend.handlelength': 1.5, 'legend.handletextpad': 0.4,
    'legend.borderaxespad': 0.4, 'legend.columnspacing': 0.8, 'patch.linewidth': 0.5,
    # 图框右侧和上方不留空白（裁边后图宽 ≈ 图框宽 + 左侧的纵轴文字），两栏图例才放得进图框
    'figure.subplot.right': 0.96, 'figure.subplot.top': 0.97,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'pdf.fonttype': 42,
})
LINE = dict(linewidth=0.8, markersize=5, markeredgewidth=0.8)
FIGSIZE = (3.1, 2.5)       # 子图（图5）；纵横比与原图（约 1.2:1）相近
FIGSIZE_LEGEND = (3.1, 3.1)  # 图例放在图框内顶部的子图（图3、图4、图6），为图例加高
FIG5_TOP = 1.15            # 图5纵轴上限 = 最终训练成本的 1.15 倍（与原图一样截去训练最初成本极高的一段）
FIG6_TOP = 1.5             # 图6在最大实例中位数的 1.5 倍处断轴，罚项极大的个别解画在断轴以上
# 基线的键名沿用 paper_numbers.json（不带后缀的是贪心解码），图中统一写出解码策略
LABEL = {'TPA-Reward': 'TPA-Reward(G)', 'MVMoE': 'MVMoE(G)'}


def _legend(ax, **kw):
    """原图样式的图例：细黑边框、白底、直角"""
    leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(0.5)
    return leg


def _legend_bottom(ax, leg, pad=0.04):
    """图例下边在图框中的相对高度减去 pad：数据画在它以下，不与图例重叠"""
    ax.figure.canvas.draw()
    return float(leg.get_window_extent().transformed(ax.transAxes.inverted()).y0) - pad


def _plot_lines(ax, xs, series, order):
    """按 order 画折线（缺失的点跳过），图例顺序即 order；本文方法画在基线之上"""
    for name in order:
        pts = [(x, v) for x, v in zip(xs, series.get(name, ())) if v is not None]
        if pts:
            ax.plot(*zip(*pts), label=name, zorder=2.2 if name.startswith('ETHC') else 2, **STYLE[name], **LINE)


def _decimals(values):
    """同一段刻度标签的小数位数：能把每个刻度值写准的最少位数（与原图一样，同一段内位数相同）"""
    for d in range(5):
        if all(abs(v - round(v, d)) < 1e-9 * max(1.0, abs(v)) for v in values):
            return d
    return 4


def _piecewise(xs, us):
    """分段线性映射 xs → us（端段线性外推），返回 (正向, 反向)，用于同一图框内的断轴"""
    xs, us = np.asarray(xs, float), np.asarray(us, float)

    def make(a, b):
        def f(v):
            v = np.asarray(v, float)
            out = np.interp(v, a, b)
            out = np.where(v < a[0], b[0] + (v - a[0]) * (b[1] - b[0]) / (a[1] - a[0]), out)
            return np.where(v > a[-1], b[-1] + (v - a[-1]) * (b[-1] - b[-2]) / (a[-1] - a[-2]), out)
        return f
    return make(xs, us), make(us, xs)


def _ticks(lo, hi, nbins):
    """[lo, hi] 内的整齐刻度（间隔为 1、2、2.5、5 乘以 10 的幂）"""
    return [t for t in MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo - 1e-9 <= t <= hi + 1e-9]


def _set_yticks(ax, lo, hi, nbins=5):
    """不断轴的纵轴：刻度只取数据所在的 [lo, hi]（其上是图例所在的空白），小数位数统一"""
    ticks = _ticks(lo, hi, nbins)
    ax.set_yticks(ticks, ['{:.{}f}'.format(t, _decimals(ticks)) for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _broken_y(ax, segments, nbins, tick_top=None):
    """
    同一图框内断开纵轴（Origin 的断轴）。segments 自下而上给出各段 (y_lo, y_hi, u_lo, u_hi)：数值区间 [y_lo, y_hi]
    画在图框高度的 [u_lo, u_hi] 处（u 为相对高度，第一段从 0 开始，最后一段到 1 为止）。相邻两段 u 不相接时，
    省略二者之间的数值；相接（上一段的 y_hi 等于下一段的 y_lo）时不省略，只改变断轴以上的刻度比例。
    左轴在断开处画断轴符号，各段分别取刻度（nbins 为各段的最多刻度数）；最后一段的刻度只取到 tick_top
    （数据的上限，其上是图例所在的空白）。
    """
    ys, us = [], []
    for y_lo, y_hi, u_lo, u_hi in segments:
        if not ys or y_lo > ys[-1]:
            ys.append(y_lo)
            us.append(u_lo)
        ys.append(y_hi)
        us.append(u_hi)
    fwd, inv = _piecewise(ys, us)
    ax.set_yscale('function', functions=(fwd, inv))
    ax.set_ylim(ys[0], ys[-1])
    h = 0.016  # 两段相接时，断轴符号画在接点上下各 h 处
    breaks = [(hi - h, lo + h) if lo <= hi else (hi, lo) for (_, _, _, hi), (_, _, lo, _) in zip(segments, segments[1:])]
    ticks, labels = [], []
    for i, ((y_lo, y_hi, _, _), nb) in enumerate(zip(segments, nbins)):
        if i == len(segments) - 1 and tick_top is not None:
            y_hi = tick_top
        seg = [t for t in _ticks(y_lo, y_hi, nb)
               if not any(a - 0.035 < float(fwd(t)) < b + 0.035 for a, b in breaks)]  # 断轴符号旁不放刻度
        ticks += seg
        labels += ['{:.{}f}'.format(t, _decimals(seg)) for t in seg]
    ax.set_yticks(ticks, labels)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # 左轴分段画出，断开处两端各画一条斜线
    ax.spines['left'].set_visible(False)
    kw = dict(transform=ax.transAxes, color=INK, linewidth=plt.rcParams['axes.linewidth'], clip_on=False, zorder=3,
              solid_capstyle='butt')
    ends = [0.0] + [u for b in breaks for u in b] + [1.0]
    for a, b in zip(ends[::2], ends[1::2]):
        ax.plot([0, 0], [a, b], **kw)
    for a, b in breaks:
        for u in (a, b):
            ax.plot([-0.022, 0.022], [u - 0.012, u + 0.012], **kw)


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
        name = base.replace(os.sep, '/')  # 记录的路径一律用 /，在 Windows 上汇总时也与服务器上生成的相同
        if not os.path.isfile(base + '.json'):
            self.missing.add(name)
            return None
        self.sources.add(name)
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
# TPA-Reward(S)、MVMoE(S) 是同预算比较（每个实例的候选数与 ETHC-VRC(S) 相同）；
# 最后一行是等时比较：OR-Tools 的单实例时限取 ETHC-VRC(S) 的实测每实例耗时
TABLE2_ROWS = ('ETHC-VRC(G)', 'ETHC-VRC(S)', 'TPA-Reward(G)', 'MVMoE(G)', 'TPA-Reward(S)', 'MVMoE(S)', 'OR-Tools', 'TS',
               'BA', 'OR-Tools (equal time)')


def table2(res, out_dir):
    cfg = res.cfg
    seed = cfg.seeds[0]
    numbers, per_instance = {}, {}
    for size in cfg.sizes:
        drl = {  # 表2的 DRL 行：方法、解码策略、掩码模式、每个实例的样本数
            'ETHC-VRC(G)': ('ethc', 'greedy', ['soft', 'hard'], None),
            'ETHC-VRC(S)': ('ethc', 'sample', ['soft', 'hard'], None),
            'TPA-Reward(G)': ('tpa', 'greedy', ['soft'], None),
            'MVMoE(G)': ('mvmoe', 'greedy', ['hard'], None),
        }
        if cfg.experiments.get('baseline_sampling', True):
            drl['TPA-Reward(S)'] = ('tpa', 'sample', ['soft'], baseline_samples(cfg))
            drl['MVMoE(S)'] = ('mvmoe', 'sample', ['hard'], baseline_samples(cfg))
        # 成本取模型训练完后运行的评估；时间取其他任务都完成后独占重跑的计时评估（tag='timing'）
        entries = {row: res.eval(v, size, MAIN_LAMBDA, seed, size, strategy, modes, samples=k)
                   for row, (v, strategy, modes, k) in drl.items()}
        timings = {row: res.eval(v, size, MAIN_LAMBDA, seed, size, strategy, modes, samples=k, tag='timing')
                   for row, (v, strategy, modes, k) in drl.items()}
        entries.update({
            'OR-Tools': res.solve('ortools', size, MAIN_LAMBDA),
            'TS': res.solve('ts', size, MAIN_LAMBDA),
            'BA': res.solve('ba', size, MAIN_LAMBDA),
            'OR-Tools (equal time)': res.solve('ortools_eq', size, MAIN_LAMBDA),
        })
        for row, e in entries.items():
            if e is None:
                continue
            summary, arrays = e
            if row in drl:  # DRL 方法：每实例耗时为整个测试集的总耗时 / 实例数（贪心按批处理摊销）
                t = timings[row][0] if timings[row] is not None else None
                total = None if t is None else t['time_s']['total']
                numbers.setdefault(row, {})[size] = dict(full_metrics(e), time_s=total,
                                                         time_per_instance_s=None if t is None else total / t['n'])
                if t is not None:
                    numbers[row][size]['timing_run_cost'] = t['best']['cost']  # 计时重跑的成本，应与上面的相同
            else:  # 元启发式：单实例时限与并行进程数
                numbers.setdefault(row, {})[size] = dict(full_metrics(e), time_s=summary['time_s']['total'],
                                                         time_limit_s=summary['time_limit_s'], workers=summary['workers'])
            per_instance[(row, size)] = arrays['best_cost']
        g = entries['ETHC-VRC(G)']
        if g is not None:
            gs = g[0]
            numbers['ETHC-VRC(G)'][size].update({
                'soft_only_cost': gs['per_mode']['soft']['cost'], 'hard_only_cost': gs['per_mode']['hard']['cost'],
                'hard_selected_share': gs['mode_share']['hard']})
        s = entries['ETHC-VRC(S)']
        if s is not None:
            numbers['ETHC-VRC(S)'][size]['hard_selected_share'] = s[0]['mode_share']['hard']
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
        for base_name in ('TPA-Reward(S)', 'MVMoE(S)'):  # 同预算采样
            if ('ETHC-VRC(S)', size) in per_instance and (base_name, size) in per_instance:
                d['S_vs_' + base_name] = paired(per_instance[('ETHC-VRC(S)', size)], per_instance[(base_name, size)])
        for base_name in ('OR-Tools', 'TS', 'BA', 'OR-Tools (equal time)'):
            if ('ETHC-VRC(S)', size) in per_instance and (base_name, size) in per_instance:
                d['S_vs_' + base_name] = paired(per_instance[('ETHC-VRC(S)', size)], per_instance[(base_name, size)])
            if ('ETHC-VRC(G)', size) in per_instance and (base_name, size) in per_instance:
                d['G_vs_' + base_name] = paired(per_instance[('ETHC-VRC(G)', size)], per_instance[(base_name, size)])
        if size in numbers.get('ETHC-VRC(G)', {}) and size in numbers.get('ETHC-VRC(S)', {}):
            g, s = numbers['ETHC-VRC(G)'][size]['cost'], numbers['ETHC-VRC(S)'][size]['cost']
            d['sampling_gain_pct'] = 100 * (g - s) / g
        for base in ('TPA-Reward', 'MVMoE'):  # 两个 DRL 基线采样相对各自贪心解码的降幅
            gb, sb = numbers.get(base + '(G)', {}).get(size), numbers.get(base + '(S)', {}).get(size)
            if gb is not None and sb is not None:
                d.setdefault('baseline_sampling_gain_pct', {})[base] = 100 * (gb['cost'] - sb['cost']) / gb['cost']
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
FIG3_ORDER = ('ETHC-VRC(G)', 'TPA-Reward(G)', 'MVMoE(G)', 'OR-Tools', 'ETHC-VRC(S)', 'TPA-Reward(S)', 'MVMoE(S)')


def figure3(res, out_dir):
    cfg = res.cfg
    seed, k = cfg.seeds[0], baseline_samples(cfg)
    numbers = {}
    for size in cfg.sizes:
        series = {'ETHC-VRC(G)': [], 'ETHC-VRC(S)': [], 'TPA-Reward': [], 'OR-Tools': [], 'TPA-Reward(S)': []}
        details = {name: {} for name in series}  # 每个 λ 的路径长度、迟到与车辆数（混合成本之外的分项）
        for lmbda in cfg.lambdas:
            entries = {'ETHC-VRC(G)': res.eval('ethc', size, lmbda, seed, size, 'greedy', ['soft', 'hard']),
                       'ETHC-VRC(S)': res.eval('ethc', size, lmbda, seed, size, 'sample', ['soft', 'hard']),
                       'TPA-Reward': res.eval('tpa', size, lmbda, seed, size, 'greedy', ['soft']),
                       'OR-Tools': res.solve('ortools', size, lmbda),
                       'TPA-Reward(S)': res.eval('tpa', size, lmbda, seed, size, 'sample', ['soft'], samples=k)}
            for name, e in entries.items():
                series[name].append(None if e is None else e[0]['best']['cost'])
                if e is not None:
                    details[name][fmt_lambda(lmbda)] = full_metrics(e)
        cost = {name: dict(zip(map(fmt_lambda, cfg.lambdas), vals)) for name, vals in series.items()}
        # MVMoE 训练和推理都用硬掩码，解从不迟到，混合成本不含罚项、与 λ 无关：只有 λ = 3 的模型，
        # 图中在每个 λ 处画同一个值（先检查确实没有迟到的实例）
        for name, strategy, samples in (('MVMoE(G)', 'greedy', None), ('MVMoE(S)', 'sample', k)):
            e = res.eval('mvmoe', size, MAIN_LAMBDA, seed, size, strategy, ['hard'], samples=samples)
            on_time = e is not None and e[0]['best']['late_instance_rate'] == 0
            series[name] = [e[0]['best']['cost'] if on_time else None] * len(cfg.lambdas)
            cost[name] = {fmt_lambda(MAIN_LAMBDA): None if e is None else e[0]['best']['cost']}
            details[name] = {} if e is None else {fmt_lambda(MAIN_LAMBDA): full_metrics(e)}
        numbers[size] = {'cost': cost, 'metrics': details}

        # 图例两栏放在图框内顶部（同一方法的 G、S 在同一行），数据画在图例以下。
        # TPA-Reward 的成本是其余方法的 1.2–2.8 倍，共用一段纵轴会把其余方法压成平线：
        # 同一图框内断轴，TPA-Reward 画在断轴以上，其余方法画在断轴以下
        shown = {LABEL.get(name, name): vals for name, vals in series.items()}
        fig, ax = plt.subplots(figsize=FIGSIZE_LEGEND)
        _plot_lines(ax, cfg.lambdas, shown, FIG3_ORDER)
        u_top = _legend_bottom(ax, _legend(ax, loc='upper center', ncol=2))
        low = [v for name in FIG3_ORDER if not name.startswith('TPA') for v in shown[name] if v is not None]
        high = [v for name in ('TPA-Reward(G)', 'TPA-Reward(S)') for v in shown[name] if v is not None]
        pad_lo = 0.12 * (max(low) - min(low))
        pad_hi = 0.15 * max(max(high) - min(high), 0.02 * max(high))
        y0, y1, y2, y_hi = min(low) - pad_lo, max(low) + pad_lo, min(high) - pad_hi, max(high) + pad_hi
        u1 = 0.6 * u_top  # 断轴以下占数据区的 60%
        u2 = u1 + 0.04
        y3 = y2 + (y_hi - y2) * (1 - u2) / (u_top - u2)  # 断轴以上的一段按同一比例延伸到图框顶部（图例所在处）
        if y2 > y1:
            _broken_y(ax, [(y0, y1, 0, u1), (y2, y3, u2, 1)], nbins=(4, 4), tick_top=y_hi)
            numbers[size]['axis_break'] = {'lower': [y0, y1], 'upper': [y2, y_hi]}
        else:  # 各方法的成本区间重叠，不断轴
            ax.set_ylim(y0, y0 + (y_hi - y0) / u_top)
            _set_yticks(ax, y0, y_hi)
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel(r'Penalty Scaling Factor $\lambda$')
        ax.set_ylabel('Hybrid Cost')
        fig.savefig(os.path.join(out_dir, 'Penalty{}.pdf'.format(size)))
        plt.close(fig)
    return numbers


# ---------------------------------------------------------------- 图4：跨规模泛化
FIG4_ORDER = ('ETHC-VRC(G)', 'TPA-Reward(G)', 'MVMoE(G)', 'ETHC-VRC(S)', 'TPA-Reward(S)', 'MVMoE(S)')


def figure4(res, out_dir):
    cfg = res.cfg
    seed, k = cfg.seeds[0], baseline_samples(cfg)
    numbers = {}
    for size in cfg.sizes:
        series = {'ETHC-VRC(G)': [], 'ETHC-VRC(S)': [], 'TPA-Reward': [], 'MVMoE': [], 'TPA-Reward(S)': [], 'MVMoE(S)': []}
        details = {name: {} for name in series}
        # ETHC-VRC(G) 对两个贪心基线、ETHC-VRC(S) 对两个同预算采样基线的逐实例配对比较（每个训练—测试组合）
        paired_g, paired_s = {}, {}
        for t in cfg.test_sizes:
            entries = {'ETHC-VRC(G)': res.eval('ethc', size, MAIN_LAMBDA, seed, t, 'greedy', ['soft', 'hard']),
                       'ETHC-VRC(S)': res.eval('ethc', size, MAIN_LAMBDA, seed, t, 'sample', ['soft', 'hard']),
                       'TPA-Reward': res.eval('tpa', size, MAIN_LAMBDA, seed, t, 'greedy', ['soft']),
                       'MVMoE': res.eval('mvmoe', size, MAIN_LAMBDA, seed, t, 'greedy', ['hard']),
                       'TPA-Reward(S)': res.eval('tpa', size, MAIN_LAMBDA, seed, t, 'sample', ['soft'], samples=k),
                       'MVMoE(S)': res.eval('mvmoe', size, MAIN_LAMBDA, seed, t, 'sample', ['hard'], samples=k)}
            for name, e in entries.items():
                series[name].append(None if e is None else e[0]['best']['cost'])
                if e is not None:
                    details[name][str(t)] = full_metrics(e)
            for ours, bases, out in (('ETHC-VRC(G)', ('TPA-Reward', 'MVMoE'), paired_g),
                                     ('ETHC-VRC(S)', ('TPA-Reward(S)', 'MVMoE(S)'), paired_s)):
                if entries[ours] is not None:
                    out[str(t)] = {base: paired(entries[ours][1]['best_cost'], entries[base][1]['best_cost'])
                                   for base in bases if entries[base] is not None}
        numbers[size] = {'cost': {name: dict(zip(map(str, cfg.test_sizes), vals)) for name, vals in series.items()},
                         'metrics': details, 'G_vs_baselines': paired_g, 'S_vs_baselines': paired_s}

        # 图例两栏放在图框内顶部，数据画在图例以下。TPA-Reward 随测试规模增长得快得多（测试 100 个节点时
        # 是其余方法的 2.1–2.4 倍），共用一个比例会把其余四条线挤在一起：同一图框内断轴，
        # 不省略数值，只把断轴以上的刻度比例压缩；断轴取在其余四种方法的最大成本之上
        shown = {LABEL.get(name, name): vals for name, vals in series.items()}
        fig, ax = plt.subplots(figsize=FIGSIZE_LEGEND)
        _plot_lines(ax, cfg.test_sizes, shown, FIG4_ORDER)
        u_top = _legend_bottom(ax, _legend(ax, loc='upper center', ncol=2))
        values = [v for vals in shown.values() for v in vals if v is not None]
        low = [v for name in FIG4_ORDER if not name.startswith('TPA') for v in shown[name] if v is not None]
        y0 = min(values) - 0.05 * (max(low) - min(values))
        y_b = max(low) + 0.04 * (max(low) - min(values))
        y_hi = max(values) + 0.04 * (max(values) - y_b)
        if max(values) > y_b:
            u_b = 0.62 * u_top  # 断轴以下占数据区的 62%
            y_top = y_b + (y_hi - y_b) * (1 - u_b) / (u_top - u_b)
            _broken_y(ax, [(y0, y_b, 0, u_b), (y_b, y_top, u_b, 1)], nbins=(5, 3), tick_top=y_hi)
            numbers[size]['axis_break'] = {'at': y_b}
        else:
            ax.set_ylim(y0, y0 + (y_b - y0) / u_top)
            _set_yticks(ax, y0, y_b)
        ax.set_xlim(12, 108)
        ax.set_xticks(range(20, 101, 20))
        ax.set_xlabel('Test Size $n$')
        ax.set_ylabel('Hybrid Cost')
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
    window = max(1, steps_per_epoch // cfg.train.log_step)  # 一个 epoch 的记录条数
    for size in cfg.sizes:
        n = {}
        fig, ax = plt.subplots(figsize=FIGSIZE)
        lows, finals = [], []  # 纵轴范围：各曲线 均值−标准差 的最小值、最终成本
        for variant, name in (('ethc', 'ETHC-VRC'), ('ethc_woet', 'w/o ET')):
            logs = [_train_log(cfg, variant, size, seed) for seed in cfg.seeds]
            logs = [l for l in logs if l is not None and len(l)]
            if not logs:
                continue
            k = min(len(l) for l in logs)
            step = logs[0]['step'].values[:k]
            # 每个种子的训练成本先按一个 epoch 做居中的滑动平均（逐批成本的波动远大于两条曲线之差），
            # 再取各种子的均值，阴影为种子间 ±1 个标准差；本文方法画在上层
            curves = np.stack([l['cost'].rolling(window, center=True, min_periods=1).mean().values[:k] for l in logs])
            mean, std = curves.mean(0), curves.std(0, ddof=1) if len(logs) > 1 else np.zeros(k)
            z = 2.2 if variant == 'ethc' else 2
            ax.fill_between(step / 1000, mean - std, mean + std, color=STYLE[name]['color'], alpha=0.2, linewidth=0,
                            zorder=z)
            ax.plot(step / 1000, mean, label=name, color=STYLE[name]['color'], linewidth=0.9, zorder=z)
            lows.append(float((mean - std).min()))
            finals.append(float(mean[-1]))
            final = [float(l['cost'].values[-window:].mean()) for l in logs]
            n[variant] = {'final_train_cost_per_seed': final,
                          'final_train_cost_std_across_seeds': float(np.std(final, ddof=1)) if len(final) > 1 else None,
                          'seeds': len(logs)}
        # 训练开始时成本高出最终值几十到上百倍，共用纵轴时收敛段贴在底部、看不出差别：
        # 与原图一样截去纵轴上方，只画到最终成本的 FIG5_TOP 倍，训练最初阶段在图的上方之外
        if finals:
            top = FIG5_TOP * max(finals)
            ax.set_ylim(min(lows) - 0.04 * (top - min(lows)), top)
            n['plot_top'] = top
        ax.set_xlim(-6, 256)
        ax.set_xticks(range(0, 251, 50))
        ax.set_xlabel(r'Training Step ($\times 10^3$)')
        ax.set_ylabel('Hybrid Cost')
        _legend(ax, loc='upper right')
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


# ---------------------------------------------------------------- 表3：消融
# 每格为三个种子的平均成本。w/o HC 用完整模型的权重只在硬掩码模式下解码：贪心取同一次评估中的硬掩码解，
# 采样单独运行 2×samples 个硬掩码样本（与 ETHC-VRC(S) 的候选数相同）。另给正文用的逐实例选优相对单一模式的变化
TABLE3_ROWS = ('ETHC-VRC', 'w/o ET', 'w/o HC')


def table3(res, out_dir):
    cfg = res.cfg
    numbers = {}
    for size in cfg.sizes:
        cost = {(row, key): [] for row in TABLE3_ROWS for key in ('G', 'S')}  # 每个种子的平均成本
        change = {}  # 逐实例选优相对单一模式的变化（%），每个种子一个值
        for seed in cfg.seeds:
            for strategy, key in (('greedy', 'G'), ('sample', 'S')):
                full = res.eval('ethc', size, MAIN_LAMBDA, seed, size, strategy, ['soft', 'hard'])
                woet = res.eval('ethc_woet', size, MAIN_LAMBDA, seed, size, strategy, ['soft', 'hard'])
                if strategy == 'greedy':  # 单一模式的解：贪心取同一次评估中该模式的解
                    single = {m: None if full is None else (full[0]['per_mode'][m]['cost'], full[1][m + '_cost'])
                              for m in ('soft', 'hard')}
                else:  # 采样单独运行，每个实例 2×samples 个样本
                    single = {}
                    for m in ('soft', 'hard'):
                        e = res.eval('ethc', size, MAIN_LAMBDA, seed, size, 'sample', [m], samples=2 * cfg.eval.samples)
                        single[m] = None if e is None else (e[0]['best']['cost'], e[1]['best_cost'])
                if full is not None:
                    cost[('ETHC-VRC', key)].append(full[0]['best']['cost'])
                    for m, s in single.items():
                        if s is not None:
                            change.setdefault('{}_vs_{}_only'.format(key, m), []).append(
                                paired(full[1]['best_cost'], s[1])['rel_mean_pct'])
                if woet is not None:
                    cost[('w/o ET', key)].append(woet[0]['best']['cost'])
                    if strategy == 'greedy':
                        change.setdefault('G_woET_vs_soft_only', []).append(
                            paired(woet[1]['best_cost'], woet[1]['soft_cost'])['rel_mean_pct'])
                if single['hard'] is not None:
                    cost[('w/o HC', key)].append(single['hard'][0])
        k = len(cfg.seeds)  # 所有种子都有结果才报告均值
        numbers[size] = {
            'cost': {row: {key: float(np.mean(cost[(row, key)])) if len(cost[(row, key)]) == k else None
                           for key in ('G', 'S')} for row in TABLE3_ROWS},
            'cost_per_seed': {row: {key: cost[(row, key)] for key in ('G', 'S')} for row in TABLE3_ROWS},
            'selection_change_pct': {name: float(np.mean(v)) if len(v) == k else None for name, v in change.items()},
        }

    # CSV 与 LaTeX（列：贪心 n=20/50/100，采样 n=20/50/100；每列最低的成本加粗）
    cols = [(key, size) for key in ('G', 'S') for size in cfg.sizes]
    best = {(key, size): min((numbers[size]['cost'][r][key] for r in TABLE3_ROWS
                              if numbers[size]['cost'][r][key] is not None), default=None) for key, size in cols}
    rows, tex = [], []
    for r in TABLE3_ROWS:
        row, cells = {'Method': r}, [r]
        for key, size in cols:
            v = numbers[size]['cost'][r][key]
            row['{}_n{}'.format(key, size)] = None if v is None else round(v, 4)
            text = '--' if v is None else '{:.3f}'.format(v)
            cells.append('\\textbf{%s}' % text if v is not None and v == best[(key, size)] else text)
        rows.append(row)
        tex.append(' & '.join(cells) + ' \\\\')
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'table3.csv'), index=False)
    with open(os.path.join(out_dir, 'table3.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
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
        # 原图的箱线图样式：箱体填色，中位线、须与端线为黑色，均值为空心小方块，离群点为黑色小菱形
        fig, ax = plt.subplots(figsize=FIGSIZE_LEGEND)
        thin = dict(color=INK, linewidth=0.6)
        mean_style = dict(marker='s', markerfacecolor='white', markeredgecolor=INK, markersize=3, markeredgewidth=0.6)
        flier_style = dict(marker='D', markerfacecolor=INK, markeredgecolor='none', markersize=2.4)
        ax.boxplot(list(data), widths=0.55, showmeans=True, whis=1.5, patch_artist=True,
                   boxprops=dict(facecolor=BOX_FILL, edgecolor=INK, linewidth=0.6),
                   medianprops=dict(color=INK, linewidth=0.8), whiskerprops=thin, capprops=thin,
                   meanprops=mean_style, flierprops=flier_style)
        handles = [matplotlib.patches.Patch(facecolor=BOX_FILL, edgecolor=INK, linewidth=0.6),
                   matplotlib.lines.Line2D([], [], marker='$\\mathrm{I}$', markersize=7, linestyle='none', color=INK),
                   matplotlib.lines.Line2D([], [], **thin),
                   matplotlib.lines.Line2D([], [], linestyle='none', **mean_style),
                   matplotlib.lines.Line2D([], [], linestyle='none', **dict(flier_style, markersize=3.5))]
        leg = _legend(ax, handles=handles, labels=['25%~75%', '1.5 IQR Range', 'Median Line', 'Mean', 'Outliers'],
                      loc='upper center', ncol=2)
        u_top = _legend_bottom(ax, leg)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([str(i + 1) for i in range(len(data))])
        ax.set_xlim(0.4, len(data) + 0.6)
        ax.set_xlabel('Instance')
        ax.set_ylabel('Hybrid Cost')
        # 个别采样解的罚项极大（混合成本可达中位数的 7 倍），共用一个比例会把箱体压扁：在最大实例中位数的
        # FIG6_TOP 倍处断轴，所有箱体和须都在断轴以下，超出的个别解画在断轴以上
        med = np.median(data, 1)
        y1 = FIG6_TOP * float(med.max())
        y0 = float(data.min()) - 0.04 * (y1 - float(data.min()))
        above = (data > y1).sum(1)
        if above.any():
            high = data[data > y1]
            pad = 0.08 * max(float(high.max() - high.min()), 0.1 * y1)
            y2, y_hi = float(high.min()) - pad, float(high.max()) + pad
            u1 = 0.7 * u_top  # 断轴以下占数据区的 70%
            u2 = u1 + 0.04
            y3 = y2 + (y_hi - y2) * (1 - u2) / (u_top - u2)
            _broken_y(ax, [(y0, y1, 0, u1), (max(y2, y1 + 1e-6), y3, u2, 1)], nbins=(5, 2), tick_top=y_hi)
        else:
            ax.set_ylim(y0, y0 + (y1 - y0) / u_top)
            _set_yticks(ax, y0, y1)
        fig.savefig(os.path.join(out_dir, 'Sample{}.pdf'.format(size)))
        plt.close(fig)
        q1, q3 = np.percentile(data, 25, 1), np.percentile(data, 75, 1)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)  # 与箱线图的须一致（whis=1.5），须外的点为离群点
        numbers[size] = {'instances': int(len(data)), 'solutions_per_instance': int(data.shape[1]),
                         'median_per_instance': med.tolist(), 'min_per_instance': data.min(1).tolist(),
                         'q1_per_instance': q1.tolist(), 'q3_per_instance': q3.tolist(),
                         'max_per_instance': data.max(1).tolist(), 'mean_per_instance': data.mean(1).tolist(),
                         'outliers_low_per_instance': (data < low[:, None]).sum(1).tolist(),
                         'outliers_high_per_instance': (data > high[:, None]).sum(1).tolist(),
                         'median_per_mode': {mode: np.median(arrays[mode + '_kept_costs'], 1).tolist()
                                             for mode in ('soft', 'hard')},
                         'axis_break': y1, 'above_axis_break_per_instance': above.tolist()}
    return numbers


def _differences(new, old, path=''):
    """old 中的每一项在 new 中是否相同（浮点按相对误差 1e-9 比较；new 多出的项不计）；返回不同项的路径"""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [path]
        keys = {str(k): k for k in new}
        out = []
        for k, v in old.items():
            if str(k) not in keys:
                out.append('{}/{} (missing)'.format(path, k))
            else:
                out += _differences(new[keys[str(k)]], v, '{}/{}'.format(path, k))
        return out
    if isinstance(old, list):
        if not isinstance(new, (list, tuple)) or len(new) != len(old):
            return [path]
        return [d for i, (a, b) in enumerate(zip(new, old)) for d in _differences(a, b, '{}[{}]'.format(path, i))]
    if isinstance(old, float) or isinstance(new, float):
        ok = old is not None and new is not None and bool(np.isclose(float(new), float(old), rtol=1e-9, atol=1e-12))
        return [] if ok else [path]
    return [] if new == old else [path]


def figures_only(res, fig_dir, numbers_path):
    """只重画图3–6，表2与 paper_numbers.json 不动；并核对画图所用的数字与 paper_numbers.json 中的相同"""
    cfg = res.cfg
    new = {}
    if cfg.experiments.fig3:
        new['figure3'] = figure3(res, fig_dir)
    if cfg.experiments.fig4:
        new['figure4'] = figure4(res, fig_dir)
    if cfg.experiments.ablation:
        new['ablation'] = figure5_and_ablation(res, fig_dir)
    if cfg.experiments.fig6:
        new['figure6'] = figure6(res, fig_dir)
    old = _json(numbers_path) if os.path.isfile(numbers_path) else {}
    for key, value in new.items():
        diffs = _differences(value, old[key], key) if key in old else ['{} not in {}'.format(key, numbers_path)]
        print('{}: {}'.format(key, 'same numbers as paper_numbers.json' if not diffs
                              else '{} differences, e.g. {}'.format(len(diffs), diffs[:5])))
    # 画图所用的全部数字（含 paper_numbers.json 中没有的新曲线）与缺失的结果文件
    new['missing'] = sorted(res.missing)
    from atomic_io import save_json
    save_json(new, os.path.join(fig_dir, 'figure_numbers.json'), indent=1, default=float)
    print('Figures and figure_numbers.json written to {} (missing results: {})'.format(fig_dir, len(res.missing)))


def main():
    parser = argparse.ArgumentParser(description="Build Tables 2-3, Figures 3-6 and paper_numbers.json")
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--fig-dir', default=None, help='where to write Figures 3-6 (default: <results>/figures)')
    parser.add_argument('--figures-only', action='store_true',
                        help='redraw Figures 3-6 only and check their numbers against the existing paper_numbers.json; '
                             'Table 2 and paper_numbers.json are left untouched')
    opts = parser.parse_args()
    cfg = load_config(opts.config)
    res = Results(cfg)
    out_dir = cfg.paths.results
    fig_dir = opts.fig_dir or os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    if opts.figures_only:
        figures_only(res, fig_dir, os.path.join(out_dir, 'paper_numbers.json'))
        return

    numbers = {'config': opts.config}
    if cfg.experiments.table2:
        numbers['table2'] = table2(res, out_dir)
    if cfg.experiments.fig3:
        numbers['figure3'] = figure3(res, fig_dir)
    if cfg.experiments.fig4:
        numbers['figure4'] = figure4(res, fig_dir)
    if cfg.experiments.ablation:
        numbers['ablation'] = figure5_and_ablation(res, fig_dir)
        numbers['table3'] = table3(res, out_dir)
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
