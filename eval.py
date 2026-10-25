#!/usr/bin/env python
"""
统一评估器，口径与论文第5.1.1节一致：
- greedy：每种掩码模式各做一次贪心解码，实例按每批 eval_batch_size(=1024) 个处理；
- sample：每种掩码模式下，每个实例的 samples(=1280) 个样本在一次批量前向中完成；
多种模式时逐实例保留混合成本最低的解；计时为整个测试集的总耗时，包含所有模式。
输出 <out>.npz（逐实例结果：各模式与所选解的指标和解本身，即访问序列）和 <out>.json（汇总指标与计时）。
访问序列与元启发式的格式相同：各路径的客户依次排列，路径之间为仓库 0，末尾补 0。
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from atomic_io import save_json, savez_compressed
from utils import load_model, move_to
from utils.functions import do_batch_rep

FIELDS = ('cost', 'length', 'lateness', 'n_late', 'vehicles')


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _metrics(problem, batch, pi):
    """逐实例指标：混合成本、路径长度、总迟到时长（小时）、迟到客户数、车辆数，形状 (batch_size, 5)"""
    cost, _, lateness, n_late, length = problem.get_costs(batch, pi)
    vehicles = problem.count_vehicles(pi).float()
    return torch.stack((cost, length, lateness, n_late, vehicles), 1)


def _slice(data, start, end):
    return {k: v[start:end] for k, v in data.items()}


def set_mode(model, decode_type, mask_mode):
    model.set_decode_type(decode_type)
    model.set_mask_mode(mask_mode)


def eval_greedy(model, data, n, mask_mode, opts):
    """一次贪心解码，返回 (n, 5) 逐实例指标与各批的访问序列（留在 GPU 上，计时结束后再用 stack_tours 合并）"""
    set_mode(model, 'greedy', mask_mode)
    torch.manual_seed(opts.seed)  # MVMoE 的分层门控在贪心解码时也随机抽取分支（与官方实现相同），固定种子使结果可复现
    out, tours = [], []
    with torch.no_grad():
        for start in range(0, n, opts.eval_batch_size):
            batch = _slice(data, start, min(start + opts.eval_batch_size, n))
            _, pi = model.decode(batch, model.encode(batch))
            out.append(_metrics(model.problem, batch, pi))
            tours.append(pi)
    return torch.cat(out, 0), tours


def eval_sample(model, data, n, mask_mode, opts, keep=0):
    """
    每个实例采样 opts.samples 次，保留混合成本最低的样本，返回 (n, 5) 逐实例指标、
    keep>0 时前 keep 个实例全部样本的混合成本 (keep, samples)（用于图6），以及所保留样本的访问序列
    """
    set_mode(model, 'sampling', mask_mode)
    torch.manual_seed(opts.seed)
    k = opts.instances_per_pass
    best, kept, tours = [], [], []
    with torch.no_grad():
        for start in range(0, n, k):
            batch = _slice(data, start, min(start + k, n))
            m = batch['loc'].size(0)
            embeddings = model.encode(batch)
            rep_batch, rep_embeddings = do_batch_rep((batch, embeddings), opts.samples)
            _, pi = model.decode(rep_batch, rep_embeddings)
            metrics = _metrics(model.problem, rep_batch, pi).view(opts.samples, m, len(FIELDS))
            idx = metrics[:, :, 0].argmin(0)
            rows = torch.arange(m, device=idx.device)
            best.append(metrics[idx, rows])
            tours.append(pi.view(opts.samples, m, -1)[idx, rows])  # 样本按 do_batch_rep 的顺序排列：第 s 个样本在第 s 块
            if start < keep:
                kept.append(metrics[:, :, 0].t()[:max(0, keep - start)])
    return torch.cat(best, 0), (torch.cat(kept, 0) if kept else None), tours


def stack_tours(tours, width):
    """各批访问序列（长度不一）末尾补 0 到同一宽度，合并为 int16 数组"""
    out = np.zeros((sum(t.size(0) for t in tours), width), dtype=np.int16)
    row = 0
    for t in tours:
        out[row:row + t.size(0), :t.size(1)] = t.cpu().numpy()
        row += t.size(0)
    return out


def summarize(metrics, problem_penalty):
    """metrics: (n, 5) numpy"""
    cost, length, lateness, n_late, vehicles = metrics.T
    late_customers = n_late.sum()
    return {
        'cost': float(cost.mean()),
        'length': float(length.mean()),
        'lateness_per_instance_h': float(lateness.mean()),
        'feasibility_rate': float((n_late == 0).mean()),
        'late_instance_rate': float((n_late > 0).mean()),
        'avg_lateness_per_late_customer_h': float(lateness.sum() / late_customers) if late_customers > 0 else 0.0,
        'vehicles': float(vehicles.mean()),
        'penalty_beta': problem_penalty,
    }


def run(opts):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    if opts.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, args = load_model(opts.model)
    problem = model.problem
    penalty_lambda = opts.penalty_lambda if opts.penalty_lambda is not None else args.get('penalty_lambda', 3.0)
    problem.set_penalty_lambda(penalty_lambda)
    model.to(device)
    model.eval()

    dataset = problem.make_dataset(filename=opts.dataset, num_samples=opts.num_instances, offset=opts.offset)
    n = len(dataset)
    data = move_to(dataset.data, device)

    if opts.timing:  # 预热，不计时
        warm = _slice(data, 0, min(n, 64 if opts.strategy == 'sample' else opts.eval_batch_size))
        warm_opts = argparse.Namespace(**{**vars(opts), 'samples': min(opts.samples, 128), 'instances_per_pass': 1})
        for mode in opts.modes:
            if opts.strategy == 'greedy':
                eval_greedy(model, warm, warm['loc'].size(0), mode, opts)
            else:
                eval_sample(model, warm, 2, mode, warm_opts)
        _sync(device)

    per_mode, times, kept, tours = {}, {}, {}, {}
    for mode in opts.modes:
        _sync(device)
        t0 = time.perf_counter()
        if opts.strategy == 'greedy':
            m, tours[mode] = eval_greedy(model, data, n, mode, opts)
        else:
            m, kept_costs, tours[mode] = eval_sample(model, data, n, mode, opts, keep=opts.keep_samples)
            if kept_costs is not None:
                kept[mode] = kept_costs.cpu().numpy()
        _sync(device)
        times[mode] = time.perf_counter() - t0
        per_mode[mode] = m.cpu().numpy()
    width = max(t.size(1) for mode_tours in tours.values() for t in mode_tours)
    tours = {mode: stack_tours(mode_tours, width) for mode, mode_tours in tours.items()}

    # 逐实例保留混合成本较低的模式（成本相同时取列表中靠前的模式，即soft）
    stacked = np.stack([per_mode[mode] for mode in opts.modes], 0)  # (n_modes, n, 5)
    best_mode = stacked[:, :, 0].argmin(0)
    best = stacked[best_mode, np.arange(n)]
    best_tours = np.stack([tours[mode] for mode in opts.modes], 0)[best_mode, np.arange(n)]

    summary = {
        'model': opts.model,
        'dataset': opts.dataset,
        'strategy': opts.strategy,
        'modes': list(opts.modes),
        'samples': opts.samples if opts.strategy == 'sample' else 1,
        'instances_per_pass': opts.instances_per_pass if opts.strategy == 'sample' else None,
        'eval_batch_size': opts.eval_batch_size if opts.strategy == 'greedy' else None,
        'n': n,
        'offset': opts.offset,
        'penalty_lambda': penalty_lambda,
        'seed': opts.seed,
        'best': summarize(best, problem.penalty()),
        'per_mode': {mode: summarize(per_mode[mode], problem.penalty()) for mode in opts.modes},
        'mode_share': {mode: float((best_mode == i).mean()) for i, mode in enumerate(opts.modes)},
        'time_s': {**times, 'total': float(sum(times.values()))},
        'timing_protocol': bool(opts.timing),
        'tf32': bool(opts.tf32),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'torch': torch.__version__,
    }

    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    arrays = {f'{mode}_{field}': per_mode[mode][:, j] for mode in opts.modes for j, field in enumerate(FIELDS)}
    arrays.update({f'best_{field}': best[:, j] for j, field in enumerate(FIELDS)})
    arrays['best_mode'] = best_mode
    arrays.update({f'{mode}_tours': tours[mode] for mode in opts.modes})
    arrays['best_tours'] = best_tours
    arrays.update({f'{mode}_kept_costs': v for mode, v in kept.items()})
    # 先写逐实例结果，再写汇总（run_all.py 以汇总文件判断任务完成）；都用原子写入，中途断电不会留下写坏的文件
    savez_compressed(opts.out + '.npz', **arrays)
    save_json(summary, opts.out + '.json', indent=1, ensure_ascii=False)

    print('{} {} {} n={}: cost {:.4f} | {} | time {:.2f}s'.format(
        os.path.basename(opts.model), opts.strategy, '+'.join(opts.modes), n, summary['best']['cost'],
        ', '.join('{} {:.4f}'.format(mode, summary['per_mode'][mode]['cost']) for mode in opts.modes),
        summary['time_s']['total']))
    return summary


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a CVRPTW test set")
    parser.add_argument('--model', required=True, help='Checkpoint file (.pt) or run directory')
    parser.add_argument('--dataset', required=True, help='Test set (.pkl)')
    parser.add_argument('--out', required=True, help='Output path without extension (.npz and .json are written)')
    parser.add_argument('--strategy', choices=('greedy', 'sample'), default='greedy')
    parser.add_argument('--modes', nargs='+', choices=('soft', 'hard'), default=['soft', 'hard'],
                        help='Masking modes decoded per instance; the lowest-cost solution is kept')
    parser.add_argument('--samples', type=int, default=1280, help='Samples per instance and mode (sample)')
    parser.add_argument('--instances_per_pass', type=int, default=1,
                        help='Instances decoded together when sampling (1 = the timing protocol of the paper)')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='Instances per batch (greedy)')
    parser.add_argument('--keep_samples', type=int, default=0,
                        help='Store the costs of all samples for the first k instances (Figure 6)')
    parser.add_argument('--num_instances', type=int, default=10000)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--penalty_lambda', type=float, default=None,
                        help='Penalty scaling factor lambda (beta = lambda * T_w); defaults to the training value')
    parser.add_argument('--seed', type=int, default=1234, help='Seed of the sampling stream of each mode')
    parser.add_argument('--timing', action='store_true', help='Warm up before timing (timing protocol of the paper)')
    parser.add_argument('--tf32', action='store_true', help='Allow TF32 matrix multiplications')
    parser.add_argument('--no_cuda', action='store_true')
    return parser.parse_args(args)


if __name__ == "__main__":
    run(get_options())
