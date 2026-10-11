import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def batch_loader(dataset, batch_size):
    """
    按整批取数据：数据集支持以一批下标取出整批（见CVRPTWDataset），省去逐条拼接
    """
    return DataLoader(dataset, batch_size=None,
                      sampler=BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False))


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost, violated_time, route_length, violated_node = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    avg_violated_time = violated_time.mean()
    avg_route_length = route_length.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    print('Validation overall avg_violated_time: {} +- {}'.format(
        avg_violated_time, torch.std(violated_time) / math.sqrt(len(violated_time))))
    print('Validation overall avg_route_length: {} +- {}'.format(
        avg_route_length, torch.std(route_length) / math.sqrt(len(route_length))))

    return {
        'val_cost': avg_cost.item(),
        'val_length': avg_route_length.item(),
        'val_lateness': avg_violated_time.item(),
        'val_late_rate': (violated_node > 0).float().mean().item()
    }


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, ll, violated_time, violated_node, route_length = model(move_to(bat, opts.device))
        return torch.stack((cost.data.cpu(), violated_time.data.cpu(), route_length.data.cpu(),
                            violated_node.data.cpu()), dim=1)

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(batch_loader(dataset, opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0).unbind(dim=1)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def save_checkpoint(path, model, optimizer, baseline, epoch):
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'baseline': baseline.state_dict(),
            'epoch': epoch
        },
        path
    )


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, csv_loggers=None):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if tb_logger is not None:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    # 包装数据集：接收数据集作为参数，返回数据集和基线值（eval后的值）
    random_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    training_dataset = baseline.wrap_dataset(random_dataset)
    training_dataloader = batch_loader(training_dataset, opts.batch_size)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            csv_loggers
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    val = validate(model, val_dataset, opts)

    if tb_logger is not None:
        tb_logger.log_value('val_avg_reward', val['val_cost'], step)
        tb_logger.log_value('val_avg_violated_time', val['val_lateness'], step)
        tb_logger.log_value('val_avg_route_length', val['val_length'], step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    if csv_loggers is not None:
        inner_baseline = getattr(baseline, 'baseline', baseline)  # WarmupBaseline包装了内层基线
        csv_loggers['val'].log({
            'epoch': epoch, 'step': step, **val, 'epoch_time_s': epoch_duration,
            'baseline_epoch': getattr(inner_baseline, 'epoch', None),
            'baseline_p': getattr(inner_baseline, 'last_p', None),
            'baseline_updated': getattr(inner_baseline, 'last_updated', None)
        })

    # 每个epoch结束后保存可续训的检查点（覆盖），最后一个epoch另存为epoch-{n}.pt供测试使用
    save_checkpoint(os.path.join(opts.save_dir, 'last.pt'), model, optimizer, baseline, epoch)
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        save_checkpoint(os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)), model, optimizer, baseline, epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch, # loc、demand、depot、tw
        tb_logger,
        opts,
        csv_loggers=None
):
    x, bl_val = baseline.unwrap_batch(batch) # 解包数据集：接收批次数据作为参数，返回数据和基线值
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    # cost：采样（sampling）得到的混合成本(batch_size, )，训练只用软约束模式
    # log_likelihood：采样序列的总对数似然(batch_size, )
    cost, log_likelihood, violated_time_num, violated_node_num, route_length = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss，公式(15)：(L(π) - b(s)) ∇log p(π|s)，b(s)为基线策略的贪心解成本
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # MVMoE：专家负载均衡的辅助损失
    aux_loss = getattr(get_inner_model(model), 'aux_loss', None)
    if aux_loss is not None:
        loss = loss + opts.moe_loss_coef * aux_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, violated_time_num, violated_node_num, route_length)
        if csv_loggers is not None:
            csv_loggers['train'].log({
                'step': step, 'epoch': epoch, 'batch_id': batch_id,
                'cost': cost.mean().item(), 'length': route_length.mean().item(),
                'lateness': violated_time_num.mean().item(), 'n_late': violated_node_num.mean().item(),
                'loss': reinforce_loss.item(), 'grad_norm': float(grad_norms[0][0]),
                'aux_loss': aux_loss.item() if aux_loss is not None else None
            })


def probe_training(model, problem, opts, warmup_steps=5):
    """
    计时探针（供estimate.py估算总时长）：在完整batch下测训练每次迭代的耗时和显存峰值、
    验证集规模（10,000个实例）的贪心评估耗时，以及生成训练数据的耗时，写入 save_dir/probe.json。
    一个epoch约为：迭代次数×每次迭代耗时 + (epoch_size/val_size + 2)×贪心评估耗时 + 数据生成耗时，
    其中贪心评估依次用于rollout基线包装训练数据、验证和基线检验。
    """
    import json

    def sync():
        if opts.device.type == 'cuda':
            torch.cuda.synchronize(opts.device)

    n_batches = warmup_steps + opts.probe_steps
    t0 = time.perf_counter()
    dataset = problem.make_dataset(size=opts.graph_size, num_samples=n_batches * opts.batch_size,
                                   distribution=opts.data_distribution)
    gen_time = (time.perf_counter() - t0) * opts.epoch_size / (n_batches * opts.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr_model)
    model.train()
    set_decode_type(model, "sampling")
    if opts.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(opts.device)
    step_times = []
    for i, batch in enumerate(batch_loader(dataset, opts.batch_size)):
        x = move_to(batch, opts.device)
        sync()
        t0 = time.perf_counter()
        cost, log_likelihood, *_ = model(x)
        loss = ((cost - cost.mean().detach()) * log_likelihood).mean()
        aux_loss = getattr(get_inner_model(model), 'aux_loss', None)
        if aux_loss is not None:
            loss = loss + opts.moe_loss_coef * aux_loss
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()
        sync()
        if i >= warmup_steps:
            step_times.append(time.perf_counter() - t0)
    peak_mem_gb = torch.cuda.max_memory_allocated(opts.device) / 2 ** 30 if opts.device.type == 'cuda' else None

    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size,
                                       distribution=opts.data_distribution)
    rollout(model, val_dataset, opts)  # 预热
    sync()
    t0 = time.perf_counter()
    rollout(model, val_dataset, opts)
    sync()
    greedy_time = time.perf_counter() - t0

    steps_per_epoch = opts.epoch_size // opts.batch_size
    step_time = sum(step_times) / len(step_times)
    epoch_time = steps_per_epoch * step_time + (opts.epoch_size / opts.val_size + 2) * greedy_time + gen_time
    result = {
        'model': opts.model, 'graph_size': opts.graph_size, 'use_time_state': opts.use_time_state,
        'batch_size': opts.batch_size, 'epoch_size': opts.epoch_size, 'val_size': opts.val_size,
        'tf32': bool(opts.tf32), 'probe_steps': opts.probe_steps,
        'step_time_s': step_time, 'greedy_val_time_s': greedy_time, 'data_gen_epoch_s': gen_time,
        'peak_mem_gb': peak_mem_gb, 'epoch_time_s': epoch_time, 'n_epochs': opts.n_epochs,
        'train_time_h': epoch_time * opts.n_epochs / 3600,
        'device': torch.cuda.get_device_name(opts.device) if opts.device.type == 'cuda' else 'cpu',
        'torch': torch.__version__
    }
    with open(os.path.join(opts.save_dir, 'probe.json'), 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return result
