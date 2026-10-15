#!/usr/bin/env python

import os
import json
import time
import pprint as pp

import torch
import torch.optim as optim

from nets.critic_network import CriticNetwork
from nets.factory import make_model
from options import get_options
from train import train_epoch, validate, get_inner_model, probe_training
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.pointer_network import CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from utils.log_utils import CsvLogger, truncate_csv_log
from atomic_io import save_json

TRAIN_LOG_FIELDS = ('step', 'epoch', 'batch_id', 'cost', 'length', 'lateness', 'n_late', 'loss', 'grad_norm', 'aux_loss')
VAL_LOG_FIELDS = ('epoch', 'step', 'val_cost', 'val_length', 'val_lateness', 'val_late_rate', 'epoch_time_s',
                  'baseline_epoch', 'baseline_p', 'baseline_updated')


def run(opts):

    # 打印运行的参数
    pp.pprint(vars(opts))

    # 设置随机种子
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if opts.tensorboard:
        from tensorboard_logger import Logger as TbLogger
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

####################  创建输出目录，续训时沿用已有目录  ####################
    os.makedirs(opts.save_dir, exist_ok=True)
    if opts.resume == 'auto':
        last_path = os.path.join(opts.save_dir, 'last.pt')
        opts.resume = last_path if os.path.isfile(last_path) else None
    if opts.resume is None and opts.probe_steps == 0:
        # Save arguments so exact configuration can always be found
        save_json(vars(opts), os.path.join(opts.save_dir, "args.json"), indent=True)
#########################################################################
    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    if opts.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

#########################  加载问题  #########################
    problem = load_problem(opts.problem)
    if problem.NAME == 'cvrptw':
        problem.set_penalty_lambda(opts.penalty_lambda)  # β = λ·T_w
#############################################################

####################  加载模型参数  ####################
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path) # 从load_path中加载模型参数到load_data（字典）
######################################################

#####################################  初始化模型  #####################################
    model = make_model(opts, problem).to(opts.device)
    # 如果使用cuda并且有多个GPU，则使用DataParallel，在多个GPU上并行运行
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 用load_data中的模型参数覆盖model中的模型参数
    model_ = get_inner_model(model) # model_是对model的内部模型的引用，修改model_就是修改model
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
#########################################################################################

    if opts.probe_steps > 0:
        probe_training(model, problem, opts)
        return

###########################################  初始化baseline  ###########################################
    if opts.baseline == 'exponential': # 指数类型基线
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm': # critic类型基线（只支持TSP）
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout': # rollout类型基线
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])
############################################################################################################

########################################  初始化优化器  ########################################
    optimizer = optim.Adam( # 需要优化的模型参数，以及学习率
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device) # 张量需要在指定设备上进行计算，所以需要确保所有张量都在正确的设备上

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
###############################################################################################

#######################################################  开始训练  #######################################################
    val_dataset = problem.make_dataset(  # 加载或生成验证集
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
        distribution=opts.data_distribution)

    if opts.resume: # 如果是恢复训练，则从上次训练的epoch之后开始
        if 'epoch' in load_data:
            # last.pt/epoch-n.pt 在验证与基线检验之后保存，直接从下一个epoch继续
            epoch_resume = load_data['epoch']
        else:
            # 旧格式的检查点在基线检验之前保存，需补做一次检验
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            baseline.epoch_callback(model, epoch_resume)

        if isinstance(baseline, WarmupBaseline):
            baseline.resume(epoch_resume)

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        print("Resuming after {}{}".format(
            epoch_resume, ', warmup alpha = {}'.format(baseline.alpha) if isinstance(baseline, WarmupBaseline) else ''))
        opts.epoch_start = epoch_resume + 1

    if not opts.eval_only:
        # 续训：删去检查点之后已写入的日志行（中断的那个 epoch 会重新训练、重新记录）；从头训练：清空旧日志
        for name in ('train_log.csv', 'val_log.csv'):
            dropped = truncate_csv_log(os.path.join(opts.save_dir, name), opts.epoch_start - 1)
            if dropped:
                print('Removed {} rows logged after epoch {} from {}'.format(dropped, opts.epoch_start - 1, name))

    csv_loggers = {
        'train': CsvLogger(os.path.join(opts.save_dir, 'train_log.csv'), TRAIN_LOG_FIELDS),
        'val': CsvLogger(os.path.join(opts.save_dir, 'val_log.csv'), VAL_LOG_FIELDS)
    }

    if opts.eval_only: # 如果只是评估模型，则只进行评估
        validate(model, val_dataset, opts)
    else: # 否则进行训练，共 n_epochs 个 epoch（续训时只训练剩余的 epoch）
        start_time = time.time()
        for epoch in range(opts.epoch_start, opts.n_epochs):
            train_epoch(
                model, # 模型
                optimizer, # 优化器
                baseline, # 基线（强化学习）
                lr_scheduler, # 学习率调度器
                epoch, # 当前epoch
                val_dataset, # 验证集
                problem, # 问题实例
                tb_logger, # tensorboard日志
                opts, # 运行参数
                csv_loggers # CSV日志：训练曲线与验证记录
            )
        # 训练完成的标记，编排脚本据此跳过已完成的任务
        save_json({'final_epoch': opts.n_epochs - 1, 'resumed_from': opts.epoch_start,
                   'session_time_s': time.time() - start_time}, os.path.join(opts.save_dir, 'done.json'), indent=1)
#########################################################################################################################

if __name__ == "__main__":
    run(get_options())
