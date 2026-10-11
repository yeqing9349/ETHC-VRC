import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model
    parser.add_argument('--model', default='attention',
                        help="Model: 'attention' (ETHC-VRC, default), 'tpa' (TPA-Reward), 'mvmoe' (MVMoE) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--no_time_state', action='store_true',
                        help='CVRPTW: remove the explicit time-state (T_a, S_v) from the decoder context (w/o ET)')
    parser.add_argument('--time_state', default='paper',
                        help="CVRPTW: 'paper' (Algorithm 1) or 'legacy' (old implementation, regression checks only)")
    parser.add_argument('--tpa_vehicle_ratio', type=float, default=0.1,
                        help='TPA-Reward: number of vehicle slots K = round(ratio * n) (K = N/10 in Zong et al.)')
    parser.add_argument('--mvmoe_encode_layers', type=int, default=6, help='MVMoE: number of encoder layers')
    parser.add_argument('--mvmoe_experts', type=int, default=4, help='MVMoE: number of experts per MoE layer')
    parser.add_argument('--mvmoe_topk', type=int, default=2, help='MVMoE: experts selected per input')
    parser.add_argument('--moe_loss_coef', type=float, default=1.0,
                        help='MVMoE: weight of the load-balancing loss, which the MoE layers already scale by 1e-3 '
                             '(as in the official code, where it is added with weight 1)')

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--penalty_lambda', type=float, default=3.0,
                        help='CVRPTW: penalty scaling factor lambda, beta = lambda * T_w (Eq. 10)')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default=None,
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=0,
                        help='Number of epochs to warmup the baseline with an exponential baseline, default 0: the '
                             'greedy rollout baseline is used from the start (Algorithm 2). Only with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')
    parser.add_argument('--tf32', action='store_true', help='Allow TF32 matrix multiplications on the GPU')
    parser.add_argument('--probe_steps', type=int, default=0,
                        help='Only time this many training iterations (after warm-up) and exit, for estimate.py')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--save_dir', default=None,
                        help='Exact output directory of the run (no timestamp); default output_dir/problem_size/run_name')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=0,
                        help='Also keep epoch-n.pt every n epochs; 0 (default) keeps only last.pt and the final epoch')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help="Resume from previous checkpoint file, or 'auto' to resume from "
                                         "save_dir/last.pt when it exists")
    parser.add_argument('--tensorboard', action='store_true',
                        help='Also log to TensorBoard (needs tensorboard_logger); CSV logs are always written')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.use_time_state = not opts.no_time_state
    if opts.save_dir is None:
        opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
        opts.save_dir = os.path.join(
            opts.output_dir,
            "{}_{}".format(opts.problem, opts.graph_size),
            opts.run_name
        )
    else:
        opts.run_name = os.path.basename(os.path.normpath(opts.save_dir))
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
