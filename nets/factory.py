"""按运行参数构造模型，训练（run.py）与加载（utils.load_model）共用"""


def make_model(args, problem):
    """
    :param args: dict（args.json）或 argparse.Namespace（命令行参数）
    :param problem: 问题类，如 CVRPTW
    :return: 未加载参数的模型
    """
    if not isinstance(args, dict):
        args = vars(args)
    name = args.get('model', 'attention')

    if name == 'attention':
        from nets.attention_model import AttentionModel
        # 旧模型的 args.json 没有 time_state 字段：按旧版到达时刻实现加载，只用于回归检验
        time_state = args.get('time_state', 'legacy' if problem.NAME == 'cvrptw' else 'paper')
        return AttentionModel(
            args['embedding_dim'],
            args['hidden_dim'],
            problem,
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            n_heads=args.get('n_heads', 8),
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None),
            use_time_state=args.get('use_time_state', True),
            time_state=time_state
        )

    if name == 'tpa':
        from nets.tpa_reward import TPARewardModel
        return TPARewardModel(
            args['embedding_dim'],
            problem,
            n_encode_layers=args['n_encode_layers'],
            n_heads=args.get('n_heads', 8),
            tanh_clipping=args['tanh_clipping'],
            vehicle_ratio=args.get('tpa_vehicle_ratio', 0.1)
        )

    if name == 'mvmoe':
        from nets.mvmoe import MVMoEModel
        return MVMoEModel(
            args['embedding_dim'],
            problem,
            n_encode_layers=args.get('mvmoe_encode_layers', 6),
            n_heads=args.get('n_heads', 8),
            tanh_clipping=args['tanh_clipping'],
            num_experts=args.get('mvmoe_experts', 4),
            topk=args.get('mvmoe_topk', 2)
        )

    if name == 'pointer':
        from nets.pointer_network import PointerNetwork
        return PointerNetwork(
            args['embedding_dim'],
            args['hidden_dim'],
            problem,
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None)
        )

    assert False, "Unknown model: {}".format(name)
