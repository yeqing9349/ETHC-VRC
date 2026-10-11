"""
模型层的单元测试：随机实例上用未训练的 ETHC-VRC、w/o ET、TPA-Reward、MVMoE 解码，检查
- 解的合法性与容量约束（CVRPTW.get_costs 内部检查）；
- 解码过程中状态记录的迟到量与成本函数一致；
- 硬掩码模式（ETHC-VRC 的 hard、MVMoE）在全部实例上零迟到。
运行：uv run pytest -q
"""
import argparse

import pytest
import torch

from nets.factory import make_model
from problems.vrp.problem_vrp import CVRPTW

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def args(model, **kw):
    a = dict(model=model, embedding_dim=64, hidden_dim=64, n_encode_layers=2, normalization='batch', tanh_clipping=10.,
             n_heads=8, use_time_state=True, time_state='paper', tpa_vehicle_ratio=0.1, mvmoe_encode_layers=2,
             mvmoe_experts=4, mvmoe_topk=2)
    a.update(kw)
    return argparse.Namespace(**a)


def batch(n=20, size=64, seed=0):
    torch.manual_seed(seed)
    data = CVRPTW.make_dataset(size=n, num_samples=size)
    return {k: v.to(DEVICE) for k, v in data.data.items()}


def replay_lateness(input, pi, hard):
    state = CVRPTW.make_state(input, hard=hard)
    for t in range(pi.size(1)):
        mask = state.get_mask()[:, 0, :]
        assert not mask.gather(1, pi[:, t:t + 1]).any()
        state = state.update(pi[:, t])
    return state.lateness[:, 0], state.n_late[:, 0]


@pytest.mark.parametrize('n', [20, 50])
@pytest.mark.parametrize('use_time_state', [True, False])
@pytest.mark.parametrize('decode_type', ['greedy', 'sampling'])
def test_ethc_modes(n, use_time_state, decode_type):
    model = make_model(args('attention', use_time_state=use_time_state), CVRPTW).to(DEVICE).eval()
    model.set_decode_type(decode_type)
    input = batch(n)
    with torch.no_grad():
        for mode in ('soft', 'hard'):
            model.set_mask_mode(mode)
            _, pi = model.decode(input, model.encode(input))
            cost, _, lateness, n_late, length = CVRPTW.get_costs(input, pi)
            state_lateness, state_n_late = replay_lateness(input, pi, hard=mode == 'hard')
            assert torch.allclose(state_lateness, lateness, atol=1e-5)
            assert torch.equal(state_n_late, n_late)
            assert torch.allclose(cost, length + CVRPTW.penalty() * lateness)
            if mode == 'hard':
                assert (n_late == 0).all() and (lateness < 1e-5).all()


def test_ethc_forward_matches_decode():
    model = make_model(args('attention'), CVRPTW).to(DEVICE).eval()
    model.set_decode_type('greedy')
    input = batch(20)
    with torch.no_grad():
        cost, ll, lateness, n_late, length = model(input)
        _, pi = model.decode(input, model.encode(input))
    assert torch.allclose(cost, CVRPTW.get_costs(input, pi)[0])
    assert (ll <= 0).all()


@pytest.mark.parametrize('n', [20, 50])
@pytest.mark.parametrize('decode_type', ['greedy', 'sampling'])
def test_tpa_reward(n, decode_type):
    model = make_model(args('tpa'), CVRPTW).to(DEVICE).eval()
    model.set_decode_type(decode_type)
    input = batch(n)
    with torch.no_grad():
        cost, ll, lateness, n_late, length = model(input)
    assert torch.isfinite(cost).all() and (ll <= 0).all()
    assert model.n_vehicles(n) == n // 10


def test_tpa_handler_reloads_when_capacity_runs_out():
    """容量很小时每辆车只能服务少数客户：调度器必须让车回仓库装货（新车），解仍然合法"""
    model = make_model(args('tpa'), CVRPTW).to(DEVICE).eval()
    model.set_decode_type('sampling')
    input = batch(20)
    input['demand'] = torch.full_like(input['demand'], 0.45)
    with torch.no_grad():
        _, pi = model.decode(input, model.encode(input))
    CVRPTW.get_costs(input, pi)  # 内部检查容量与合法性
    assert (CVRPTW.count_vehicles(pi) >= 10).all()


@pytest.mark.parametrize('decode_type', ['greedy', 'sampling'])
def test_mvmoe_hard_time_windows(decode_type):
    model = make_model(args('mvmoe'), CVRPTW).to(DEVICE)
    model.train()
    model.set_decode_type('sampling')
    input = batch(20)
    cost, ll, lateness, n_late, length = model(input)
    assert torch.isfinite(model.aux_loss) and model.aux_loss.requires_grad
    (cost.detach() * ll).mean().backward()
    model.eval()
    model.set_decode_type(decode_type)
    with torch.no_grad():
        cost, ll, lateness, n_late, length = model(input)
    assert (n_late == 0).all()


def test_sampling_repeats_embeddings():
    """评估器的采样：编码一次，再把实例与嵌入重复 k 份解码，结果合法"""
    from utils.functions import do_batch_rep
    model = make_model(args('attention'), CVRPTW).to(DEVICE).eval()
    model.set_decode_type('sampling')
    model.set_mask_mode('hard')
    input = batch(20, size=2)
    with torch.no_grad():
        emb = model.encode(input)
        rep_input, rep_emb = do_batch_rep((input, emb), 16)
        _, pi = model.decode(rep_input, rep_emb)
        cost, _, lateness, n_late, _ = CVRPTW.get_costs(rep_input, pi)
    assert pi.size(0) == 32 and (n_late == 0).all()
