import argparse
import json
import os
import pickle
import numpy as np
from utils.data_utils import check_extension, save_dataset
from problems.vrp.problem_vrp import CVRPTW


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))

def generate_cvrptw_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        30: 35.,
        50: 40.,
        70: 45.,
        100: 50.
    }

    """
    生成时间窗数据，形状为[batch_size, 2]，左时间窗从[0,2,4,6]中随机选择，右时间窗为左时间窗+2
    """
    time_windows_left_options = np.arange(0, CVRPTW.WORK_TIME, CVRPTW.TIME_WINDOW_LENGTH)  # [0, 2, 4, 6]
    tw_left = np.random.choice(time_windows_left_options, size=(dataset_size, vrp_size))  # 随机选择一个时间窗的左边界
    tw_right = tw_left + CVRPTW.TIME_WINDOW_LENGTH  # 时间窗的右边界

    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # depot坐标 [batch_size, 2]
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # node坐标 [batch_size, vrp_size, 2]
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # demand（1-9） [batch_size, vrp_size]
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist(),  # Capacity [batch_size]
        np.stack((tw_left, tw_right), axis=-1).tolist(),
        np.full(dataset_size, CVRPTW.WORK_TIME).tolist()
    ))

def generate_op_data(dataset_size, op_size, prize_type='const'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist', 'cvrptw'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 30, 50, 70, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--check', default=None,
                        help="CVRPTW only: regenerate the test (seed 1234) and validation (seed 4321) sets in memory, "
                             "compare them with the files in data_dir and write the result to this JSON file")

    opts = parser.parse_args()

    if opts.check is not None:
        # 数据检查：重新生成测试集与验证集并与仓库内的 pkl 逐值比较；只作记录，正式实验始终使用仓库内的 pkl
        report = {}
        for name, seed in (('test', 1234), ('validation', 4321)):
            for graph_size in opts.graph_sizes:
                filename = os.path.join(opts.data_dir, 'cvrptw', 'cvrptw{}_{}_seed{}.pkl'.format(graph_size, name, seed))
                np.random.seed(seed)
                regenerated = generate_cvrptw_data(opts.dataset_size, graph_size)
                with open(filename, 'rb') as f:
                    stored = pickle.load(f)
                identical = len(stored) == len(regenerated) and all(
                    np.array_equal(np.array(a, dtype=float), np.array(b, dtype=float))
                    for x, y in zip(stored, regenerated) for a, b in zip(x, y))
                report[os.path.basename(filename)] = {'instances': len(stored), 'regenerated_identical': bool(identical)}
                print(filename, 'identical' if identical else 'DIFFERENT (the stored file is used)')
        os.makedirs(os.path.dirname(os.path.abspath(opts.check)), exist_ok=True)
        from atomic_io import save_json
        save_json(report, opts.check, indent=1)
        raise SystemExit(0)

    assert opts.name is not None, "--name is required"
    assert opts.filename is None or (opts.problem != 'all' and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'vrp': [None],
        'pctsp': [None],
        'op': ['const', 'unif', 'dist'],
        'cvrptw': [None]
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'vrp':
                    dataset = generate_vrp_data(
                        opts.dataset_size, graph_size)
                elif problem == 'pctsp':
                    dataset = generate_pctsp_data(opts.dataset_size, graph_size)
                elif problem == "op":
                    dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution)
                elif problem == 'cvrptw':
                    dataset = generate_cvrptw_data(opts.dataset_size, graph_size)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)
