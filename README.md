# ETHC-VRC

求解带时间窗的容量约束车辆路径问题（CVRPTW）的深度强化学习方法：显式时间状态解码器 + 混合约束处理（训练用软约束，推理时软约束解码与硬掩码解码各一次，逐实例保留混合成本较低的解）。本仓库包含方法、四个基线（TPA-Reward、MVMoE、TS、BA）、OR-Tools 对照，以及一条命令复现论文全部数据的编排脚本。

## 环境

```bash
uv sync                      # Python 3.11 + PyTorch 2.x（CUDA 12.6），版本锁定在 uv.lock
uv run pytest -q             # 单元测试：时间状态、掩码、成本函数、各模型与元启发式
```

仓库若放在云同步目录中，可把虚拟环境和模型检查点放到同步目录之外：

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/ethc-vrc   # 虚拟环境
export ETHC_OUTPUTS=~/ethc-vrc-runs                # 模型检查点（结果文件仍写在 results/）
```

## 一条命令复现论文数据

```bash
# 1) 冒烟测试：极小设置跑通全部阶段，测各模型的计时并标定元启发式时限，然后估算正式运行的 GPU·h 与 CPU·h
uv run python run_all.py --config configs/smoke.yaml
uv run python estimate.py                      # 输出 results/smoke/estimate.md

# 2) 正式运行（表2、图3–6、正文统计）
uv run python run_all.py --config configs/paper.yaml            # 单卡
uv run python run_all.py --config configs/paper.yaml --gpus 0,1,2,3
```

- 结果：`results/paper/table2.csv`、`table2.tex`、`figures/*.pdf`（图3–6），以及 `paper_numbers.json`（正文引用的每个数字及其来源文件）。
- 已完成的任务自动跳过；训练和元启发式求解中断后重新运行会从断点继续。
- `--dry-run` 列出任务及状态；`--only train,eval` 只运行指定阶段；`--emit-slurm slurm/` 为算力中心生成 Slurm 作业脚本（每个阶段一个作业数组，按依赖提交）。
- 全部设置集中在 `configs/paper.yaml`：规模、λ 列表 {1, 3, 5, 7, 9}、消融种子 {1234, 1235, 1236}、训练与评估参数、元启发式参数。
- 阶段：`data`（数据检查）→ `train` → `eval` → `solve`（OR-Tools、TS、BA，时限读取 `results/calibration/time_limits.json`）→ `timing`（表2的计时评估，独占运行）→ `equal_time`（OR-Tools 以 ETHC-VRC(S) 的实测每实例耗时为时限的等时比较）→ `analysis`。

## 在 AutoDL 等租用服务器上运行

训练、评估与元启发式求解都在服务器上进行；元启发式的时限在服务器的 CPU 上重新标定（`configs/autodl*.yaml`）。显卡驱动需支持 CUDA 12.8（≥ 570，RTX 5090 主机均满足）。

```bash
# 1) 安装（代码、Python 3.11 和锁定版本的依赖都放在数据盘 /root/autodl-tmp）
source /etc/network_turbo        # AutoDL 学术资源加速，用于访问 GitHub
git clone -b paper-repro https://github.com/yeqing9349/ETHC-VRC.git /root/autodl-tmp/ETHC-VRC
bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_setup.sh
source /root/autodl-tmp/venv/bin/activate && cd /root/autodl-tmp/ETHC-VRC

# 2) 先租 1 张卡测速：冒烟测试 + 在服务器 CPU 上标定元启发式时限，再估算不同卡数的时长与费用
#    （--cpus-per-gpu 为每张卡分配的 CPU 核数，--price 为每卡每小时的租金）
python run_all.py --config configs/autodl_smoke.yaml --gpus all
python estimate.py --paper configs/autodl.yaml --smoke configs/autodl_smoke.yaml --gpus 1,2,4,8 --cpus-per-gpu 25 --price 2.78

# 3) 正式运行（多卡机器上；断线或关机后重新运行同一命令即可续跑）
nohup python run_all.py --config configs/autodl.yaml --gpus all > run_paper.log 2>&1 &
```

运行结束后取回 `results/paper/`（表2、图3–6、`paper_numbers.json` 与逐实例结果）；模型在 `outputs/paper/`。

## 代码与论文的对应

| 论文 | 代码 |
|---|---|
| 实例生成（第5.1.1节） | `generate_data.py`（测试集 NumPy 种子 1234、验证集 4321）；训练实例 `CVRPTWDataset`（PyTorch 随机数） |
| 混合成本（公式1）与 β = λ·T_w（公式10） | `problems/vrp/problem_vrp.py` 中的 `CVRPTW.get_costs`、`CVRPTW.penalty` |
| 时间状态更新（算法1） | `problems/vrp/state_cvrp.py` 中的 `StateCVRPTW.update` |
| 软约束与硬掩码（公式18、19）及仓库规则 | `StateCVRPTW.get_mask` |
| 编码器与显式时间状态解码器（公式11–17） | `nets/attention_model.py`；`--no_time_state` 即消融的 w/o ET |
| 训练（算法2，REINFORCE + rollout 基线） | `run.py`、`train.py`、`reinforce_baselines.py` |
| 推理与评估口径（第5.1.1节） | `eval.py` |
| 基线 | `nets/tpa_reward.py`、`nets/mvmoe.py`、`BaselineMethods/ortools_cvrptw.py`、`tabu_search.py`、`bees_algorithm.py` |
| 表2、图3–6 与统计检验 | `analysis/report.py` |

## 论文中未逐项列出的实现细节

- 网络输入：需求与剩余容量 C_r 按车辆容量 C 归一，时间窗按 T_w 归一；解码上下文中的到达时刻 T_a 以小时计，S_v = 2。仓库没有需求和时间窗，由其坐标经单独的线性层嵌入。
- 公式(14)–(16) 中相邻的线性投影按 Kool et al. 的实现合并（W_Q^g 并入上下文投影，W_Q 并入多头注意力的输出投影），二者数学上等价。
- 训练：梯度按 L2 范数 1.0 裁剪；从第一个 epoch 起使用贪心 rollout 基线（不预热）；基线每次更新时重新生成 10,000 个检验实例。`train_log.csv` 每 50 步记录一次训练批次的平均成本，`val_log.csv` 每个 epoch 记录验证集成本与基线检验结果。
- 评估：贪心解码每批 1,024 个实例；采样时每个实例的 1,280 个样本在一次批量前向中完成，每种掩码模式各用种子 1234 的随机数流。计时任务（表2的时间列）在其他任务完成后独占运行。
- TPA-Reward 与 MVMoE 的结构及其对本文问题的适配写在各自文件开头；二者与 ETHC-VRC 使用同一训练流程与预算。
- OR-Tools、TS、BA：同一目标（混合成本）、容量硬约束、车队 12 辆、同一单实例时限；时限由 `BaselineMethods/calibrate.py` 标定（取平均成本与最长时限相差不超过 0.1% 的最短时限，再乘 2）。其余参数见 `configs/paper.yaml`。

## 旧版本

最初提交的代码保留为 git 标签 `v1-submission`。`outputs/cvrptw_*` 中的旧模型按旧版的到达时刻实现训练，加载时自动使用 `time_state: legacy`，只用于检验新评估器（贪心、两种模式逐实例取优在 n=20/50/100 上为 7.24/13.05/20.33）。
