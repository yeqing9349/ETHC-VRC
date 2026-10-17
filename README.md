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
- 已完成的任务自动跳过；训练和元启发式求解中断后重新运行会从断点继续。同一配置同时只能有一个 `run_all.py` 在运行（锁文件 `results/<配置名>/run_all.lock`）。
- `--dry-run` 列出任务及状态；`--only train,eval` 只运行指定阶段；`--tasks-per-gpu` 指定每张卡同时运行的 GPU 任务数，可按规模给出（如 `20:3,50:3,100:2`）；`--emit-slurm slurm/` 为算力中心生成 Slurm 作业脚本（每个阶段一个作业数组，按依赖提交）。
- 全部设置集中在 `configs/paper.yaml`：规模、λ 列表 {1, 3, 5, 7, 9}、消融种子 {1234, 1235, 1236}、训练与评估参数、元启发式参数。
- 阶段：`data`（数据检查）→ `train` → `eval` → `solve`（OR-Tools、TS、BA，时限读取 `results/calibration/time_limits.json`）→ `timing`（表2的计时：同规模的评估在其他任务都完成后独占重跑一次，任务名带 `_timing`）→ `equal_time`（OR-Tools 以 ETHC-VRC(S) 的实测每实例耗时为时限的等时比较）→ `analysis`。
- 派发顺序在 `schedule` 段：表2（按规模从小到大）→ 多种子消融、图4、图6 → λ 扫描，同一级里预计时长长的先跑，空出的卡与 CPU 由后面的任务填上；`schedule.solo` 匹配的训练（n=50、n=100 表2的 ETHC-VRC 与 MVMoE）单独占一张卡。表2各规模的对比因此先出来，可用 `python -m analysis.compare --config configs/paper.yaml --size 50` 查看。

## 在 AutoDL 等租用服务器上运行

训练、评估与元启发式求解都在服务器上进行；元启发式的时限在服务器的 CPU 上重新标定（`configs/autodl*.yaml`）。显卡驱动需支持 CUDA 12.8（≥ 570，RTX 5090 主机均满足）。

### 安装与测速

```bash
# 1) 安装（代码、Python 3.11 和锁定版本的依赖都放在数据盘 /root/autodl-tmp）
source /etc/network_turbo        # AutoDL 学术资源加速，用于访问 GitHub
git clone -b paper-repro https://github.com/yeqing9349/ETHC-VRC.git /root/autodl-tmp/ETHC-VRC
bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_setup.sh
source /root/autodl-tmp/venv/bin/activate && cd /root/autodl-tmp/ETHC-VRC

# 2) 先租 1 张卡测速：冒烟测试 + 在服务器 CPU 上标定元启发式时限，再估算不同卡数的时长与费用
#    （--cpus-per-gpu 为每张卡分配的 CPU 线程数，--price 为每卡每小时的租金；--gain 为每卡同时运行 --tasks-per-gpu
#    个任务时相对单个任务的吞吐倍数，RTX 5090 上实测 n=20、50 各 3 个时约为 2.15、1.91，n=100 两个时约为 1.37）
python run_all.py --config configs/autodl_smoke.yaml --gpus all
python estimate.py --paper configs/autodl.yaml --smoke configs/autodl_smoke.yaml --gpus 1,2,4 --cpus-per-gpu 25 \
    --price 2.78 --tasks-per-gpu 20:3,50:3,100:2 --gain 20:2.15,50:1.91,100:1.37
```

### 正式运行

论文的正式运行用 2 张 RTX 5090，约 12 天。在实例上执行：

```bash
nohup bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_run.sh >> /root/autodl-tmp/run_paper.log 2>&1 &
```

`scripts/autodl_run.sh` 在仓库根目录运行下面这条命令，结束后关机：

```bash
/root/autodl-tmp/venv/bin/python run_all.py --config configs/autodl.yaml --gpus all --tasks-per-gpu 20:3,50:3,100:2
```

| 部分 | 作用 |
|---|---|
| `--config configs/autodl.yaml` | 实验设置与 `paper.yaml` 相同。元启发式时限读取服务器上的标定结果 `results/calibration/time_limits_autodl.json`；并行进程数按容器的 CPU 配额自动确定（2 张卡共 18 个）；另有硬件检查（见下文） |
| `--gpus all` | 使用实例上的全部显卡，卡数变了也不用改命令 |
| `--tasks-per-gpu 20:3,50:3,100:2` | 每张卡同时运行的 GPU 任务数：n=20、50 的任务 3 个，n=100 的 2 个；评估按训练规模与测试规模中较大的一个计。小规模训练受 CPU 发起计算的速度限制，单个任务用不满 GPU；两个 n=100 训练任务的显存合计约 27 GB，3 个放不下 32 GB |
| `nohup … &` | 在后台运行，关闭终端或断开 SSH 不影响运行 |
| `>> /root/autodl-tmp/run_paper.log 2>&1` | 调度进度（每个任务的开始、完成与失败）追加写入日志，续跑时不覆盖；各任务自己的输出在 `results/paper/logs/` |
| 结束后关机 | 全部完成、有任务失败、或有任务因硬件不符暂未运行时关机。已有运行在进行、本机没有可运行的任务、被 `kill` 中断时不关机。不想自动关机，就在命令前加 `NO_SHUTDOWN=1` |

查看进度：

```bash
tail -f /root/autodl-tmp/run_paper.log                       # 调度进度
cat /root/autodl-tmp/ETHC-VRC/results/paper/run_all.lock     # 正在运行的任务及其进程号
cd /root/autodl-tmp/ETHC-VRC && /root/autodl-tmp/venv/bin/python run_all.py --config configs/autodl.yaml \
    --gpus all --tasks-per-gpu 20:3,50:3,100:2 --dry-run       # 各阶段已完成与剩余的任务数、剩余任务列表
nvidia-smi                                                    # 显卡占用
```

### 没有空闲的 8 卡机器时

可以用 4 台 2 卡机器分担（每台都是 RTX 5090、Xeon Platinum 8470Q、每卡 25 个 CPU 线程）。`scripts/autodl_shard.sh` 把训练、评估与元启发式分成互不依赖的 4 份，评估与其模型在同一台，n=50、n=100 的 OR-Tools、TS、BA 分在三台上同时跑。按冒烟测试的耗时模拟，n=50 的表2结果约第 23 小时齐，n=100 约第 46 小时，多种子消融约第 56 小时，各台约 77–79 小时跑完。某个规模的表2结果出来后，在 1 号机上运行 `autodl_shard.sh compare 规模 …`，拉取其他各台该规模的结果（不合并），对比本方法与各基线。4 份都完成后，把 2–4 号机的结果拉到 1 号机合并，再在 1 号机上运行表2的计时、等时比较与汇总（约 5 小时，计时都在这一台上）。1 号机运行试点模型的评估，须有试点结果。各步的命令见脚本开头。

### 中断与续跑

重新执行同一条启动命令即从断点继续，已经跑完的任务不会重跑：

| 情况 | 重新运行后 | 最多损失 |
|---|---|---|
| 已完成的任务 | 跳过（以结果文件是否存在判断，试点阶段跑完的任务也一样） | 无 |
| 训练中断 | 从上一个跑完的 epoch 接着训练（`last.pt`）；中断的那个 epoch 已写入的训练日志删去重写 | 一个 epoch（n=100 约 30–40 分钟） |
| 元启发式中断 | 从最近一次保存的进度继续（每 240 个实例保存一次） | n=50、100 约半小时 |
| 评估中断 | 这一个评估从头重做 | 一两个小时 |

- 中途换卡数也可以：`--gpus all` 按当时的卡数分配任务，元启发式的进程数随之调整。
- 检查点与结果文件都先写临时文件、写完再改名替换，保存时断电或关机不会留下写坏的文件。
- 主动停止：`kill <run_all.py 的进程号>`（进程号见 `results/paper/run_all.lock`）。正在运行的任务随之结束，下次续跑；这样停止不会关机。
- 防止重复启动：同一配置同时只能有一个运行。上一次的运行或它启动的任务还没结束时，再次启动会列出它们的进程号后退出，也不会关机。

### 硬件检查

`configs/autodl.yaml` 的 `hardware` 段规定两类任务的硬件，与论文第5.1.3节一致：

- 表2的计时只在 RTX 5090 上运行；
- 元启发式（OR-Tools、TS、BA 及等时比较）只在 Xeon Platinum 8470Q 上运行：时限是在这种 CPU 上标定的，而 n=50、100 在时限内还没收敛，结果取决于单核速度。

本机不符时，这些任务以及依赖它们的任务暂不运行，其余任务照常运行，日志列出暂缓的任务及原因；之后在符合的机器上执行同一条命令即可补跑。用无卡模式开机时看不到显卡，命令不会启动任何任务。确需换用其他硬件时，先在新硬件上重新标定时限（`autodl_smoke.yaml`）并重跑已完成的元启发式，再修改 `hardware` 段。

### 计费提示

- 按量计费：运行结束自动关机后即停止计费；但关机后显卡不保留，再开机时可能没有空闲的卡。
- 包周、包月：租期内显卡一直保留，关机也照常计时；运行结束后在控制台转回按量计费，退还剩余费用。

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
| 表2、图3–6 与统计检验 | `analysis/report.py`；运行中查看某个规模的表2对比用 `analysis/compare.py` |

## 论文中未逐项列出的实现细节

- 网络输入：需求与剩余容量 C_r 按车辆容量 C 归一，时间窗按 T_w 归一；解码上下文中的到达时刻 T_a 以小时计，S_v = 2。仓库没有需求和时间窗，由其坐标经单独的线性层嵌入。
- 公式(14)–(16) 中相邻的线性投影按 Kool et al. 的实现合并（W_Q^g 并入上下文投影，W_Q 并入多头注意力的输出投影），二者数学上等价。
- 训练：梯度按 L2 范数 1.0 裁剪；第一个 epoch 用指数移动平均基线（衰减系数 0.8）预热，之后用贪心 rollout 基线，与 Kool et al. 的实现及原始实验相同（不预热时第一个 epoch 的基线是未训练的随机策略，软约束模型在这一 epoch 几乎学不到东西）；基线每次更新时重新生成 10,000 个检验实例。`train_log.csv` 每 50 步记录一次训练批次的平均成本，`val_log.csv` 每个 epoch 记录验证集成本与基线检验结果。
- 评估：贪心解码每批 1,024 个实例；采样时每个实例的 1,280 个样本在一次批量前向中完成，每种掩码模式各用种子 1234 的随机数流。表2另报 TPA-Reward(S) 与 MVMoE(S)：在各自的掩码模式下每个实例采样 2,560 个解，与 ETHC-VRC(S) 两种模式合计的候选数相同，计时口径也相同。表2的成本取模型训练完后运行的评估；时间列由同样的评估在其他任务都完成后独占重跑一次测得（任务名带 `_timing`）。
- TPA-Reward 与 MVMoE 的结构及其对本文问题的适配写在各自文件开头；二者与 ETHC-VRC 使用同一训练流程与预算。
- OR-Tools、TS、BA：同一目标（混合成本）、容量硬约束、车队 12 辆、同一单实例时限；时限由 `BaselineMethods/calibrate.py` 标定（取平均成本与最长时限相差不超过 0.1% 的最短时限，再乘 2）。其余参数见 `configs/paper.yaml`。

## 旧版本

最初提交的代码保留为 git 标签 `v1-submission`。`outputs/cvrptw_*` 中的旧模型按旧版的到达时刻实现训练，加载时自动使用 `time_state: legacy`，只用于检验新评估器（贪心、两种模式逐实例取优在 n=20/50/100 上为 7.24/13.05/20.33）。
