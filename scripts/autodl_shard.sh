#!/bin/bash
# 没有空闲的 8 卡机器时，用 4 台 2 卡机器分担 AutoDL 上的正式运行。每台都须是 RTX 5090、Xeon Platinum 8470Q、每卡 25 个
# CPU 线程（与冒烟测试、时限标定的机器相同）。训练、评估与元启发式分成下面 4 份，评估与其模型在同一台，各台之间没有依赖；
# 各台按配置 schedule 段的顺序派发：n=20、n=50、n=100 的表2 → 多种子消融（与图4、图6）→ λ 扫描。按冒烟测试的单任务耗时与
# run_all.py 的调度规则模拟，n=50 的表2结果约第 23 小时齐，n=100 约第 46 小时，多种子消融约第 56 小时，各台约 77–79 小时
# 跑完。4 份都完成后把 2–4 号机的结果拉到 1 号机合并，再在 1 号机上运行表2的计时、等时比较与汇总（约 5 小时，计时都在这一台
# 上）。试点已训练的三个 n=20 模型的评估在 1 号机，1 号机须有试点结果（原实例或它的克隆）。
# 用法（代码与环境按 README 安装）：
#   1 号机（跑完不关机，留着显卡做最后的计时）：
#     NO_SHUTDOWN=1 nohup bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_shard.sh run 1 >> /root/autodl-tmp/run_paper.log 2>&1 &
#   2–4 号机（把 2 换成 3、4；跑完自动关机）：
#     nohup bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_shard.sh run 2 >> /root/autodl-tmp/run_paper.log 2>&1 &
#   某个规模的表2结果出来后，在 1 号机上拉取其他各台该规模的评估与求解结果，对比本方法与各基线（不合并，运行中随时可用；
#   参数为规模与 AutoDL 控制台中其他各台的登录指令，逐台输入密码）：
#     bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_shard.sh compare 50 "ssh -p 端口 root@地址" "ssh -p 端口 root@地址" ...
#   2–4 号机都跑完后以无卡模式开机，在 1 号机上拉取并合并它们的结果：
#     bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_shard.sh pull "ssh -p 端口 root@地址" "ssh -p 端口 root@地址" ...
#   合并后 train、eval、solve 都没有待运行的任务时，在 1 号机上运行计时、等时比较与汇总（跑完关机）：
#     nohup bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_run.sh >> /root/autodl-tmp/run_paper.log 2>&1 &
# run 后的额外参数原样传给 run_all.py（如 run 2 --dry-run 预演）；merge DIR 合并已拷到本机的另一台的结果
# （DIR 下有 outputs/paper 与 results/paper）。
cd "$(dirname "$0")/.." || exit 1
PYTHON=${PYTHON:-/root/autodl-tmp/venv/bin/python}

# 各台运行的任务（run_all.py --match 的正则）。1 号机的元启发式最先跑完，λ 扫描中 2 号机的 ortools_n50_l1 与 3、4 号机
# n=20 的 OR-Tools 在运行中改到 1 号机，各台的元启发式差不多同时跑完
SHARD_1='^((train|eval):(ethc_n100_l(1_s1234|3_s1235)|ethc_n20_l3_s1234|ethc_n50_l1_s1234|mvmoe_n100_l3_s1234|mvmoe_n20_l3_s1234|tpa_n100_l(3_s1234|9_s1234)|tpa_n20_l3_s1234|tpa_n50_l(5_s1234|7_s1234|9_s1234))(__|$)|solve:(ortools_n20_l(1|5|7|9)|ortools_n50_l(1|9)|ts_n50_l3)$)'
SHARD_2='^((train|eval):(ethc_n20_l(3_s1236|7_s1234|9_s1234)|ethc_n50_l(3_s1234|5_s1234|9_s1234)|ethc_woet_n100_l(3_s1234|3_s1236)|ethc_woet_n20_l3_s1234|ethc_woet_n50_l(3_s1234|3_s1235)|tpa_n100_l5_s1234|tpa_n20_l7_s1234)(__|$)|solve:(ba_n50_l3|ortools_n100_l(3|5))$)'
SHARD_3='^((train|eval):(ethc_n100_l(3_s1234|3_s1236)|ethc_n20_l(1_s1234|3_s1235|5_s1234)|ethc_n50_l(3_s1236|7_s1234)|ethc_woet_n20_l(3_s1235|3_s1236)|ethc_woet_n50_l3_s1236|tpa_n100_l(1_s1234|7_s1234)|tpa_n20_l(5_s1234|9_s1234)|tpa_n50_l1_s1234)(__|$)|solve:(ba_n100_l3|ortools_n100_l7|ortools_n50_l(5|7))$)'
SHARD_4='^((train|eval):(ethc_n100_l(5_s1234|7_s1234|9_s1234)|ethc_n50_l3_s1235|ethc_woet_n100_l3_s1235|mvmoe_n50_l3_s1234|tpa_n20_l1_s1234|tpa_n50_l3_s1234)(__|$)|solve:(ortools_n100_l(1|9)|ortools_n50_l3|ts_n100_l3)$)'

merge() {
    # 拷来本机还没有的文件（运行环境记录合并，锁文件不拷），再列出各阶段剩余的任务数
    "$PYTHON" - "$1" <<'PY' || exit 1
import json, os, shutil, sys
from atomic_io import save_json
src = sys.argv[1]
env = os.path.join('results', 'paper', 'environment.json')
skip = {env, os.path.join('results', 'paper', 'run_all.lock')}
copied = present = 0
for sub in (os.path.join('outputs', 'paper'), os.path.join('results', 'paper')):
    if not os.path.isdir(os.path.join(src, sub)):
        sys.exit('{} not found'.format(os.path.join(src, sub)))
    for root, _, files in os.walk(os.path.join(src, sub)):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), src)
            if rel in skip:
                continue
            if os.path.exists(rel):
                present += 1
                continue
            os.makedirs(os.path.dirname(rel), exist_ok=True)
            shutil.copy2(os.path.join(root, name), rel)
            copied += 1
records = json.load(open(env, encoding='utf-8')) if os.path.isfile(env) else []
if os.path.isfile(os.path.join(src, env)):
    records += [r for r in json.load(open(os.path.join(src, env), encoding='utf-8')) if r not in records]
    save_json(sorted(records, key=lambda r: r.get('time', '')), env, indent=1, ensure_ascii=False)
print('{}: copied {} files ({} already here)'.format(src, copied, present))
PY
    "$PYTHON" run_all.py --config configs/autodl.yaml --gpus all --tasks-per-gpu 20:3,50:3,100:2 --dry-run | grep -v '^  \['
}

case $1 in
    run)
        k=$2
        shift 2
        case $k in
            1) re=$SHARD_1 ;;
            2) re=$SHARD_2 ;;
            3) re=$SHARD_3 ;;
            4) re=$SHARD_4 ;;
            *) echo "usage: autodl_shard.sh run 1|2|3|4 [run_all.py options]"; exit 1 ;;
        esac
        [ -f results/calibration/time_limits_autodl.json ] \
            || { echo "缺少元启发式时限 results/calibration/time_limits_autodl.json：先 git pull 到最新代码"; exit 1; }
        cpu=$("$PYTHON" -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/autodl.yaml').hardware.solver_cpu)")
        grep -qF "$cpu" /proc/cpuinfo || { echo "本机 CPU 不是 $cpu，这一份的元启发式不能在这里运行"; exit 1; }
        if [ "$k" = 1 ]; then
            for f in outputs/paper/{ethc,tpa,mvmoe}_n20_l3_s1234/done.json results/paper/solvers/{ortools,ts,ba}_n20_l3.json; do
                [ -f "$f" ] || { echo "1 号机缺少试点结果 $f"; exit 1; }
            done
        fi
        exec bash scripts/autodl_run.sh --only train,eval,solve --match "$re" "$@"
        ;;
    compare)
        size=$2
        shift 2
        [ -n "$size" ] && [ $# -gt 0 ] || { echo 'usage: autodl_shard.sh compare SIZE "ssh -p PORT root@HOST" ...'; exit 1; }
        set -o pipefail
        dirs=()
        for login in "$@"; do
            # 只拷该规模已完成的评估与求解结果，放在单独的目录里；其他各台还在运行，合并留到最后
            dir=/root/autodl-tmp/compare/$(echo "$login" | tr -c 'A-Za-z0-9.\n' '_')
            rm -rf "$dir"
            mkdir -p "$dir"
            $login "cd /root/autodl-tmp/ETHC-VRC && find results/paper/eval results/paper/solvers -name '*_n${size}_*' ! -name '*.partial.npz' -print0 2>/dev/null | tar --null -T - -cf -" \
                | tar -C "$dir" -xf - || { echo "未能从 $login 拷贝结果"; exit 1; }
            dirs+=("$dir/results/paper")
        done
        exec "$PYTHON" -m analysis.compare --config configs/autodl.yaml --size "$size" --results results/paper "${dirs[@]}"
        ;;
    merge)
        [ -n "$2" ] || { echo "usage: autodl_shard.sh merge DIR"; exit 1; }
        merge "$2"
        ;;
    pull)
        shift
        [ $# -gt 0 ] || { echo 'usage: autodl_shard.sh pull "ssh -p PORT root@HOST" ...'; exit 1; }
        set -o pipefail
        for login in "$@"; do
            dir=/root/autodl-tmp/gather/$(echo "$login" | tr -c 'A-Za-z0-9.\n' '_')
            mkdir -p "$dir"
            $login "tar -C /root/autodl-tmp/ETHC-VRC -cf - outputs/paper results/paper" | tar -C "$dir" -xf - \
                || { echo "未能从 $login 拷贝结果"; exit 1; }
            merge "$dir"
        done
        ;;
    *)
        echo "usage: autodl_shard.sh run 1|2|3|4 | compare SIZE \"ssh -p PORT root@HOST\" ... | pull \"ssh -p PORT root@HOST\" ... | merge DIR"
        exit 1
        ;;
esac
