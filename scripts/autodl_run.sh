#!/bin/bash
# AutoDL 上的正式运行：在实例的全部显卡上运行 run_all.py（每卡同时运行的任务数：n=20、50 为 3 个，n=100 为 2 个），
# 运行结束后关机（按量计费随之停止）。中断后重新执行同一条命令即从断点继续。
# 用法（在任意目录）：
#   nohup bash /root/autodl-tmp/ETHC-VRC/scripts/autodl_run.sh >> /root/autodl-tmp/run_paper.log 2>&1 &
# 额外参数原样传给 run_all.py（如 --match REGEX）；环境变量 NO_SHUTDOWN=1 时结束后不关机；PYTHON 可指定解释器。
# 关机的条件：运行结束——全部完成、有任务失败、或有任务因硬件不符暂未运行（退出码 0/1/3）。以下情况不关机：
# 已有运行在进行（2）、本机没有可运行的任务（4，例如无卡模式）、被 kill 或 Ctrl-C 中断（130）、带 --dry-run。
cd "$(dirname "$0")/.." || exit 1
PYTHON=${PYTHON:-/root/autodl-tmp/venv/bin/python}
ARGS=(--config configs/autodl.yaml --gpus all --tasks-per-gpu 20:3,50:3,100:2 "$@")
echo "===== $(date '+%F %T') start: run_all.py ${ARGS[*]}"
"$PYTHON" run_all.py "${ARGS[@]}"
code=$?
echo "===== $(date '+%F %T') run_all.py exited with code $code"
case " $* " in
    *" --dry-run "*) NO_SHUTDOWN=1 ;;
esac
case $code in
    2|4|130) exit $code ;;
esac
if [ "${NO_SHUTDOWN:-0}" != "1" ]; then
    echo "===== $(date '+%F %T') shutting down"
    /usr/bin/shutdown
fi
exit $code
