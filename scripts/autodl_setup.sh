#!/bin/bash
# AutoDL 服务器上的一次性环境安装（代码、Python 3.11 与 uv.lock 锁定的依赖都放在数据盘 /root/autodl-tmp）。
# 用法：bash autodl_setup.sh            （在任意目录运行；已克隆过则只更新代码）
set -e
ROOT=/root/autodl-tmp
REPO=$ROOT/ETHC-VRC
VENV=$ROOT/venv
export UV_CACHE_DIR=$ROOT/.uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/.uv-python
cd $ROOT

# 1) AutoDL 的学术资源加速，用于访问 GitHub
source /etc/network_turbo 2>/dev/null || true
if [ ! -d "$REPO" ]; then
    git clone -b paper-repro https://github.com/yeqing9349/ETHC-VRC.git "$REPO"
else
    git -C "$REPO" pull --ff-only
fi

# 2) uv 与 Python 3.11（Python 安装包来自 GitHub，失败时改用 npmmirror 镜像）
pip install -q -U uv
uv python install 3.11 || UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone uv python install 3.11
unset http_proxy https_proxy

# 3) 依赖：版本与 uv.lock 完全一致；PyPI 包走阿里云镜像，torch 走 PyTorch 官方 CUDA 12.8 源
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
cd "$REPO"
uv export --frozen --no-dev --no-hashes --format requirements-txt | grep -v '^-' > /tmp/ethc_requirements.txt
VIRTUAL_ENV="$VENV" uv pip install -r /tmp/ethc_requirements.txt \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match

"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo "环境就绪：source $VENV/bin/activate && cd $REPO"
