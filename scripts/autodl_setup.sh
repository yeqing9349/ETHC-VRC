#!/bin/bash
# AutoDL 服务器上的一次性环境安装（代码、Python 3.11 与 uv.lock 锁定的依赖都放在数据盘 /root/autodl-tmp）。
# 用法：bash autodl_setup.sh            （在任意目录运行；已克隆过则只更新代码）
set -e
ROOT=/root/autodl-tmp
REPO=$ROOT/ETHC-VRC
VENV=$ROOT/venv
WHEELS=$ROOT/wheels
export PATH=/root/miniconda3/bin:$PATH
export UV_CACHE_DIR=$ROOT/.uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/.uv-python
cd $ROOT

# 1) 代码：经 AutoDL 的学术资源加速访问 GitHub；已有仓库时尝试更新，失败则沿用现有代码
if [ ! -d "$REPO" ]; then
    (source /etc/network_turbo >/dev/null 2>&1 || true; git clone -b paper-repro https://github.com/yeqing9349/ETHC-VRC.git "$REPO")
else
    (source /etc/network_turbo >/dev/null 2>&1 || true; git -C "$REPO" pull --ff-only) || echo "未能从 GitHub 更新，使用现有代码"
fi

# 2) uv 与 Python 3.11（Python 安装包来自 GitHub，失败时改用 npmmirror 镜像）
pip install -q -U uv
(source /etc/network_turbo >/dev/null 2>&1 || true; uv python install 3.11) \
    || UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone uv python install 3.11

# 3) 依赖：版本与 uv.lock 完全一致。国内访问 download.pytorch.org 很慢，torch（CUDA 12.8 版）改从阿里云的 PyTorch 镜像下载，
#    并用 uv.lock 中记录的 sha256 校验；其余包走阿里云 PyPI 镜像
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
cd "$REPO"
TORCH_URL_NAME=$(grep -o 'torch-[0-9.]*%2Bcu128-cp311-cp311-manylinux_2_28_x86_64\.whl' uv.lock | head -1)
TORCH_SHA=$(grep "$TORCH_URL_NAME" uv.lock | grep -o 'sha256:[0-9a-f]*' | head -1 | cut -d: -f2)
TORCH_WHL=$(echo "$TORCH_URL_NAME" | sed 's/%2B/+/')
mkdir -p "$WHEELS"
if [ ! -f "$WHEELS/$TORCH_WHL" ]; then
    curl -fL --retry 3 -o "$WHEELS/$TORCH_WHL.part" "https://mirrors.aliyun.com/pytorch-wheels/cu128/$TORCH_WHL"
    mv "$WHEELS/$TORCH_WHL.part" "$WHEELS/$TORCH_WHL"
fi
echo "$TORCH_SHA  $WHEELS/$TORCH_WHL" | sha256sum -c -
uv export --frozen --no-dev --no-hashes --format requirements-txt | grep -v '^-' > /tmp/ethc_requirements.txt
VIRTUAL_ENV="$VENV" uv pip install -r /tmp/ethc_requirements.txt \
    --index-url https://mirrors.aliyun.com/pypi/simple/ --find-links "$WHEELS" --index-strategy unsafe-best-match

"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
echo "环境就绪：source $VENV/bin/activate && cd $REPO"
