"""
原子写入：先写到同一目录下的临时文件并刷到磁盘，再改名替换目标文件。写到一半时断电或进程被杀，目标文件要么是上一次的
完整版本，要么还不存在，不会留下写坏的检查点或结果文件（run_all.py 以结果文件是否存在判断任务是否完成，续跑时要读检查点）。
只依赖标准库（torch、numpy 在用到时才导入），元启发式的求解子进程导入它也不会加载 PyTorch。
"""
import json
import os


def _write(path, write, mode='wb', **open_kwargs):
    tmp = path + '.tmp'
    try:
        with open(tmp, mode, **open_kwargs) as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    if os.name == 'posix':  # 改名也刷到磁盘
        try:
            fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def save_json(obj, path, **kwargs):
    _write(path, lambda f: json.dump(obj, f, **kwargs), 'w', encoding='utf-8')


def save_torch(obj, path):
    import torch
    _write(path, lambda f: torch.save(obj, f))


def savez_compressed(path, **arrays):
    """path 写全文件名（含 .npz）"""
    import numpy as np
    _write(path, lambda f: np.savez_compressed(f, **arrays))
