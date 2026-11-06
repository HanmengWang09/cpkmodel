"""将运行临时文件和缓存放在模型目录的runs下。"""
import os
from pathlib import Path
import sys


def configure_runtime():
    root = Path(__file__).resolve().parent.parent / 'runs' / 'runtime'
    paths = {'TEMP': root, 'TMP': root, 'MPLCONFIGDIR': root / 'matplotlib',
             'NUMBA_CACHE_DIR': root / 'numba', 'PIP_CACHE_DIR': root / 'pip-cache',
             'PYTHONPYCACHEPREFIX': root / 'pycache'}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.pycache_prefix = str(paths['PYTHONPYCACHEPREFIX'])
    return root
