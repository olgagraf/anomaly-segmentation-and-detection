import torch
import random
import numpy as np
import shutil
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config_snapshot(output_dir, filename="config_snapshot.py"):
    config_path = Path(__file__).resolve().parents[1] / "config.py"
    dst_path = Path(output_dir) / filename
    shutil.copy2(config_path, dst_path)
    return str(dst_path)
