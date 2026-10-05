import json
import os
import random
import numpy as np
import torch
from pathlib import Path

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, default=str)
