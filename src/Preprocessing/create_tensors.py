from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent           # .../src/scripts/Preprocessing
REPO_ROOT = BASE_DIR.parents[2]                      # .../CS4287
input_dir = REPO_ROOT / "Output"

class CarDDDataset:

    def __init__(self, root):
        