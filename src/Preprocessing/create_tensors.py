from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import sys

torch.set_printoptions(threshold=sys.maxsize, linewidth=250)

BASE_DIR = Path(__file__).resolve().parent           # .../src/scripts/Preprocessing
#REPO_ROOT = BASE_DIR.parents[2]                      # .../CS4287
input_dir = BASE_DIR / "Output"

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.ToTensor()
])

img = Image.open(input_dir / "CarDD-TE" / "CarDD-TE-Image" / "000012.jpg").convert("RGB")
msk = Image.open(input_dir  / "CarDD-TE" / "CarDD-TE-Mask" / "000012.png").convert("L")

image_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255
mask_tensor = torch.from_numpy(np.array(msk)).unsqueeze(0).float()

mask_tensor = (mask_tensor > 127).float()

pair = (image_tensor, mask_tensor)

#tensor = data_transform(img)

#print(pair)

print(image_tensor.shape, mask_tensor.shape)   # [3, 576, 384] and [1, 576, 384]
print(image_tensor.min(), image_tensor.max())  # 0.0 and something near 1.0
print(mask_tensor.unique())                    # tensor([0., 1.]) means some damage pixels exist
print(mask_tensor.sum())                       # number of mask pixels, should be > 0

with open(BASE_DIR / "tensor_dump.txt", "w") as f:
    f.write("IMAGE\n" + str(image_tensor) + "\n\nMASK\n" + str(mask_tensor))