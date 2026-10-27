from pathlib import Path
import random
from torch.utils.data import Subset, ConcatDataset
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import sys
from sklearn.model_selection import KFold


BASE_DIR = Path(__file__).resolve().parent           # .../src/scripts/Preprocessing
input_dir = BASE_DIR / "Output"

class CarDDDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.names = sorted(p.stem for p in self.image_dir.iterdir()
                            if p.suffix.lower() in (".jpg", ".png"))
        self.augment = augment
        
    def __len__(self):
        return len(self.names)
    
    def _find(self, folder, stem):
        for ext in (".jpg", ".png"):
            p = folder / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"{stem} not found in {folder}")
    
    def __getitem__(self, idx):
        name = self.names[idx]
        with Image.open(self._find(self.image_dir, name)) as f:
            img = f.convert("RGB")
        with Image.open(self._find(self.mask_dir, name)) as f:
            msk = f.convert("L")

        image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = (torch.from_numpy(np.array(msk)).unsqueeze(0) > 127).float()
        if self.augment:
            chance = random.random()
            if chance < 0.25:
                image = torch.rot90(image, k=2, dims=(1, 2))
                mask = torch.rot90(mask, k=2, dims=(1, 2))

        return image, mask
    

def split(name, augment=False):
        return CarDDDataset(
            input_dir / f"CarDD-{name}" / f"CarDD-{name}-Image",
            input_dir / f"CarDD-{name}" / f"CarDD-{name}-Mask",
            augment=augment
        )

def make_loaders(batch_size = 6):
    
    train_ds, val_ds, test_ds = split("TR"), split("VAL"), split("TE")
    return (
        DataLoader(train_ds, batch_size = batch_size, shuffle=True),
        DataLoader(val_ds, batch_size = batch_size, shuffle=False),
        DataLoader(test_ds, batch_size = batch_size, shuffle=False)
    )

def make_fold_loaders(k = 5, batch_size = 6):
    train_pool = ConcatDataset([split("TR", augment=True), split("VAL", augment=True)])
    val_pool = ConcatDataset([split("TR"), split("VAL")])

    kf = KFold(n_splits = k, shuffle=True, random_state=42)
    fold_loaders = []

    for train_idx, val_idx in kf.split(range(len(train_pool))):
        train_subset = Subset(train_pool, train_idx)
        val_subset = Subset(val_pool, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
    
if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_loaders()
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    images, masks = next(iter(train_loader))
    print(images.shape, masks.shape)
    print(masks.unique())

