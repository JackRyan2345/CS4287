import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from unet_model import UNET
from BCEDiceLoss import BCEDiceLoss
from metrics import confusion, scores

import torch
import torch.nn as nn

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, tp, fp, fn = 0, 0, 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += criterion(logits, masks).item()
        
        t, p, n = confusion(logits, masks)
        tp, fp, fn = tp + t, fp + p, fn + n
    return total / len(loader), scores(tp, fp, fn)

def train(train_loader, val_loader, epochs=30, save_path="unet_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_iou = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Scores: {val_score}")

        if best_iou < val_score["iou"]:
            best_iou = val_score["iou"]
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with IoU: {best_iou:.4f}")

    return model

if __name__ == "__main__":
    # Smoke test with fake data: checks the loop runs end to end, not that the model learns
    import os
    import tempfile
    from torch.utils.data import TensorDataset, DataLoader

    images = torch.rand(8, 3, 64, 96)                     # H and W must be divisible by 16
    masks = (torch.rand(8, 1, 64, 96) > 0.8).float()      # binary 0/1, about 20% damage
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=True)

    save_path = os.path.join(tempfile.gettempdir(), "smoke_test_unet.pth")
    train(loader, loader, epochs=30, save_path=save_path)
    os.remove(save_path)
    print("Smoke test passed")
