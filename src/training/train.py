import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from unet_model import UNET

import torch
import torch.nn as nn

import time
import statistics
from BCEDiceLoss import BCEDiceLoss
from metrics import confusion, scores

from torch.utils.data import DataLoader

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

    history = {"train_loss": [], "val_loss": [], "iou": [], "f1": [], "precision": [], "recall": []}
    best_iou = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in ("iou", "f1", "precision", "recall"):
            history[k].append(val_score[k])

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Scores: {val_score}")

        if best_iou < val_score["iou"]:
            best_iou = val_score["iou"]
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with IoU: {best_iou:.4f}")

    return model, history

def cross_validate(fold_loaders, epochs=10):
    fold_scores = []

    for i, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n===== Fold {i+1}/{len(fold_loaders)} =====")
        save_path = f"unet_fold{i+1}.pth"
        model, _ = train(train_loader, val_loader, epochs=epochs, save_path=save_path)

        model.load_state_dict(torch.load(save_path))
        device = next(model.parameters()).device
        _, best = validate(model, val_loader, BCEDiceLoss(), device)
        fold_scores.append(best)

    for key in ("iou", "f1", "precision", "recall"):
        values = [s[key] for s in fold_scores]
        print(f"{key}: {statistics.mean(values):.4f} +- {statistics.stdev(values):.4f}")

    return fold_scores

if __name__ == "__main__":
    # Smoke test with fake data: checks the loop runs end to end, not that the model learns
    import os
    import tempfile
    from torch.utils.data import TensorDataset, DataLoader
    from plot import plot_history, plot_predictions

    images = torch.rand(8, 3, 64, 96)                     # H and W must be divisible by 16
    masks = (torch.rand(8, 1, 64, 96) > 0.8).float()      # binary 0/1, about 20% damage
    loader = DataLoader(TensorDataset(images, masks), batch_size=4, shuffle=True)

    # save_path = os.path.join(tempfile.gettempdir(), "smoke_test_unet.pth")
    # train(loader, loader, epochs=30, save_path=save_path)
    # os.remove(save_path)
    # print("Smoke test passed")

    model, history = train(loader, loader, epochs=30)
    plot_history(history, save_path="training_curves.png")
    plot_predictions(model, loader, device="cuda" if
                     torch.cuda.is_available() else "cpu",
                     n=4, save_path="predictions.png")

