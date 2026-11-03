import sys
from pathlib import Path
from tracemalloc import start
# so we can import from src/ and src/training/ wherever its run from
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

# one pass over the training data, returns the average loss which is one point on the curves (see 4.2)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                                       # training mode, batchnorm uses the stats from the current batch
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)    # batch has to be on the same device as the model

        logits = model(images)                          # forward pass, raw logits for every pixel
        loss = criterion(logits, masks)                 # how far off the prediction is from the real mask

        optimizer.zero_grad()                           # clear the old gradients
        loss.backward()                                 # backprop, works out the gradient for every weight
        optimizer.step()                                # adam updates the weights using those gradients (see 3.3)

        running_loss += loss.item()                     # .item() so we only keep the number, not the whole graph
    return running_loss / len(loader)                   # average loss per batch

# runs the model on a loader without training it, returns the average loss and the scores
# tp, fp and fn are added up over every batch first so every pixel counts the same (see 4.1)
@torch.no_grad()                                        # no gradients needed here, saves memory and its faster
def validate(model, loader, criterion, device):
    model.eval()                                        # eval mode, batchnorm uses the running mean and variance from training
    total, tp, fp, fn = 0, 0, 0, 0

    start = time.time()
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += criterion(logits, masks).item()        # add up the loss so we can average it after

        t, p, n = confusion(logits, masks)              # pixel counts for this batch
        tp, fp, fn = tp + t, fp + p, fn + n             # add this batch onto the totals
        print(f"batch {i+1}/{len(loader)}  {time.time() - start:.1f}s")    # shows progress
    return total / len(loader), scores(tp, fp, fn)     # average loss and the scores for the whole set

# trains a new model, checks it on the val set every epoch and saves it when the iou gets better (see 3.4)
# returns the model from the last epoch, the best one is in save_path
def train(train_loader, val_loader, epochs=30, save_path="unet_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"     # use the gpu if there is one
    model = UNET().to(device)                           # fresh untrained unet
    criterion = BCEDiceLoss()                           # BCE + Dice loss from above (see 3.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # adam gives each weight its own step size, constant 1e-4 and no weight decay (see 3.3)

    history = {"train_loss": [], "val_loss": [], "iou": [], "f1": [], "precision": [], "recall": []}   # kept for the training curves (see 4.2)
    best_iou = 0.0                                      # best val iou so far
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)    # learn from the training set
        val_loss, val_score = validate(model, val_loader, criterion, device)    # then see how it does on data it hasnt trained on

        history["train_loss"].append(train_loss)        # save this epochs numbers for the plots
        history["val_loss"].append(val_loss)
        for k in ("iou", "f1", "precision", "recall"):
            history[k].append(val_score[k])

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Scores: {val_score}")

        # only save if iou beat the best so far, iou is what we pick the checkpoint on (see 4.1, 4.7)
        if best_iou < val_score["iou"]:
            best_iou = val_score["iou"]
            torch.save(model.state_dict(), save_path)   # saves just the weights, not the whole model
            print(f"Model saved with IoU: {best_iou:.4f}")

    return model, history

# loads a saved model and runs it on the test set, only done once at the end (see 4.4)
def test(train_loader, model_path="unet_fold1.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET().to(device)                           # same architecture so the saved weights fit
    model.load_state_dict(torch.load(model_path, map_location=device))   # map_location so it loads on cpu too
    print(f"Running model on {device}")

    criterion = BCEDiceLoss()
    test_loss, test_score = validate(model, train_loader, criterion, device)    # same as validating but on the test set, never used for training
    print(f"Test Loss: {test_loss:.4f}, Test Scores: {test_score}")

# k-fold cross validation, trains a brand new model for each fold and prints the average scores at the end (see 3.4)
def cross_validate(fold_loaders, epochs=10):
    fold_scores = []

    for i, (train_loader, val_loader) in enumerate(fold_loaders):    # each fold uses a different chunk as the val set
        print(f"\n===== Fold {i+1}/{len(fold_loaders)} =====")
        save_path = f"unet_fold{i+1}.pth"               # separate file for each fold
        model, _ = train(train_loader, val_loader, epochs=epochs, save_path=save_path)

        model.load_state_dict(torch.load(save_path))    # load the best model back in, train() gives back the last one
        device = next(model.parameters()).device
        _, best = validate(model, val_loader, BCEDiceLoss(), device)    # scores of the best model for this fold
        fold_scores.append(best)

    # mean and standard deviation across the folds
    for key in ("iou", "f1", "precision", "recall"):
        values = [s[key] for s in fold_scores]
        print(f"{key}: {statistics.mean(values):.4f} +- {statistics.stdev(values):.4f}")   # small std means the model does about the same on every fold (see 4.3)

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

