import torch

def confusion(out, target, threshold=0.5):
    preds = (torch.sigmoid(out) > threshold).float()

    tp = (preds * target).sum()
    fp = (preds * (1 - target)).sum()
    fn = ((1 - preds) * target).sum()

    return tp, fp, fn

def scores(tp, fp, fn, eps=1e-7):                               # eps is epsilon, fixed errors dividing by zero
    return {
        "f1": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item(),
        "precision": ((tp + eps) / (tp + fp + eps)).item(),
        "recall": ((tp + eps) / (tp + fn + eps)).item(),
        "iou": ((tp + eps) / (tp + fp + fn + eps)).item()         # iou is intersection over union, the overlap between the predicted and true masks
    }

if __name__ == "__main__":
    masks = torch.zeros(1, 1, 4, 4)
    masks[..., :2, :] = 1

    perfect = masks * 20 - 10
    print(scores(*confusion(perfect, masks)))

    half = torch.full_like(masks, -10)
    half[..., 0, :] = 10
    print(scores(*confusion(half, masks)))