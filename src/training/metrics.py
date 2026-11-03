import torch

# counts true positives, false positives and false negatives over all the pixels (see 4.1)
def confusion(out, target, threshold=0.5):
    preds = (torch.sigmoid(out) > threshold).float()    # turns logits into a 0/1 mask, 0.5 cut off picked in the sweep (see 4.5)

    tp = (preds * target).sum()                         # said damage, was damage
    fp = (preds * (1 - target)).sum()                   # said damage, wasnt damage
    fn = ((1 - preds) * target).sum()                   # missed damage

    return tp, fp, fn                                   # dont need true negatives, thats why we dont use pixel accuracy (see 4.1)

# works out the scores from the counts, formulas are in 4.1
def scores(tp, fp, fn, eps=1e-7):                               # eps is epsilon, fixed errors dividing by zero
    return {
        "f1": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item(),         # same as the dice score (see 4.1)
        "precision": ((tp + eps) / (tp + fp + eps)).item(),               # how much of what we predicted was actually damage
        "recall": ((tp + eps) / (tp + fn + eps)).item(),                  # how much of the real damage we found
        "iou": ((tp + eps) / (tp + fp + fn + eps)).item()         # iou is intersection over union, the overlap between the predicted and true masks, our main metric (see 4.1)
    }

if __name__ == "__main__":
    # quick check on a 4x4 mask, top 2 rows are damage
    masks = torch.zeros(1, 1, 4, 4)
    masks[..., :2, :] = 1

    perfect = masks * 20 - 10                           # perfect prediction, should all be 1
    print(scores(*confusion(perfect, masks)))

    half = torch.full_like(masks, -10)                  # only gets the first row, recall and iou should be 0.5
    half[..., 0, :] = 10
    print(scores(*confusion(half, masks)))