import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__ (
            self, weight=0.5, eps=1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight = weight
        self.eps = eps

    def forward(self, logits, target):
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * target).sum(dims)
        union = probs.sum(dims) + target.sum(dims)
        dice_loss = 1 - ((2 * inter + self.eps) / (union + self.eps)).mean()

        return self.weight * bce + (1 - self.weight) * dice_loss
