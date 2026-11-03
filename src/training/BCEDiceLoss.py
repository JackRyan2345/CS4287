import torch
import torch.nn as nn

# BCE + Dice loss mixed together, dice helps since most of the image isnt damage (see 3.1, other options we looked at in 3.2)
class BCEDiceLoss(nn.Module):
    def __init__ (
            self, weight=0.5, eps=1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()           # does the sigmoid itself so we pass in raw logits, the model has no sigmoid at the end
        self.weight = weight                        # how much of the loss is BCE, the rest is dice, 0.5 so neither dominates (see 3.1)
        self.eps = eps                              # stops dividing by zero when a mask is empty

    def forward(self, logits, target):
        bce = self.bce(logits, target)              # per pixel loss, gives every pixel a small push so training starts smooth (see 3.1)

        probs = torch.sigmoid(logits)               # logits to 0-1 probabilities for dice
        dims = (1, 2, 3)                            # sum over channel, height and width so we get a score per image
        inter = (probs * target).sum(dims)          # overlap between prediction and mask
        union = probs.sum(dims) + target.sum(dims)  # size of the prediction and the mask added together
        dice_loss = 1 - ((2 * inter + self.eps) / (union + self.eps)).mean()   # 1 - dice, so the lazy all background model gets the worst score (see 3.1)

        return self.weight * bce + (1 - self.weight) * dice_loss  # mix of the two, half each by default
