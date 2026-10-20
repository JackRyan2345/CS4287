import matplotlib.pyplot as plt
import torch

def plot_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("BCE+Dice loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    for key in ("iou", "f1", "precision", "recall"):
        ax_metric.plot(epochs, history[key], label=key)
    ax_metric.set_xlabel("epoch")
    ax_metric.set_ylabel("score")
    ax_metric.set_title("Validation metrics")
    ax_metric.set_ylim(0, 1)
    ax_metric.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, loader, device, n=4, threshold=0.5, save_path=None):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images[:n], masks[:n]

    logits = model(images.to(device))
    preds = (torch.sigmoid(logits) > threshold).float().cpu()

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["image", "ground truth", "prediction", "overlay"]
    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        gt = masks[i, 0].numpy()
        pr = preds[i, 0].numpy()

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 2].imshow(pr, cmap="gray")
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(pr, cmap="Reds", alpha=0.4)

        for j in range(4):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)