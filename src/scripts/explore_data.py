import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SOD = "CarDD_release/CarDD_SOD"
TR = os.path.join(SOD, "CarDD-TR")

with open(os.path.join(TR, "train_pair.lst")) as f:
    pairs = [line.split() for line in f if line.strip()]

n = 4
fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n))

for row, (img_rel, mask_rel) in enumerate(pairs[:n]):
    img = np.array(Image.open(os.path.join(TR, img_rel)))
    mask = np.array(Image.open(os.path.join(TR, mask_rel)).convert("L"))
    ax[row, 0].imshow(img)
    ax[row, 1].imshow(mask, cmap="gray")
    ax[row, 2].imshow(img); ax[row, 2].imshow(mask, cmap="Reds", alpha=0.4)

for a, t in zip(ax[0], ["Image", "Mask", "Overlay"]):
    a.set_title(t)

for a in ax.flat: a.axis("off")

plt.tight_layout(); plt.show()


widths, heights, damage = [], [], []

for img_rel, mask_rel in pairs:
    w, h = Image.open(os.path.join(TR, img_rel)).size
    widths.append(w); heights.append(h)
    mask = np.array(Image.open(os.path.join(TR, mask_rel)).convert("L"))
    damage.append((mask > 127).sum() / 100)

print("Width: ", min(widths), " - ", max(widths), " Height: ", min(heights), " - ", max(heights))
print(f"Damage per image: mean {np.mean(damage):.1f}%, median {np.median(damage):.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(widths, heights, s=4)
ax[0].set_xlabel("width")
ax[0].set_ylabel("height"); ax[0].set_title("Image sizes")
ax[1].hist(damage, bins=50)
ax[1].set_xlabel("% of pixels that are damage")
ax[1].set_ylabel("images")
ax[1].set_title("Damage per mask")

plt.tight_layout(); plt.show()

for split in ["CarDD-TR", "CarDD-VAL", "CarDD-TE"]:
    print(split, len(os.listdir(os.path.join(SOD, split, split + "-Image"))))