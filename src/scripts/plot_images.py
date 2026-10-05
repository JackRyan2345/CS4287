from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root = "CarDD_release/CarDD_SOD/CarDD-TR/"
img = np.array(Image.open(root + "CarDD-TR-Image/000002.jpg"))
mask = np.array(Image.open(root + "CarDD-TR-Mask/000002.png").convert("L"))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img);      ax[0].set_title("Photo")
ax[1].imshow(mask);     ax[1].set_title("Mask")
ax[2].imshow(img); ax[2].imshow(mask, cmap="Reds", alpha=0.4); ax[2].set_title("Overlay")
for a in ax: a.axis("off")
plt.show()