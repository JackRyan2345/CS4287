import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(__file__)
pairs = pd.read_csv(os.path.join(here, "pairs.csv"))

rows = []

for r in pairs.itertuples():
    try:
        img, msk = Image.open(r.image), Image.open(r.mask)
        m = np.array(msk)
        rows.append(dict(image = r.image, mask = r.mask, ok = True,
                        img_size = img.size, mask_size = msk.size,
                        img_mode = img.mode, mask_vals = tuple(np.unique(m)),
                        damage_frac = float((m > 0).mean())))
    except Exception as e:
        rows.append(dict(image = r.image, mask = r.mask, ok = False, error = str(e)))

audit = pd.DataFrame(rows)
audit.to_csv(os.path.join(here, "audit.csv"), index = False)

good = audit[audit.ok]

print("Pairs:", len(audit), "| Unreadable:", (~audit.ok).sum())
print("Size mismatches:", (good.img_size != good.mask_size).sum())
print("Image sizes:\n", good.img_size.value_counts().head())
print("Mask value sets:\n", good.mask_vals.value_counts().head())
print("Empty masks:", (good.damage_frac == 0).sum())
print(good.damage_frac.describe())