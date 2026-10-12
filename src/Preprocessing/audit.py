import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOD = os.path.join(ROOT, "CarDD_release", "CarDD_SOD")
TR = os.path.join(SOD, "CarDD-TR")
lst_path = os.path.join(TR, "train_pair.lst")

pairs = []
with open(lst_path) as f:
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))


df = pd.DataFrame({
    "image": [os.path.join(TR, i) for i, m in pairs],
    "mask": [os.path.join(TR, m,) for i, m in pairs],
})

df.to_csv("pairs.csv", index=False)
print(len(df), "pairs saved to", os.path.abspath("pairs.csv"))
print(df.head(3))

missing = df[~df["image"].map(os.path.exists) | ~df["mask"].map(os.path.exists)]
print("Missing files:", len(missing))