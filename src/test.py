import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Preprocessing.create_tensors import make_loaders, make_kfold_loaders

from training.train import validate, train, cross_validate
import training.BCEDiceLoss

if __name__ == "__main__":
    fold_loaders = make_kfold_loaders(k=5)
    cross_validate(fold_loaders, epochs=10)
