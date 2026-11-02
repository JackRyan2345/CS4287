import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Preprocessing.create_tensors import make_loaders, make_fold_loaders

from training.train import validate, train, cross_validate, test
import training.BCEDiceLoss

if __name__ == "__main__":
    #fold_loaders = make_fold_loaders(k=5)
    #cross_validate(fold_loaders, epochs=10)

    _, _, test_loader = make_loaders(batch_size=6)
    test(test_loader, model_path="unet_fold1.pth")