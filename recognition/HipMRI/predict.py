# Showing example usage of your trained model. It should load your saved model weights, run inference
# on test cases, print out any results, and provide visualisations where applicable (e.g., prediction overlays, generated
# samples, heatmaps). NumPy is allowed in this script for visualisation or loading data.

import torch

from dataset import get_pairs, split_pairs, get_loaders
from modules import Improved3DUNet
from train import num_classes, class_names, base_filters, dice_score

data_root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data"
model_path = "3D_Model/unet3d_hipmri.pt"

def main(root=data_root):
    # test split only, same seed as training so these patients were never seen
    splits = split_pairs(get_pairs(root))
    test_loader = get_loaders({"test": splits["test"]})["test"] # batch size 1, one volume at a time
    print(f"Test volumes: {len(splits['test'])}")

    model = Improved3DUNet(in_channels=1, num_classes=num_classes, base_filters=base_filters)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # dice per class for each volume, then averaged over volumes
    total_dice = torch.zeros(num_classes)
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            total_dice += dice_score(logits, labels.long(), num_classes)
    test_dice = total_dice / len(test_loader)

    for name, d in zip(class_names, test_dice):
        print(f"Test DSC {name}: {d:.4f}")
    print(f"Min class DSC: {test_dice.min():.4f} | Mean DSC: {test_dice.mean():.4f}")

if __name__ == "__main__":
    main()
