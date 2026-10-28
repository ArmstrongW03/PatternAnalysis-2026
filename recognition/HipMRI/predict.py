# Showing example usage of your trained model. It should load your saved model weights, run inference
# on test cases, print out any results, and provide visualisations where applicable (e.g., prediction overlays, generated
# samples, heatmaps). NumPy is allowed in this script for visualisation or loading data.

import torch
import matplotlib.pyplot as plt

import UNet2D
from dataset import get_pairs, split_pairs, get_loaders
from modules import Improved3DUNet
from train import num_classes, class_names, base_filters, dice_score

data_root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data"
is_3d = False # False tests the 2D UNet instead
out_dir = "3D_Model" if is_3d else "2D_Unet_output"
model_name = "unet3d" if is_3d else "unet2d"
model_path = f"{out_dir}/{model_name}_hipmri.pt"
PROSTATE = 5
end_slices = 5 # how many slices at each end of the prostate count as the apex / base
num_examples = 3 # how many test volumes to save a picture of

def predict_2d(model, images):
    # runs each slice through the 2D model, then stacks them back into a volume shaped like the 3D model's output
    slices = images[0].permute(3, 0, 1, 2) # (1, 1, H, W, D) -> (D, 1, H, W), each slice is one image in the batch
    logits = torch.cat([model(chunk) for chunk in slices.split(16)]) # 16 slices at a time so it fits in memory
    return logits.permute(1, 2, 3, 0).unsqueeze(0) # (D, C, H, W) -> (1, C, H, W, D)

def dice_and_iou(pred, target):
    # overlap of two boolean masks, a region missing from both counts as perfect
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    total = pred.sum().item() + target.sum().item()
    dice = 2 * intersection / total if total > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return dice, iou

def prostate_regions(preds, labels):
    # dice and iou for the whole prostate, and for its first / last few slices
    # slices run along the last axis, the lowest slices are the apex (inferior) and the highest are the base (next to the bladder)
    pred = preds == PROSTATE # (H, W, D)
    target = labels == PROSTATE
    prostate_slices = target.any(dim=0).any(dim=0).nonzero().flatten() # slices where the ground truth has prostate
    apex = prostate_slices[:end_slices]
    base = prostate_slices[-end_slices:]
    return {
        "whole": dice_and_iou(pred, target),
        "apex": dice_and_iou(pred[..., apex], target[..., apex]),
        "base": dice_and_iou(pred[..., base], target[..., base]),
    }

def save_example(image, label, pred, save_path):
    # one volume, rows are the apex / middle / base prostate slices, columns are input MRI | ground truth | prediction
    prostate_slices = (label == PROSTATE).any(dim=0).any(dim=0).nonzero().flatten()
    rows = {
        "Apex": prostate_slices[0].item(), # first slice with prostate
        "Middle": prostate_slices[len(prostate_slices) // 2].item(),
        "Base": prostate_slices[-1].item(), # last slice with prostate
    }

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, (region, k) in enumerate(rows.items()):
        axes[row, 0].imshow(image[..., k], cmap="gray")
        axes[row, 1].imshow(label[..., k], cmap="viridis", vmin=0, vmax=num_classes - 1) # fixed range so each class keeps the same colour
        axes[row, 2].imshow(pred[..., k], cmap="viridis", vmin=0, vmax=num_classes - 1)
        axes[row, 0].set_title(f"{region} input (slice {k})")
        axes[row, 1].set_title(f"{region} ground truth")
        axes[row, 2].set_title(f"{region} prediction")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"saved {save_path}")

def main(root=data_root):
    # test split only, same seed as training so these patients were never seen
    splits = split_pairs(get_pairs(root))
    test_loader = get_loaders({"test": splits["test"]})["test"] # batch size 1, one volume at a time
    print(f"Test volumes: {len(splits['test'])}")

    if is_3d:
        model = Improved3DUNet(in_channels=1, num_classes=num_classes, base_filters=base_filters)
    else:
        model = UNet2D.UNet(1, num_classes, UNet2D.base_filters) # the 2D model uses its own base_filters (32), not the 3D one
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # dice per class for each volume, then averaged over volumes
    total_dice = torch.zeros(num_classes)
    region_scores = {"whole": [], "apex": [], "base": []} # (dice, iou) per volume for each prostate region
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            logits = model(images) if is_3d else predict_2d(model, images)
            total_dice += dice_score(logits, labels.long(), num_classes)

            preds = logits.argmax(dim=1)[0] # (H, W, D), batch size is 1
            for region, score in prostate_regions(preds, labels[0]).items():
                region_scores[region].append(score)

            # pictures of the first few test volumes, [0, 0] drops the batch and channel dims
            if i < num_examples:
                save_example(images[0, 0], labels[0], preds, f"{out_dir}/predictions/{model_name}_prediction_{i}.png")
    test_dice = total_dice / len(test_loader)

    for name, d in zip(class_names, test_dice):
        print(f"Test DSC {name}: {d:.4f}")
    print(f"Min class DSC: {test_dice.min():.4f} | Mean DSC: {test_dice.mean():.4f}")

    # mean over volumes, so the apex and base can be compared against the whole prostate
    print(f"\nProstate by region ({end_slices} slices at each end):")
    for region, scores in region_scores.items():
        dice, iou = torch.tensor(scores).mean(dim=0)
        print(f"{region:>5}: DSC {dice:.4f} | IoU {iou:.4f}")

if __name__ == "__main__":
    main()
