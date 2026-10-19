# Containing the source code for training, validating, testing, and saving your model. 
# The model should be imported from modules.py and the data loader should be imported from dataset.py. 
# Make sure to plot the losses and metrics during training.

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_data_3D, get_pairs, split_pairs
from modules import Improved3DUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("WARNING: CUDA unavailable")

# find fastest conv algo for input shape
torch.backends.cudnn.benchmark = True
# tensor float 32 on newer GPUs, speedup barely any accuracy loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# hyperparams
data_root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data"
model_name = "unet3d_hipmri.pt"
num_classes = 6
class_names = ("Background", "Body", "Bone", "Bladder", "Rectum", "Prostate")
base_filters = 16
batch_size = 2 # only able to fit a few whole volumes on GPU at once
num_epochs = 30
lr = 5e-4
weight_decay = 1e-5
lr_gamma = 0.985 # lr multiplied by this every epoch

class DiceLoss(nn.Module):
    # multiclass soft dice loss

    def __init__(self, num_classes, smooth=1e-5):
        # store num_classes and the small smoothing constant
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (batch, num_classes, D, H, W) raw network output
        # target: (batch, D,H,W) integer labels
        # softmax the logits. one-hot encode target. Per class dice = 2*sum(pred*target)/(sum(pred)+sum(target)).
        # return 1 - mean dice over classes
        probs = F.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float() # (batch, D, H, W, B) -> (batch, C, D, H, W)
        dims = (0, 2, 3, 4) # sum over batch and voxels, keeping each class separate
        intersection = (probs * one_hot).sum(dims)
        total = probs.sum(dims) + one_hot.sum(dims)
        dice = (2*intersection+self.smooth)/(total+self.smooth)
        return 1 - dice.mean()# every class counts equally, so small prostate class isnt drowned out by background or body

def dice_score(logits, target, num_classes):
    # Hard Dice per class for reporting (not traiing), computes dice for each class
    preds = logits.argmax(dim=1) # most likely class per voxel
    classes = torch.arange(num_classes, device=preds.device).view(-1, 1, 1, 1, 1) # compares againstevery class at once, same as onehot
    pred_one_hot = preds.unsqueeze(0) == classes # (C, batch, D, H, W)
    target_one_hot = target.unsqueeze(0) == classes
    dims = (1, 2, 3, 4)
    intersection = (pred_one_hot & target_one_hot).sum(dims).float()
    total = (pred_one_hot.sum(dims) + target_one_hot.sum(dims)).float()
    # a class missing from both the prediction and label counts as perfect instead of Nan
    return torch.where(total > 0, 2 * intersection / total.clamp(min=1), torch.ones_like(total))

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    # Train one pass over loader. Move to device -> forward -> loss -> backward -> step. Return avg loss
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long() # onehot needs int64
        optimizer.zero_grad(set_to_none=True)

        # mized precision, halves the memory
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        loss = loss_fn(logits, labels)

        # scale the loss up so fp16 gradients dont underflow, then unscale before the optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach()
    return (running_loss / len(loader)).item()

def validate(model, loader, loss_fn, device, num_classes):
    # evaluate on validation set (model.eval, torch.nograd). Avg loss, and avg dice per class
    pass

def plot_curves(train_losses, val_losses, val_dice, save_path="training_curves.png"):
    # plot loss and dice against epoch and save the figure. If on rangpur, print values out so can plot later.
    pass

def main():
    """
        Set up and run training

        1. BUild the train an validation dataloaders
        2. Build improved unet 3d and move to gpu
        3. loss = diceloss; optimizer = adam(lr=5e-4, weight_decay=1e-5); scheduler=exponentialLr(gamma=0.985).
        4. For each epoch: train_one_epoch->validate-> scheduler.step()->save model if val dice is best so far.
        5. Plot curves at end.
    """
    pass

if __name__ == "__main__":
    main()