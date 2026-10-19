# Containing the source code for training, validating, testing, and saving your model. 
# The model should be imported from modules.py and the data loader should be imported from dataset.py. 
# Make sure to plot the losses and metrics during training.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    # multiclass soft dice loss

    def __init__(self, num_classes, smooth=1e-5):
        # store num_classes and the small smoothing constant
        super().__init__()

    def forward(self, logits, target):
        # logits: (batch, num_classes, D, H, W) raw network output
        # target: (batch, D,H,W) integer labels
        # softmax the logits. one-hot encode target. Per class dice = 2*sum(pred*target)/(sum(pred)+sum(target)).
        # return 1 - mean dice over classes
        pass

def dice_score(logits, target, num_classes):
    # Hard Dice per class for reporting (not traiing), computes dice for each class
    pass

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # Train one pass over loader. Move to device -> forward -> loss -> backward -> step. Return avg loss
    pass

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