# Containing the source code of the components of your model. Each component must be implemented as a class or a funciton in PyTorch or TensorFlow.
# Your implementation of the model should not depend on NumPy (or other python libraries not part of Pytorch / tensorFlow) in any way unless otherwise
# approved by teaching staff.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    # Encoder block: two 3x3x3 convs with dropout in between

    def __init__(self, channels, dropout=0.3):
        # instanceNorm -> leakyrelu -> 3x3x3 conv -> dropout -> instancenorm -> leakyrelu -> 3x3x3 conv (channel count stays same)
        super().__init__()

    def forward(self, x):
        # run x through layers and return result
        pass

class UpsampleBlock(nn.Module):
    # Decoder step (doubles spatial size)

    def __init__(self, in_channels, out_channels):
        # upsample scale 2-> 3x3x3 conv (in -> out channels) -> instance norm -> leaky relu
        super().__init__()

    def forward(self, x):
        # upsample x by 2 and convolve to out_channels
        pass

class LocalisationBlock(nn.Module):
    # decoder run after concatting with skip connection

    def __init__(self, in_channels, out_channels):
        # 3x3x3 convs (keeps channels) -> instance norm -> leakyrelu -> 1x1x1 conv (in->out chans) -> instance norm -> leakyrelu
        super().__init__()

    def forward(self, x):
        # combine concat features and reduce chan count
        pass

class Improved3DUNet(nn.Module):
    # the full improved 3d unet

    def __init__(self, in_channels=4, num_classes=4, base_filters=16):
        # encoder (5 levels, filters 16, 32, 64, 128, 256): each lvl = 3x3x3 conv (stride 2 except lvl 1) + res block
        # decoder (lvls 4 -> 1): upsamp -> concat skip -> localisation block (lvl 1 ends with a 3x3x3 conv instead)
        # segmentation layers: 1x1x1 convs to num_classes at decoder lvls 3, 2, 1

        super().__init__()

    def forward(self, x):
        # encoder: at each lvl, out = conv(x); out = out + res(out); save out as a skip connection
        # decoder: upsamp, concat with skip, localise
        # deep supervision: upscale seg out from lvl3, add to lvl2, upscale, add to lvl1.

        pass


