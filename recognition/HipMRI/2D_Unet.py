import os
import json
import time

import numpy as np
# no matplotlib on the GPU machine, so images are saved with torchvision and the plot data is printed / written to json instead
# import matplotlib
# matplotlib.use('Agg') # no display on the cluster, so just save the plots to files
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataset import load_data_3D, get_pairs, split_pairs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# find fastest algorithm for the first batch of each shape (keeps the fastest)
torch.backends.cudnn.benchmark = True

# lets float32 convolutions and matrix mults use tensor float 32 on newer GPUs (basically no accuracy loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# hyperparameters
demo = False # if true, loads the saved model instead of training
local = False # true when running on my computer (CPU), false on rangpur
data_root = '/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data' # holds semantic_MRs_anon and semantic_labels_anon, change this when running on rangpur
model_name = 'unet2d_hipmri.pt'
num_classes = 6
class_names = ('Background', 'Body', 'Bone', 'Bladder', 'Rectum', 'Prostate')
base_filters = 32 # filters in the first stage, doubled at every downsample
batch_size = 16 if local else 32
num_epochs = 2 if local else 25
max_lr = 1e-3 # peak learning rate (OneCycle moves this up and down)
num_workers = 4 if local else 8 # number of cpu processes loading


## HipMRI volumes, loaded once as 3D so the 2D and 3D models can be scored on the same volumes
def load_split(pairs):
    images = load_data_3D([mr for mr, _ in pairs], normImage=True) # (N, 256, 256, 128), z-score normalised per volume
    labels = load_data_3D([seg for _, seg in pairs], dtype=np.uint8) # (N, 256, 256, 128), labels 0-5
    return images, labels

## 2D slice view over the 3D volumes
class HipMRISliceDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, augment=False):
        # indexes (volume, slice) instead of reshaping the volumes into slices, so no second copy is made in memory
        self.images, self.labels = images, labels
        self.depth = images.shape[3]
        self.augment = augment

    def __len__(self):
        return self.images.shape[0] * self.depth

    def __getitem__(self, idx):
        # slice k of volume v, the image and label slice come from the same place so they always line up
        v, k = divmod(idx, self.depth)
        image = torch.from_numpy(self.images[v, :, :, k].copy()).unsqueeze(0) # (1, 256, 256), conv layers need a channel dimension
        mask = torch.from_numpy(self.labels[v, :, :, k].astype(np.int64)) # (256, 256), one label per pixel

        # the same flip is applied to the image and mask so they still line up, for the horizontal flip augmentation of the dataset
        if self.augment and torch.rand(1).item() < 0.5:
            image = image.flip(-1)
            mask = mask.flip(-1)
        return image, mask

def make_loader(dataset, shuffle, workers):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle, # random order each epoch for training, so model cant learn order
                                       num_workers=workers,
                                       pin_memory=(device.type == 'cuda'), # batches go into pinned RAM, copying to GPU from there is faster
                                       persistent_workers=(workers > 0), # workers stay alive between epochs instead of being restarted
                                       drop_last=shuffle) # drops the last smaller training batch, test and validation use every image

## Double Conv Block
class DoubleConv(nn.Module):
    # same as CNN ish
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # two 3x3 convs, padding 1 keeps the resolution the same so the skip connections line up without cropping
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # no bias as batchnorm has its own
            nn.BatchNorm2d(out_channels), # stabilises training
            nn.ReLU(inplace=True), # adds non linearity
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

## Down Block (encoder)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2) # halves the resolution, keeps the strongest response in each 2x2 area
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))

## Up Block (decoder)
class Up(nn.Module):
    # skip connection included in this, to allow for the bottleneck to not have as much of an impact on output (if any)
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # learnable upsample, doubles the resolution and halves the filters
        self.conv = DoubleConv(out_channels * 2, out_channels) # *2 because the skip connection is concatenated on

    def forward(self, x, skip):
        x = self.up(x)
        # skip connection, the encoder features at the same resolution still have the fine spatial detail lost by pooling
        x = torch.cat([skip, x], dim=1) # stacks them up on eachother
        return self.conv(x) # joins them together

## UNet
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_filters=32):
        super(UNet, self).__init__()
        f = base_filters
        # encoder, each stage halves resolution and doubles filters: 1x256x256 -> 32x256 -> 64x128 -> 128x64 -> 256x32 -> 512x16
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16) # bottleneck, most context (large receptive field) but least spatial detail

        # decoder, mirrors the encoder back up to full resolution
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)

        # 1x1 conv gives one score (logit) per class per pixel, so the output is categorical (N, 4, 256, 256)
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1) # this gives the segmentation

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # starting weights are scaled to layers fan size so activations and gradients dont explode or vanish through ReLU layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # encoder, keeping each output for the skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder, each up block joins with the encoder output of the same size
        out = self.up1(x5, x4) # the x5, x4 is the skip connection upscale
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        return self.outc(out) # output, class per pixel (the segmentation) (N, 4, 256, 257)

# Loss, cross entropy + dice
def dice_loss(logits, masks, eps=1e-6):
    # note that one wrong pixel affects two classes (the one which it is missing from and the one that its not supposed to be in)
    probs = F.softmax(logits.float(), dim=1) # probability of each class per pixel, float so the sums dont overflow in half precision
    one_hot = F.one_hot(masks, num_classes).permute(0, 3, 1, 2).float() # (N, H, W) labels -> (N, 4, H, W) categorical one-hot target
    dims = (0, 2, 3) # sum over batch and pixels, keeping each class separate
    intersection = (probs * one_hot).sum(dims) # how much it had the right number (one hot * probs = just where it is the class its supposed to be)
    total = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + eps) / (total + eps) # How much the pixel overlaps, soft DSC per class, eps stops divide by zero
    return 1 - dice.mean() # every class counts equally, so the small CSF class isnt drowned out by the large background

def seg_loss(logits, masks): # the actual training loss
    # cross entropy gives strong per pixel gradients early on, dice directly optimises the metric being marked
    return F.cross_entropy(logits.float(), masks) + dice_loss(logits, masks)

# overlap counts per class between hard predictions and labels, summed over every pixel given
def dice_counts(preds, masks):
    classes = torch.arange(num_classes, device=preds.device).view(-1, *([1] * preds.dim())) # compares against every class at once, same as one-hot
    pred_one_hot = preds.unsqueeze(0) == classes
    mask_one_hot = masks.unsqueeze(0) == classes
    dims = tuple(range(1, preds.dim() + 1))
    intersection = (pred_one_hot & mask_one_hot).sum(dims).double()
    total = (pred_one_hot.sum(dims) + mask_one_hot.sum(dims)).double()
    return intersection, total

# 
def dsc_from_counts(intersection, total, eps=1e-6):
    return (2 * intersection + eps) / (total + eps) # DSC = 2|A n B| / (|A| + |B|), eps makes a class missing from both count as perfect instead of NaN

# predicts every slice of one (H, W, D) volume and stacks them back into a (D, H, W) volume
def predict_volume(model, volume, label):
    x = torch.from_numpy(volume).permute(2, 0, 1).unsqueeze(1) # (D, 1, H, W), each slice is one image in the batch
    y = torch.from_numpy(label.astype(np.int64)).permute(2, 0, 1) # (D, H, W)
    preds, losses = [], []
    for s in range(0, x.shape[0], batch_size):
        images = x[s:s + batch_size].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        masks = y[s:s + batch_size].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(images)
        losses.append(seg_loss(logits, masks).item())
        preds.append(logits.argmax(dim=1)) # argmax is the hard prediction, the most likely class per pixel
    return torch.cat(preds), y.to(device), sum(losses) / len(losses)

# DSC per class computed on each whole 3D volume, then averaged over volumes
# this is the same way the 3D model should be scored, so the two numbers are directly comparable
def evaluate(model, images, labels):
    model.eval() # batchnorm uses its running stats so output doesnt depend on what else is in the batch
    vol_dscs, losses = [], []
    with torch.no_grad(): # no graph is built
        for v in range(images.shape[0]):
            preds, masks, loss = predict_volume(model, images[v], labels[v])
            vol_dscs.append(dsc_from_counts(*dice_counts(preds, masks)).float().cpu())
            losses.append(loss)
    vol_dscs = torch.stack(vol_dscs) # (volumes, classes)
    return vol_dscs.mean(dim=0), sum(losses) / len(losses), vol_dscs

# Train model
def train(model, trainloader, val_images, val_labels, epochs, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    # slowly warms up to the max learning rate, then cools off to a small one to settle into a minimum on the last few epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs,
                                                    steps_per_epoch=len(trainloader), pct_start=0.15)
    # multiplies loss by large factor so gradients stay representable in half precision, then divides back down before optimizer step
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    history = {'train_loss': [], 'val_loss': [], 'val_dsc': []}
    best_min_dsc = -1.0 # below any real DSC so the first epoch is always saved
    use_cuda = device.type == 'cuda' # peak VRAM is only tracked on the GPU
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device) # so the peak only counts training
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, masks in trainloader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # removes the gradients

            # where posible use lower precision (this is mixed precision)
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(images)
            loss = seg_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer) # unscales gradients and checks for inf or NaN
            scaler.update() # adjusts scale factor for next iter
            scheduler.step() # moves the learning rate (OneCycle)

            running_loss += loss.detach()

        history['train_loss'].append((running_loss / len(trainloader)).item())

        # validation DSC at every epoch to see how well it performed and make a graph
        val_dsc, val_loss, _ = evaluate(model, val_images, val_labels)
        history['val_loss'].append(val_loss)
        history['val_dsc'].append(val_dsc)

        # keeps the epoch where the worst class is best, since every label needs > 0.9
        if val_dsc.min().item() > best_min_dsc:
            best_min_dsc = val_dsc.min().item()
            torch.save(model.state_dict(), save_path)

        dsc_str = ' '.join(f'{name}: {d:.4f}' for name, d in zip(class_names, val_dsc))
        vram_str = f' | Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB' if use_cuda else ''
        print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {history["train_loss"][-1]:.4f} | Val Loss: {val_loss:.4f} '
              f'| Val DSC {dsc_str}{vram_str} | Elapsed: {time.time() - start:.1f}s')

    elapsed = time.time() - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")
    print("best validation min class DSC:", best_min_dsc, "saved to", save_path)
    if use_cuda:
        print(f'Peak VRAM allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB '
              f'| Peak VRAM reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB')
    return history

# saves the losses and the validation DSC of each class so the graphs can be recreated
def plot_curves(history, save_path):
    epochs = list(range(1, len(history['train_loss']) + 1))
    val_dscs = torch.stack(history['val_dsc'])
    plot_data = {
        'epoch': epochs,
        'train_loss': [round(float(v), 4) for v in history['train_loss']],
        'val_loss': [round(float(v), 4) for v in history['val_loss']],
        'val_dsc': {name: [round(v, 4) for v in val_dscs[:, c].tolist()] for c, name in enumerate(class_names)},
    }
    with open(save_path, 'w') as file:
        json.dump(plot_data, file)

    print('\nPlot data (also saved in ' + os.path.basename(save_path) + ')')
    print('Loss plot: x = epoch, lines = train_loss and val_loss (CE + Dice)')
    print('DSC plot: x = epoch, one line per class in val_dsc, dashed horizontal line at the 0.9 target')
    for key, values in plot_data.items():
        print(f'{key} = {values}')

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, history['train_loss'], label='Train loss')
    # plt.plot(epochs, history['val_loss'], label='Validation loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss (CE + Dice)')
    # plt.title('UNet training vs validation loss')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # for c, name in enumerate(class_names):
    #     plt.plot(epochs, val_dscs[:, c].numpy(), label=name)
    # plt.axhline(0.9, color='k', linestyle='--', label='0.9 target')
    # plt.xlabel('Epoch')
    # plt.ylabel('DSC')
    # plt.title('Validation DSC per class')
    # plt.legend()
    # plt.savefig(save_path)
    # plt.close()

# Visualise, MRI | ground truth | prediction | errors for the middle slice of some test volumes
def visualise(model, images, labels, names, save_path, count=6):
    count = min(count, images.shape[0])
    k = images.shape[3] // 2 # middle slice, where the bladder, rectum and prostate all show up
    x = torch.from_numpy(images[:count, :, :, k].copy()).unsqueeze(1).to(device).to(memory_format=torch.channels_last)
    masks = torch.from_numpy(labels[:count, :, :, k].astype(np.int64)).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)

    # tab10 colours, the same ones test.py uses for each label
    palette = torch.tensor([[0.122, 0.467, 0.706], [1.0, 0.498, 0.055], [0.173, 0.627, 0.173],
                            [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294]], device=device)
    tiles = []
    slice_dscs = []
    for row in range(count):
        slice_dscs.append([round(d, 4) for d in dsc_from_counts(*dice_counts(preds[row], masks[row])).tolist()]) # DSC per class for just this slice, saved so the picture backs up the numbers
        errors = (preds[row] != masks[row]).float()
        image = x[row].float()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) # z-scored image back to [0, 1] so it saves properly
        tiles.append(image.repeat(3, 1, 1)) # grayscale copied into rgb so every tile has 3 channels
        tiles.append(palette[masks[row]].permute(2, 0, 1)) # label -> colour gives (H, W, 3), save_image wants (3, H, W)
        tiles.append(palette[preds[row]].permute(2, 0, 1))
        tiles.append(torch.stack([torch.ones_like(errors), 1 - errors, 1 - errors])) # white background, red where the prediction is wrong
    torchvision.utils.save_image(torch.stack(tiles).cpu(), save_path, nrow=4, pad_value=1)

    vis_data = {
        'class_names': list(class_names),
        'files': [f'{name} slice {k}' for name in names[:count]],
        'slice_dsc': slice_dscs,
    }
    with open(os.path.splitext(save_path)[0] + '.json', 'w') as file:
        json.dump(vis_data, file)

    print(os.path.basename(save_path) + f': {count} rows, columns are MRI | ground truth | prediction | errors (red)')
    print('label colours: ' + ', '.join(f'{name} = {tuple(round(v, 3) for v in c)}' for name, c in zip(class_names, palette.tolist())))
    for name, d in zip(vis_data['files'], slice_dscs):
        print(f'{name} slice DSC = {d}')

    # fig, axes = plt.subplots(len(vis_idx), 4, figsize=(12, 3 * len(vis_idx)), squeeze=False) # squeeze false keeps axes 2D even with one row
    # for row in range(len(vis_idx)):
    #     slice_dsc = dsc_from_counts(*dice_counts(preds[row], masks[row]))
    #
    #     axes[row, 0].imshow(images[row, 0].cpu().numpy(), cmap='gray')
    #     axes[row, 0].set_title(os.path.basename(dataset.image_paths[vis_idx[row]]), fontsize=9)
    #     axes[row, 1].imshow(masks[row].cpu().numpy(), cmap='viridis', vmin=0, vmax=num_classes - 1) # fixed vmin/vmax so each label is the same colour in every plot
    #     axes[row, 1].set_title('Ground truth', fontsize=9)
    #     axes[row, 2].imshow(preds[row].cpu().numpy(), cmap='viridis', vmin=0, vmax=num_classes - 1)
    #     axes[row, 2].set_title('Prediction\nDSC ' + ' '.join(f'{d:.3f}' for d in slice_dsc.tolist()), fontsize=9)
    #     axes[row, 3].imshow((preds[row] != masks[row]).cpu().numpy(), cmap='Reds') # red where the prediction is wrong, mostly just thin lines on the class borders
    #     axes[row, 3].set_title('Errors', fontsize=9)
    #     for ax in axes[row]:
    #         ax.axis('off')
    # plt.suptitle('UNet OASIS segmentation (DSC order: ' + ', '.join(class_names) + ')')
    # plt.tight_layout()
    # plt.savefig(save_path)
    # plt.close()

def main(demo=demo, root=data_root, epochs=num_epochs, out_dir='.', limit=None, workers=num_workers):
    print(device)
    model_path = os.path.join(out_dir, model_name)

    # Load HipMRI, split by patient so the 3D model can use the exact same split (same seed in split_pairs)
    start = time.time()
    pairs = get_pairs(root)
    if not pairs:
        raise FileNotFoundError(f'no HipMRI volumes found in {root}') # fails straight away instead of after building the model
    splits = {name: p[:limit] for name, p in split_pairs(pairs).items()} # limit = volumes per split, only used to make a small set for testing
    train_images, train_labels = load_split(splits['train'])
    val_images, val_labels = load_split(splits['val']) # used to pick the best epoch, so the test set is never seen during training
    test_images, test_labels = load_split(splits['test'])
    test_names = [os.path.basename(mr).replace('_LFOV.nii.gz', '') for mr, _ in splits['test']]

    trainset = HipMRISliceDataset(train_images, train_labels, augment=True)
    trainloader = make_loader(trainset, shuffle=True, workers=workers)

    print("Train volumes:", len(splits['train']), f"({len(trainset)} slices)", "| Validation volumes:", len(splits['val']),
          "| Test volumes:", len(splits['test']))
    print("Dataset setup time: ", (time.time() - start))

    # creates model and passes it to the GPU
    model = UNet(1, num_classes, base_filters).to(device).to(memory_format=torch.channels_last)
    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    if not demo:
        history = train(model, trainloader, val_images, val_labels, epochs, model_path)
        plot_curves(history, os.path.join(out_dir, 'unet2d_plot_data.json'))

    # Inference on the test set from the saved (best) model
    model = UNet(1, num_classes, base_filters).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("testing")
    if (device.type == 'cuda'):
        torch.cuda.synchronize() # waits for all the queued GPU work to finish before timing
    start = time.time()
    test_dsc, test_loss, vol_dscs = evaluate(model, test_images, test_labels)
    if (device.type == 'cuda'):
        torch.cuda.synchronize()
    inference_time = time.time() - start

    for name, d in zip(class_names, test_dsc):
        print(f'Test DSC {name}: {d:.4f}')
    print(f'Min class DSC: {test_dsc.min():.4f} | Mean DSC: {test_dsc.mean():.4f} | Test Loss: {test_loss:.4f} '
          f'on {len(test_names)} volumes in {inference_time:.2f}s ({len(test_names) / inference_time:.2f} volumes/s)')

    # everything needed to compare against the 3D model, which should write the same file with model = '3D UNet'
    results = {
        'model': '2D UNet',
        'class_names': list(class_names),
        'test_cases': test_names, # if these dont match the 3D results, the comparison isnt fair
        'mean_dsc': [round(float(d), 4) for d in test_dsc],
        'per_volume_dsc': {name: [round(d, 4) for d in v.tolist()] for name, v in zip(test_names, vol_dscs)},
        'test_loss': round(test_loss, 4),
        'inference_seconds': round(inference_time, 2),
    }
    with open(os.path.join(out_dir, 'unet2d_results.json'), 'w') as file:
        json.dump(results, file)
    print("saved unet2d_results.json")

    visualise(model, test_images, test_labels, test_names, os.path.join(out_dir, 'unet2d_segmentations.png'))
    print("saved unet2d_segmentations.png and unet2d_segmentations.json")
    return test_dsc

if __name__ == '__main__':
    main()
