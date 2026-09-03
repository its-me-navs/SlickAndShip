"""
Run this LOCALLY, on your own machine with a GPU and the downloaded
Zenodo dataset — this sandbox has neither. Not part of the FastAPI
dashboard; a one-off script you run to produce detection/checkpoint.pt.

Usage:
    pip install torch torchvision rasterio --break-system-packages
    python3 notebooks/train_sar_segmentation.py \
        --data-root /path/to/extracted/zenodo/dataset \
        --epochs 25 --batch-size 8

Point --data-root at wherever you extracted the three Zenodo parts; see
detection/dataset.py's docstring for the expected folder layout, and
adjust IMAGE_DIRS/MASK_DIRS below if your folder names differ.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split

from detection.unet_model import UNet, combined_loss
from detection.dataset import SARSpillDataset


def build_dataset(data_root: str):
    # adjust these to match your actual extracted folder names if they differ
    image_dirs = [
        os.path.join(data_root, "01_Train_Val_Oil_Spill_images"),
        os.path.join(data_root, "01_Train_Val_No_Oil_Images"),
    ]
    mask_dirs = [
        os.path.join(data_root, "01_Train_Val_Oil_Spill_mask"),
        os.path.join(data_root, "01_Train_Val_No_Oil_mask"),
    ]
    return SARSpillDataset(image_dirs, mask_dirs)


def dice_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = (torch.sigmoid(logits) > thresh).float()
    probs, targets = probs.view(probs.size(0), -1), targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="detection/checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_ds = build_dataset(args.data_root)
    val_len = max(1, int(0.15 * len(full_ds)))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_len, val_len])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_dice += dice_score(logits, masks) * imgs.size(0)
        val_dice /= len(val_ds)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "val_dice": val_dice}, args.out)
            print(f"  -> saved new best checkpoint (val_dice={val_dice:.4f}) to {args.out}")

    print(f"Done. Best val_dice={best_val_dice:.4f}, checkpoint at {args.out}")


if __name__ == "__main__":
    main()
