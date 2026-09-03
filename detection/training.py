"""
Train the U-Net for Sentinel-1 SAR oil-spill segmentation.

Uses:
    - 2-channel VV + VH SAR input
    - 256x256 crops
    - BCE + Dice loss
    - IoU validation metric
    - best-model checkpointing

The checkpoint format is compatible with sar_detection.py.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import build_datasets
from .unet_model import UNet, combined_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (
        preds.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
        - intersection
    )

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean().item()


def evaluate(model, loader, device):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = combined_loss(logits, masks)
            iou = compute_iou(logits, masks)

            total_loss += loss.item()
            total_iou += iou
            batches += 1

    return (
        total_loss / max(batches, 1),
        total_iou / max(batches, 1),
    )


def train(
    data_root: str,
    output_path: str = "best_unet.pt",
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    seed: int = 42,
):

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    train_dataset, val_dataset, test_dataset = build_datasets(
        data_root=data_root,
        seed=seed,
        crop_size=256,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"\nDataset sizes:"
        f"\n  Train: {len(train_dataset)}"
        f"\n  Val:   {len(val_dataset)}"
        f"\n  Test:  {len(test_dataset)}"
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = UNet(
        in_channels=2,
        out_channels=1,
        base=32,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_iou = -1.0

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        batches = 0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = combined_loss(
                logits,
                masks,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batches += 1

        train_loss = running_loss / max(batches, 1)

        val_loss, val_iou = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_IoU={val_iou:.4f}"
        )

        # -----------------------------------------------------
        # SAVE BEST CHECKPOINT
        # -----------------------------------------------------

        if val_iou > best_iou:

            best_iou = val_iou

            checkpoint = {
                "model_state": model.state_dict(),
                "best_iou": best_iou,
                "epoch": epoch,
            }

            torch.save(
                checkpoint,
                output_path,
            )

            print(
                f"  ✓ Saved best model → {output_path}"
            )

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        output_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    test_loss, test_iou = evaluate(
        model,
        test_loader,
        device,
    )

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")
    print(f"Test loss : {test_loss:.4f}")
    print(f"Test IoU  : {test_iou:.4f}")
    print(f"Best Val IoU: {best_iou:.4f}")
    print("==============================")

    return {
        "test_loss": test_loss,
        "test_iou": test_iou,
        "best_val_iou": best_iou,
        "checkpoint": output_path,
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset root containing Images/ and Mask/",
    )

    parser.add_argument(
        "--output",
        default="best_unet.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    train(
        data_root=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )