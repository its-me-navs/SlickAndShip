"""
Train the U-Net for Sentinel-1 SAR oil-spill segmentation.

Uses:
    - 2-channel VV + VH SAR input
    - configurable crop size (default 256, use --crop-size 128 for ~3-4x
      faster CPU training)
    - BCE + Dice loss
    - IoU validation metric — reported TWO ways: overall IoU (can be
      misleadingly high, since empty-mask crops score 1.0 trivially when
      the model also predicts empty — see compute_iou below) AND
      positive-only IoU (averaged only over crops that actually contain
      real oil pixels — this is the number that actually tells you
      whether the model has learned anything, watch THIS one, not the
      overall figure)
    - best-model checkpointing (now keyed on positive-only IoU, not the
      inflatable overall IoU, so "best" actually means best-at-detecting-oil)

The checkpoint format is compatible with sar_detection.py.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # no-op fallback if tqdm isn't installed
        return iterable

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
    """
    NOTE: this epsilon-smoothed IoU scores an empty prediction against an
    empty target as 1.0 (intersection=0, union=0, (0+eps)/(0+eps)=1). On a
    dataset where most crops have no oil at all (No-Oil/Lookalike categories,
    plus oil-category crops that happen to miss the spill), this can make a
    model that predicts nothing, ever, look like it's scoring ~0.9+. Use
    compute_positive_iou (below) alongside this to see the real picture.
    """
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


def compute_positive_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Same IoU computation, but averaged ONLY over samples in the batch whose
    ground-truth mask actually contains oil pixels. A model that's collapsed
    to always predicting background scores 0.0 here regardless of how high
    compute_iou() reports, since it never overlaps a real positive region.
    Returns (mean_iou_or_None, n_positive_samples_in_batch).
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    target_sums = targets.sum(dim=(1, 2, 3))
    positive_mask = target_sums > 0

    if positive_mask.sum().item() == 0:
        return None, 0

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    positive_iou = iou[positive_mask]
    return positive_iou.mean().item(), int(positive_mask.sum().item())


def evaluate(model, loader, device, desc="val"):

    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    batches = 0

    pos_iou_sum = 0.0
    pos_sample_count = 0

    with torch.no_grad():

        for images, masks in tqdm(loader, desc=desc, leave=False):

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = combined_loss(logits, masks)
            iou = compute_iou(logits, masks)
            pos_iou, n_pos = compute_positive_iou(logits, masks)

            total_loss += loss.item()
            total_iou += iou
            batches += 1

            if pos_iou is not None:
                pos_iou_sum += pos_iou * n_pos
                pos_sample_count += n_pos

    overall_iou = total_iou / max(batches, 1)
    positive_iou = pos_iou_sum / pos_sample_count if pos_sample_count > 0 else 0.0

    return (
        total_loss / max(batches, 1),
        overall_iou,
        positive_iou,
        pos_sample_count,
    )


def train(
    data_root: str,
    output_path: str = "best_unet.pt",
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    crop_size: int = 256,
    seed: int = 42,
    resume_from: str | None = None,
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
        crop_size=crop_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    print(
        f"\nDataset sizes:"
        f"\n  Train: {len(train_dataset)}"
        f"\n  Val:   {len(val_dataset)}"
        f"\n  Test:  {len(test_dataset)}"
    )
    n_batches = max(1, len(train_dataset) // batch_size)
    print(f"  crop_size={crop_size}  num_workers={num_workers}")
    print(f"  ~{n_batches} batches/epoch, {epochs} epochs planned\n")

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

    best_positive_iou = -1.0
    start_epoch_offset = 0

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_positive_iou = ckpt.get("best_positive_iou", -1.0)
        start_epoch_offset = ckpt.get("epoch", 0)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print(f"  Restored optimizer state (momentum/adaptive LR preserved).")
        else:
            print(f"  No optimizer_state in checkpoint (older format) — "
                  f"optimizer starting fresh, Adam will re-adapt within a few batches.")
        print(f"  Resuming from epoch {start_epoch_offset}, "
              f"best_positive_iou so far = {best_positive_iou:.4f}\n")

    epoch_times = []

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch_offset + local_epoch

        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{start_epoch_offset + epochs} [train]", leave=False)
        for images, masks in pbar:

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
            pbar.set_postfix(loss=f"{running_loss / batches:.4f}")

        train_loss = running_loss / max(batches, 1)

        val_loss, val_iou, val_pos_iou, n_pos = evaluate(
            model,
            val_loader,
            device,
            desc=f"Epoch {epoch:02d}/{start_epoch_offset + epochs} [val]",
        )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = avg_epoch_time * (epochs - epoch)
        eta_min = remaining / 60

        print(
            f"Epoch {epoch:02d}/{start_epoch_offset + epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_IoU(overall)={val_iou:.4f} | "
            f"val_IoU(oil-only, n={n_pos})={val_pos_iou:.4f} | "
            f"epoch_time={epoch_time:.0f}s | "
            f"ETA remaining≈{eta_min:.1f}min"
        )

        # -----------------------------------------------------
        # SAVE BEST CHECKPOINT — keyed on the honest oil-only IoU,
        # not the inflatable overall IoU, so "best" actually means
        # best at detecting real oil, not best at ignoring it.
        # -----------------------------------------------------

        if val_pos_iou > best_positive_iou:

            best_positive_iou = val_pos_iou

            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_positive_iou": best_positive_iou,
                "val_iou_overall_at_save": val_iou,
                "epoch": epoch,
            }
            torch.save(
                checkpoint,
                output_path,
            )

            print(
                f"  ✓ Saved best model → {output_path} (oil-only IoU={best_positive_iou:.4f})"
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

    test_loss, test_iou, test_pos_iou, test_n_pos = evaluate(
        model,
        test_loader,
        device,
        desc="test",
    )

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")
    print(f"Test loss              : {test_loss:.4f}")
    print(f"Test IoU (overall)     : {test_iou:.4f}  <- can be misleadingly high, see note above compute_iou()")
    print(f"Test IoU (oil-only, n={test_n_pos}) : {test_pos_iou:.4f}  <- this is the number that matters")
    print(f"Best Val IoU (oil-only): {best_positive_iou:.4f}")
    print("==============================")

    return {
        "test_loss": test_loss,
        "test_iou_overall": test_iou,
        "test_iou_positive": test_pos_iou,
        "best_val_positive_iou": best_positive_iou,
        "checkpoint": output_path,
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the extracted Zenodo dataset root (containing the "
             "01_Train_Val_*_images / *_mask folders and, if downloaded, "
             "03_Test_* folders) — see dataset.py's build_datasets() for "
             "how folders are discovered.",
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

    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Training crop size (must be divisible by 16 — the model has "
             "4 pooling stages). Lower this (e.g. 128) for significantly "
             "faster CPU training at the cost of less spatial context per "
             "sample — roughly a 3-4x speedup going from 256 to 128.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for parallel image loading/"
             "decoding. Try 2-4 on a multi-core CPU to overlap disk I/O "
             "with compute; 0 (default) loads single-threaded.",
    )

    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to an existing checkpoint (e.g. best_unet.pt) to resume "
             "training from. --epochs is the number of ADDITIONAL epochs to "
             "run this session, not a new total — epoch numbers in logs and "
             "the saved checkpoint continue from where the checkpoint left off.",
    )

    args = parser.parse_args()

    train(
        data_root=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
    )