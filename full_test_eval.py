"""
Proper held-out evaluation: sliding-window inference over FULL images,
instead of a single random crop per image. test_only.py's random-crop
evaluation is not a reliable/reproducible measurement of full-scene
performance — this script is.

Usage: python full_test_eval.py --data data/sar_raw --checkpoint best_unet.pt
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from tqdm import tqdm

from detection.dataset import build_datasets, _read_tiff
from detection.unet_model import UNet


def sliding_window_predict(model, img, device, tile=128, stride=96):
    """img: (H, W, 2) already normalized. Returns (H, W) prediction
    probability map, built by tiling the whole image and averaging
    overlapping tile predictions."""
    h, w = img.shape[:2]
    pred_full = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    tops = list(range(0, max(h - tile, 0) + 1, stride))
    lefts = list(range(0, max(w - tile, 0) + 1, stride))
    if tops[-1] != h - tile:
        tops.append(h - tile)
    if lefts[-1] != w - tile:
        lefts.append(w - tile)

    for top in tops:
        for left in lefts:
            patch = img[top:top + tile, left:left + tile, :]
            patch_t = torch.from_numpy(np.moveaxis(patch, -1, 0)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = torch.sigmoid(model(patch_t)).cpu().numpy()[0, 0]
            pred_full[top:top + tile, left:left + tile] += out
            count[top:top + tile, left:left + tile] += 1

    return pred_full / np.maximum(count, 1)


def compute_iou_full(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Standard IoU between two binary masks. Returns None if both masks
    are empty (undefined, not 0 or 1) so callers can exclude these from
    the oil-only average rather than let them distort it."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None
    return intersection / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="best_unet.pt")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cpu")

    # build_datasets() does crop-based Dataset wrapping we don't want here —
    # we only use it to get the correctly-discovered (image_path, mask_path)
    # pairs for the test split, via the underlying dataset's .pairs list.
    _, _, test_dataset = build_datasets(data_root=args.data, seed=42, crop_size=args.tile)
    pairs = test_dataset.pairs
    print(f"\nEvaluating {len(pairs)} full test images (tile={args.tile}, stride={args.stride}, threshold={args.threshold})\n")

    model = UNet(in_channels=2, out_channels=1, base=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    overall_ious = []
    oil_only_ious = []

    for img_path, mask_path in tqdm(pairs, desc="full-scene eval"):
        img = _read_tiff(img_path).astype(np.float32)
        mask = _read_tiff(mask_path).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]

        img = np.clip(img, -35, 5)
        img = (img + 35) / 40.0

        pred_prob = sliding_window_predict(model, img, device, tile=args.tile, stride=args.stride)
        pred_mask = pred_prob >= args.threshold

        iou = compute_iou_full(pred_mask, mask)
        has_oil = mask.sum() > 0

        if iou is not None:
            overall_ious.append(iou)
            if has_oil:
                oil_only_ious.append(iou)

    print("\n==============================")
    print("FULL-SCENE TEST RESULTS")
    print("==============================")
    print(f"Images evaluated        : {len(pairs)}")
    print(f"Overall IoU (mean)      : {np.mean(overall_ious):.4f}")
    print(f"Oil-only IoU (mean, n={len(oil_only_ious)}) : {np.mean(oil_only_ious):.4f}  <- this is the number that matters")
    print("==============================")


if __name__ == "__main__":
    main()