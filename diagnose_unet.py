"""
Diagnostic: checks whether best_unet.pt actually predicts oil pixels on
known oil-spill crops, or has collapsed to always predicting background
(a common failure mode that produces misleadingly high IoU on datasets
with mostly-empty masks — see compute_iou's epsilon-smoothing behavior
on empty/empty predictions).

Usage:  python diagnose_unet.py --data data/sar_raw --checkpoint best_unet.pt
"""

import argparse
import numpy as np
import torch

from detection.unet_model import UNet
from detection.dataset import SARSpillDataset, _discover_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="best_unet.pt")
    parser.add_argument("--n-samples", type=int, default=15)
    args = parser.parse_args()

    trainval_pairs, _ = _discover_pairs(args.data)

    oil_pairs = [(ip, mp) for ip, mp in trainval_pairs
                 if "oil" in ip.lower() and "no_oil" not in ip.lower() and "no-oil" not in ip.lower()]
    print(f"Found {len(oil_pairs)} oil-category image/mask pairs to sample from.\n")

    ds = SARSpillDataset(pairs=oil_pairs, crop_size=128, augment=False)

    model = UNet(in_channels=2, out_channels=1, base=32)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    best_metric = ckpt.get("best_positive_iou", ckpt.get("best_iou"))
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "unknown"
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, best_iou_at_save={best_metric_str}\n")

    checked, had_positive_mask, model_predicted_anything, both = 0, 0, 0, 0

    with torch.no_grad():
        for i in range(min(args.n_samples, len(ds))):
            img, mask = ds[i]
            gt_positive = mask.sum().item() > 0

            logits = model(img.unsqueeze(0))
            probs = torch.sigmoid(logits)[0, 0]
            pred_positive_count = (probs > 0.5).sum().item()
            max_prob = probs.max().item()

            checked += 1
            if gt_positive:
                had_positive_mask += 1
            if pred_positive_count > 0:
                model_predicted_anything += 1
            if gt_positive and pred_positive_count > 0:
                both += 1

            print(f"[{i}] gt_oil_pixels={int(mask.sum().item()):5d}  "
                  f"pred_oil_pixels={pred_positive_count:5d}  "
                  f"max_predicted_prob={max_prob:.4f}")

    print(f"\n--- Summary over {checked} oil-category crops ---")
    print(f"Crops with ANY real oil pixels in ground truth: {had_positive_mask}/{checked}")
    print(f"Crops where model predicted ANY oil pixels:     {model_predicted_anything}/{checked}")
    print(f"Crops where BOTH gt and prediction had oil:     {both}/{checked}")

    if model_predicted_anything == 0:
        print("\n⚠️  MODEL HAS COLLAPSED: it never predicts a single oil pixel.")
    elif both == 0 and had_positive_mask > 0:
        print("\n⚠️  Model predicts SOMETHING, but never overlapping with real oil regions.")
    else:
        print("\n✓ Model is predicting oil pixels that at least sometimes overlap real "
              "oil regions — it has learned something.")


if __name__ == "__main__":
    main()