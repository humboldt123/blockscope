#!/usr/bin/env python3
"""Evaluate EXP1 SigLIP2 on VPT inventory frames.

Runs the EXP1 model on VPT frames where isGuiInventory=true,
converts per-patch predictions to a bag-of-items, and compares
to VPT's ground-truth bag-of-items.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", "/data/hf_cache_root")

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

from models.encoder import Siglip2Encoder
from models.heads import InventoryHead
from vocab import id_to_name

VPT_CONTRACTOR_DIR = Path("/data/vvm33/vpt_contractor")
VPT_FRAMES_DIR = Path("/data/vvm33/vpt_frames")
CKPT_PATH = Path("/data/vvm33/dropper_checkpoints/exp1_siglip2/best.pt")
MAX_NUM_PATCHES = 768


def load_model(device):
    encoder = Siglip2Encoder(max_num_patches=MAX_NUM_PATCHES, freeze=True).to(device)
    head = InventoryHead(encoder.hidden_size).to(device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
    encoder.load_state_dict(enc_state, strict=False)
    head.load_state_dict(head_state, strict=False)
    encoder.eval()
    head.eval()
    return encoder, head


def vpt_inv_to_set(inv_list):
    """VPT inventory list -> set of item names (without minecraft: prefix)."""
    return set(item["type"] for item in inv_list)


def predictions_to_bag(pred_patches, ignore_air=True):
    """Per-patch predictions -> Counter of item names."""
    counts = Counter()
    for p in pred_patches.tolist():
        name = id_to_name(p)
        if ignore_air and name == "air":
            continue
        counts[name] += 1
    return counts


@torch.inference_mode()
def main(max_frames=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    encoder, head = load_model(device)

    # Collect inventory-open frames with ground truth
    samples = []
    jsonl_files = sorted(VPT_CONTRACTOR_DIR.glob("*.jsonl"))
    print(f"Scanning {len(jsonl_files)} sessions...")
    for jsonl_path in tqdm(jsonl_files, desc="Scanning"):
        sess = jsonl_path.stem
        frames_dir = VPT_FRAMES_DIR / sess
        if not frames_dir.exists():
            continue
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tick.get("isGuiInventory") and tick.get("inventory"):
                    frame_idx = tick.get("tick", 0)
                    img_path = frames_dir / f"{frame_idx:05d}.jpg"
                    if img_path.exists():
                        gt_set = vpt_inv_to_set(tick["inventory"])
                        if len(gt_set) > 0:
                            samples.append((str(img_path), gt_set))
        if len(samples) >= max_frames:
            break

    print(f"Evaluating on {len(samples)} inventory frames...")

    exact_match = 0
    partial_match = 0  # at least 1 item in common
    total_pred_items = 0
    total_gt_items = 0
    intersection_sum = 0
    union_sum = 0

    # Sample-level metrics
    for img_path, gt_set in tqdm(samples, desc="Inference"):
        img = Image.open(img_path).convert("RGB")
        patches, mask = encoder([img])
        logits = head(patches)  # (1, max_patches, n_classes)

        # Only valid patches
        valid_mask = mask[0].cpu().numpy()
        pred = logits[0].argmax(dim=-1).cpu().numpy()
        pred = pred[valid_mask == 1]

        pred_bag = predictions_to_bag(torch.from_numpy(pred))
        pred_set = set(pred_bag.keys())

        if pred_set == gt_set:
            exact_match += 1
        if pred_set & gt_set:
            partial_match += 1

        total_pred_items += len(pred_set)
        total_gt_items += len(gt_set)
        intersection_sum += len(pred_set & gt_set)
        union_sum += len(pred_set | gt_set)

    n = len(samples)
    print(f"\nResults on {n} VPT inventory frames:")
    print(f"  Exact bag match:     {exact_match / n:.3f}")
    print(f"  Partial overlap:     {partial_match / n:.3f}")
    print(f"  Avg pred items:      {total_pred_items / n:.2f}")
    print(f"  Avg GT items:        {total_gt_items / n:.2f}")
    print(f"  Mean IoU (set):      {intersection_sum / union_sum:.3f}")
    print(f"  Recall (item):       {intersection_sum / total_gt_items:.3f}")
    print(f"  Precision (item):    {intersection_sum / total_pred_items:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=5000)
    args = parser.parse_args()
    main(args.max_frames)
