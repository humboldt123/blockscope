#!/usr/bin/env python3
"""Evaluate EXP1 SigLIP2 on VPT inventory frames at native 640x360 resolution.

Loads frames directly from MP4s (no pre-extraction needed), preserving
the original aspect ratio so patch→slot mappings from training transfer.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", "/data/hf_cache_root")

import json
import torch
import numpy as np
import cv2
import wandb
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

from models.encoder import Siglip2Encoder
from models.heads import InventoryHead
from vocab import id_to_name

VPT_CONTRACTOR_DIR = Path("/data/vvm33/vpt_contractor")
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
    return set(item["type"] for item in inv_list)


def predictions_to_bag(pred_patches, ignore_air=True):
    counts = Counter()
    for p in pred_patches.tolist():
        name = id_to_name(p)
        if ignore_air and name == "air":
            continue
        counts[name] += 1
    return counts


def load_frame_from_mp4(mp4_path, frame_idx):
    """Open MP4, seek to frame_idx, return PIL RGB image."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    # frame is BGR; convert to RGB PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


@torch.inference_mode()
def main(max_frames=5000, num_wandb_examples=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    encoder, head = load_model(device)

    # Init wandb
    wandb.init(project="dropper", name="exp1_vpt_inventory_eval_native", job_type="eval")

    # Collect inventory-open frames with ground truth
    samples = []
    jsonl_files = sorted(VPT_CONTRACTOR_DIR.glob("*.jsonl"))
    print(f"Scanning {len(jsonl_files)} sessions...")
    for jsonl_path in tqdm(jsonl_files, desc="Scanning jsonls"):
        sess = jsonl_path.stem
        mp4_path = VPT_CONTRACTOR_DIR / f"{sess}.mp4"
        if not mp4_path.exists():
            continue
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tick.get("isGuiInventory") and tick.get("inventory"):
                    if len(tick["inventory"]) == 0:
                        continue
                    frame_idx = tick.get("tick", 0)
                    gt_set = vpt_inv_to_set(tick["inventory"])
                    if len(gt_set) > 0:
                        samples.append((str(mp4_path), frame_idx, gt_set))
        if len(samples) >= max_frames:
            break

    print(f"Evaluating on {len(samples)} inventory frames...")

    exact_match = 0
    partial_match = 0
    total_pred_items = 0
    total_gt_items = 0
    intersection_sum = 0
    union_sum = 0

    # Examples to log to wandb
    example_images = []
    example_metadata = []

    # Cache open video captures for sessions with multiple frames
    cap_cache = {}

    for idx, (mp4_path_str, frame_idx, gt_set) in enumerate(tqdm(samples, desc="Inference")):
        mp4_path = Path(mp4_path_str)
        sess = mp4_path.stem

        # Use cached capture if available
        if sess in cap_cache:
            cap = cap_cache[sess]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Re-open if failed
                cap = cv2.VideoCapture(str(mp4_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap_cache[sess] = cap
                if not ret:
                    continue
        else:
            cap = cv2.VideoCapture(str(mp4_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            cap_cache[sess] = cap

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        patches, mask = encoder([img])
        logits = head(patches)

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

        # Collect examples for wandb
        if len(example_images) < num_wandb_examples:
            example_images.append(img)
            example_metadata.append({
                "session": sess,
                "frame": frame_idx,
                "gt": sorted(gt_set),
                "pred": sorted(pred_set),
                "overlap": sorted(pred_set & gt_set),
                "gt_only": sorted(gt_set - pred_set),
                "pred_only": sorted(pred_set - gt_set),
            })

    # Close cached captures
    for cap in cap_cache.values():
        cap.release()

    n = len(samples)
    metrics = {
        "exact_bag_match": exact_match / n,
        "partial_overlap": partial_match / n,
        "avg_pred_items": total_pred_items / n,
        "avg_gt_items": total_gt_items / n,
        "mean_iou": intersection_sum / union_sum,
        "recall": intersection_sum / total_gt_items,
        "precision": intersection_sum / total_pred_items,
        "num_frames": n,
    }

    print(f"\nResults on {n} VPT inventory frames (native 640x360):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # Log metrics
    wandb.log(metrics)

    # Log example frames
    wandb_images = []
    for img, meta in zip(example_images, example_metadata):
        caption = (
            f"Session: {meta['session']} | Frame: {meta['frame']}\n"
            f"GT ({len(meta['gt'])}): {', '.join(meta['gt'][:10])}{'...' if len(meta['gt']) > 10 else ''}\n"
            f"Pred ({len(meta['pred'])}): {', '.join(meta['pred'][:10])}{'...' if len(meta['pred']) > 10 else ''}\n"
            f"Overlap: {', '.join(meta['overlap'][:10])}{'...' if len(meta['overlap']) > 10 else ''}\n"
            f"GT only: {', '.join(meta['gt_only'][:10])}{'...' if len(meta['gt_only']) > 10 else ''}\n"
            f"Pred only: {', '.join(meta['pred_only'][:10])}{'...' if len(meta['pred_only']) > 10 else ''}"
        )
        wandb_images.append(wandb.Image(img, caption=caption))

    wandb.log({"example_frames": wandb_images})
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=5000)
    parser.add_argument("--num_wandb_examples", type=int, default=30)
    args = parser.parse_args()
    main(args.max_frames, args.num_wandb_examples)
