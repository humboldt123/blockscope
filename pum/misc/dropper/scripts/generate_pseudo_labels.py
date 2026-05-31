#!/usr/bin/env python3
"""Generate per-slot pseudo-labels for VPT frames where inventory GUI is visible.

Scans all VPT jsonls for isGuiInventory=true, runs the EXP1 SigLIP2 model on those
frames, and saves argmax predictions per patch as pseudo-labels.

Usage:
    cd /home/vvm33/dropper && \
    CUDA_VISIBLE_DEVICES=0 /data/conda_envs/dropper/bin/python \
        scripts/generate_pseudo_labels.py
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
from torch.utils.data import Dataset, DataLoader

from models.encoder import Siglip2Encoder
from models.heads import InventoryHead


VPT_CONTRACTOR_DIR = Path("/data/vvm33/vpt_contractor")
VPT_FRAMES_DIR = Path("/data/vvm33/vpt_frames")
OUTPUT_DIR = Path("/data/vvm33/vpt_frames_pseudo_inv")
CKPT_PATH = Path("/data/vvm33/dropper_checkpoints/exp1_siglip2/best.pt")

MAX_NUM_PATCHES = 768
BATCH_SIZE = 64
NUM_WORKERS = 8


class VPTInventoryFrameDataset(Dataset):
    """Yields (session, frame_idx, pil_image) for frames where isGuiInventory=true."""

    def __init__(self):
        self.samples = []
        jsonl_files = sorted(VPT_CONTRACTOR_DIR.glob("*.jsonl"))
        print(f"Scanning {len(jsonl_files)} jsonl files for inventory-open frames...")
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
                    if tick.get("isGuiInventory"):
                        frame_idx = tick.get("tick", 0)
                        img_path = frames_dir / f"{frame_idx:05d}.jpg"
                        if img_path.exists():
                            self.samples.append((sess, frame_idx, str(img_path)))
        print(f"Found {len(self.samples)} inventory-open frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sess, frame_idx, img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return sess, frame_idx, img


def collate_fn(batch):
    sessions = [b[0] for b in batch]
    frame_idxs = [b[1] for b in batch]
    images = [b[2] for b in batch]
    return sessions, frame_idxs, images


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    encoder = Siglip2Encoder(max_num_patches=MAX_NUM_PATCHES, freeze=True).to(device)
    head = InventoryHead(encoder.hidden_size).to(device)

    print(f"Loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    encoder.load_state_dict(
        {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")},
        strict=False,
    )
    head.load_state_dict(
        {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")},
        strict=False,
    )
    encoder.eval()
    head.eval()

    # Build dataset
    dataset = VPTInventoryFrameDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    for sessions, frame_idxs, images in tqdm(loader, desc="Generating pseudo-labels"):
        patches, mask = encoder(images)  # (B, max_patches, 768), (B, max_patches)
        logits = head(patches)           # (B, max_patches, n_classes)

        # Only keep valid patches (mask out padded ones)
        preds = logits.argmax(dim=-1).cpu().numpy()  # (B, max_patches)
        mask_np = mask.cpu().numpy()                  # (B, max_patches)

        for i in range(len(sessions)):
            sess = sessions[i]
            frame_idx = frame_idxs[i]
            valid_preds = preds[i][mask_np[i] == 1]  # only valid patches

            out_dir = OUTPUT_DIR / sess
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{frame_idx:05d}.pt"
            torch.save(torch.from_numpy(valid_preds), out_path)
            n_saved += 1

    print(f"Done. Saved {n_saved} pseudo-labels to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
