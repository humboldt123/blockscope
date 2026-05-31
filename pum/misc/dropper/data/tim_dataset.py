"""Temporal Inventory Memory (TIM) dataset.

Builds T=16 frame sequences from VPT data where at least one frame has
isGuiInventory=true. The target is the per-slot pseudo-label from the most
recent inventory-open frame in the sequence.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

VPT_CONTRACTOR_DIR = Path("/data/vvm33/vpt_contractor")
VPT_FRAMES_DIR = Path("/data/vvm33/vpt_frames")
PSEUDO_LABEL_DIR = Path("/data/vvm33/vpt_frames_pseudo_inv")


class TIMDataset(Dataset):
    """VPT sequences with pseudo-labeled inventory frames.

    Each sample is (frames, target) where:
      - frames: list of T PIL RGB images (all 224x224)
      - target: (num_patches,) tensor of pseudo-label class indices

    The target comes from the most recent inventory-open frame in the sequence.
    """

    def __init__(
        self,
        seq_len: int = 16,
        stride: int = 8,
        split: str = "train",
        val_sessions: int = 100,
    ):
        self.seq_len = seq_len
        self.stride = stride

        # List all sessions
        jsonl_files = sorted(VPT_CONTRACTOR_DIR.glob("*.jsonl"))
        all_sessions = [p.stem for p in jsonl_files]

        if split == "val":
            self.sessions = all_sessions[:val_sessions]
        elif split == "train":
            self.sessions = all_sessions[val_sessions:]
        else:
            self.sessions = all_sessions

        # Pre-compute windows and inventory flags
        self.windows = []  # list of (session, start_tick, target_tick)
        print(f"Building TIM {split} dataset...")
        for sess in self.sessions:
            jsonl_path = VPT_CONTRACTOR_DIR / f"{sess}.jsonl"
            frames_dir = VPT_FRAMES_DIR / sess
            pseudo_dir = PSEUDO_LABEL_DIR / sess
            if not frames_dir.exists() or not pseudo_dir.exists():
                continue

            # Load isGuiInventory flags and tick-to-frame mapping
            inv_ticks = set()
            with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        tick = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if tick.get("isGuiInventory"):
                        frame_idx = tick.get("tick", 0)
                        if (pseudo_dir / f"{frame_idx:05d}.pt").exists():
                            inv_ticks.add(frame_idx)

            if not inv_ticks:
                continue

            # Count available frames
            n_frames = len(list(frames_dir.glob("*.jpg")))
            for start in range(0, n_frames - seq_len, stride):
                # Find most recent inventory-open frame in this window
                target_tick = None
                for t in range(start + seq_len - 1, start - 1, -1):
                    if t in inv_ticks:
                        target_tick = t
                        break
                if target_tick is not None:
                    self.windows.append((sess, start, target_tick))

        print(f"  {split}: {len(self.windows)} sequences from {len(self.sessions)} sessions")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sess, start, target_tick = self.windows[idx]
        frames_dir = VPT_FRAMES_DIR / sess

        # Load T frames
        frames = []
        for t in range(start, start + self.seq_len):
            img_path = frames_dir / f"{t:05d}.jpg"
            frames.append(Image.open(img_path).convert("RGB"))

        # Load pseudo-label for target frame
        pseudo_path = PSEUDO_LABEL_DIR / sess / f"{target_tick:05d}.pt"
        target = torch.load(pseudo_path, map_location="cpu", weights_only=True)

        return frames, target


def tim_collate(batch):
    """Collate for TIM dataset.

    Returns:
        all_frames: flat list of B*T PIL Images
        targets: (B, num_patches) stacked tensor
    """
    all_frames = []
    targets = []
    for frames, target in batch:
        all_frames.extend(frames)
        targets.append(target)
    targets = torch.stack(targets)
    return all_frames, targets
