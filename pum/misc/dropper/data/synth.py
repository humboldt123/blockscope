"""Synthetic inventory dataset from inv_synth/.

Each sample: a 640x360 JPEG with a randomly-positioned Minecraft GUI panel.
The full frame is resized to 224x224 for SigLIP (same as the policy sees).
Per-patch labels are computed dynamically by detecting the GUI panel bbox
and mapping slot positions to patch indices in 224x224 space.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

from vocab import name_to_id, AIR_ID
from models.geometry import (
    build_patch_target_tensor, slot_to_patch_indices,
    find_gui_bbox, centered_gui_bbox, get_slot_type,
)

DATA_DIR = Path("/data/vvm33/inv_synth")
IMAGES_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "meta.jsonl"


def _parse_slot_labels(entry: dict) -> dict[int, int]:
    labels = {}
    player = entry.get("player", {})
    if isinstance(player, dict):
        for slot_str, info in player.items():
            labels[int(slot_str)] = name_to_id(info["name"])
    container = entry.get("container")
    if isinstance(container, list):
        for i, info in enumerate(container):
            if info is not None:
                labels[1000 + i] = name_to_id(info["name"])
    return labels


_resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class SynthInventoryDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        val_frac: float = 0.05,
        transform=None,
        return_raw: bool = False,
    ):
        self.entries = []
        with open(META_PATH, encoding="utf-8", errors="replace") as f:
            for line in f:
                self.entries.append(json.loads(line))

        n = len(self.entries)
        n_val = int(n * val_frac)
        if split == "val":
            self.entries = self.entries[:n_val]
        elif split == "train":
            self.entries = self.entries[n_val:]

        self.transform = transform or _resize_transform
        self.return_raw = return_raw

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = IMAGES_DIR / entry["filename"]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Full frame → 224x224 for standard SigLIP, or raw PIL for naFlex
        if self.return_raw:
            pixel_values = img
        else:
            pixel_values = self.transform(img)

        gui_type = entry["gui"]
        slot_labels = _parse_slot_labels(entry)

        # Detect GUI bbox in this specific image for dynamic patch targets
        bbox = find_gui_bbox(img_np)
        if bbox is None:
            bbox = centered_gui_bbox(gui_type)

        target = build_patch_target_tensor(gui_type, slot_labels, bbox)

        slot_type_map = {}
        mapping = slot_to_patch_indices(gui_type, bbox)
        for slot_idx, patch_idx in mapping.items():
            slot_type_map[patch_idx] = get_slot_type(slot_idx, gui_type)

        return {
            "pixel_values": pixel_values,
            "target": target,
            "gui_type": gui_type,
            "slot_type_map": slot_type_map,
            "filename": entry["filename"],
            "bbox": bbox,
        }


class SynthInventoryForEncoder(Dataset):
    def __init__(self, split: str = "train", val_frac: float = 0.05, transform=None):
        self._inner = SynthInventoryDataset(split=split, val_frac=val_frac, transform=transform)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        sample = self._inner[idx]
        return sample["pixel_values"], sample["target"], sample["gui_type"]
