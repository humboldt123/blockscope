"""Survival-only synthetic inventory dataset.

Filters inv_synth/meta.jsonl to gui == "survival" only.
Returns per-slot targets instead of per-patch targets.
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
    build_slot_target_tensor, find_gui_bbox, centered_gui_bbox, get_slot_type,
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
    return labels


_resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class SurvivalInventoryDataset(Dataset):
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
                meta = json.loads(line)
                if meta.get("gui") == "survival":
                    self.entries.append(meta)

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

        if self.return_raw:
            pixel_values = img
        else:
            pixel_values = self.transform(img)

        gui_type = "survival"
        slot_labels = _parse_slot_labels(entry)

        # Detect GUI bbox in this specific image for dynamic patch targets
        bbox = find_gui_bbox(img_np)
        if bbox is None:
            bbox = centered_gui_bbox(gui_type)

        target, slot_indices = build_slot_target_tensor(gui_type, slot_labels, bbox)

        slot_types = [get_slot_type(idx, gui_type) for idx in slot_indices]

        return {
            "pixel_values": pixel_values,
            "target": target,
            "slot_indices": slot_indices,
            "slot_types": slot_types,
            "filename": entry["filename"],
            "bbox": bbox,
        }


def survival_collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    bboxes = [b["bbox"] for b in batch]
    slot_indices = [b["slot_indices"] for b in batch]
    slot_types = [b["slot_types"] for b in batch]
    filenames = [b["filename"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "target": targets,
        "bboxes": bboxes,
        "slot_indices": slot_indices,
        "slot_types": slot_types,
        "filename": filenames,
    }
