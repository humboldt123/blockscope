"""
patch_label_dataset.py — Dataset for the SigLIP NaFlex visible-block head.

Yields (frame_rgb_np, patch_class_labels) per tick:
  frame_rgb_np       : (360, 640, 3) uint8    raw RGB — pass to AutoProcessor
  patch_class_labels : (22, 40)      int64    — per-patch class index

Class encoding (n_classes=91):
  0 .. 90    SCATTER_BLOCKS  — only trained on patches with a visible scatter block
  -1         sky / no block, or OOV block → ignored in loss (ignore_index=-1)

Sky patches are ignored so the model trains purely on block classification signal.
Requires patch_labels_s20.npz in each session's labels/ dir.
Run precompute_patch_labels.py first.
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

SIGLIP_W = 40   # patches per row
SIGLIP_H = 22   # patches per col

PATCH_LABELS_STEM = "patch_labels_s20.npz"

RAW_ROOT     = Path("/data/vvm33/BLOCKSCOPE_DATA")
SMELTED_ROOT = Path("/data/vvm33/SMELTED_DATA")


def build_viblock_lut(blocktype_vocab_path: Path) -> tuple[np.ndarray, int]:
    """
    Build class LUT for viblock (91 classes, same as blocktype).
    Returns (lut, n_classes):
      lut[scatter_sid] = scatter_blocktype_class  (0..90)
      lut[0]           = -1   sky → ignore
      lut[oov_sid]     = -1   OOV → ignore
    Sky patches produce no gradient; the model trains only on scatter block patches.
    """
    import json
    with open(blocktype_vocab_path) as f:
        v = json.load(f)
    lut = np.full(65536, -1, dtype=np.int64)   # default: ignore
    for sid_str, scatter_cls in v["sid_to_class"].items():
        sid = int(sid_str)
        if 0 < sid < 65536:
            lut[sid] = int(scatter_cls)
    n_classes = int(v["n_classes"])   # 91
    return lut, n_classes


class PatchLabelDataset(IterableDataset):
    """
    Streams (frame_rgb_np, patch_class_labels) tuples from session videos.

    frame_rgb_np       : (360, 640, 3) uint8    raw RGB — pass to AutoProcessor
    patch_class_labels : (22, 40)      int64    class indices (see module docstring)
    """

    def __init__(self, session_names: list[str], lut: np.ndarray,
                 shuffle: bool = True):
        self.session_names = session_names
        self.lut           = lut
        self.shuffle       = shuffle

    def __iter__(self):
        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per = int(np.ceil(len(sessions) / worker_info.num_workers))
            start = worker_info.id * per
            sessions = sessions[start: start + per]

        for session in sessions:
            labels_dir = SMELTED_ROOT / session / "labels"
            pl_path    = labels_dir / PATCH_LABELS_STEM

            if not pl_path.exists():
                log.warning("No patch labels for %s — run precompute_patch_labels.py", session)
                continue

            data         = np.load(pl_path)
            tick_indices = data["tick_indices"].tolist()   # (K,) int
            patch_labels = data["patch_labels"]            # (K, 22, 40) uint16

            raw        = RAW_ROOT / session
            video_path = str(raw / "video.mp4")
            cap        = cv2.VideoCapture(video_path)
            n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            order = list(range(len(tick_indices)))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                tick_idx = int(tick_indices[k])
                if tick_idx >= n_frames:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, tick_idx)
                ok, bgr = cap.read()
                if not ok:
                    continue

                frame_rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (360, 640, 3) uint8

                raw_sids   = patch_labels[k].astype(np.int32)     # (22, 40)
                cls_labels = self.lut[np.clip(raw_sids, 0, len(self.lut) - 1)]  # (22, 40)

                yield (frame_rgb,
                       torch.from_numpy(cls_labels).long())

            cap.release()


def patch_label_collate(batch):
    """Collate raw-numpy frames + label tensors — frames stay as a list for the processor."""
    frames_np = [b[0] for b in batch]              # list of (360, 640, 3) uint8
    labels    = torch.stack([b[1] for b in batch]) # (B, 22, 40)
    return frames_np, labels
