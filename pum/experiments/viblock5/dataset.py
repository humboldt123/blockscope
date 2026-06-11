"""
ViBlock5Dataset — image-only dense voxel prediction dataset with VISIBILITY MASK.

This is a fixed re-baseline of the viblock3/4 dataset.  For each tick it yields:
  frame_rgb  : (H, W, 3)    uint8   — raw RGB frame (no normalisation here)
  cam_blocks : (32, 32, 32) int64   — camera-relative block class indices
               0 = air, 1..n_classes = block type, -1 = OOV (ignore in loss)
  cam_vis    : (32, 32, 32) uint8   — Furnace visibility mask, 1 = visible
               from the camera this tick, 0 = occluded / behind camera / off-grid.

WHAT CHANGED vs the viblock3 dataset that viblock4 reused (and WHY):

  E10 (headline) — cam_vis is now yielded.  viblock4 never loaded Furnace's
      visibility mask, so it trained CE over the full 32,768-voxel grid — ~half
      of which sits *behind the camera* (player at window index 15).  The loss
      therefore rewarded a learned terrain prior, not perception.  viblock5
      carries the mask through so train_finetune.py can weight occluded / behind-
      camera voxels to ~0.  Ticks without a precomputed mask are SKIPPED (never
      fabricate an all-visible mask — that would silently reintroduce the bug).
      Run precompute_cam_vis.py before training.

  T1 — the rotate-to-camera "fake air" front plane is neutralised for free by the
      mask: those cells have no visibility source, so cam_vis=0 there and they
      drop out of the loss.  cam_classes is left byte-for-byte identical to what
      viblock4 trained on, so the architecture comparison stays clean.  (The only
      residual T1 effect — a real world slab rotated out of the asymmetric window
      at 90°/180° — is an inherent capture-window limitation, fixable only in the
      mod, and out of scope for a training-side re-baseline.)

  T2 — worker sharding.  The old code shuffled the session list with each worker's
      independent RNG and *then* sliced by worker id, so sessions were duplicated
      across some workers and dropped from others every epoch.  Fixed: shard
      deterministically (disjoint, complete) FIRST, then shuffle within the shard.

  C1 — deleted the cap.set(CAP_PROP_POS_FRAMES, tick_idx) video fallback.  Tick
      indices are NOT video frame indices (they drift past every loading screen /
      dropped frame), so that path silently loaded the wrong frame.  Frames now
      come only from the precomputed JPEGs (extracted correctly via
      frame_mapping.jsonl by precompute_frames.py); a missing JPEG skips the tick.

  The precomputed-SigLIP-feature path is removed: viblock5 always runs online
  SigLIP (voxel slot tokens live inside the encoder, so the backbone must be in
  the forward pass), exactly as viblock4 did.

Camera-relative coordinate convention is unchanged from viblock3 (yaw snapped to
nearest 90°, grid rotated so the camera faces +Z; world Y unchanged — no pitch).
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pum.data.vis_dataset import SMELTED_ROOT

STEM               = "viblock3_data.npz"   # reuse the per-session metadata viblock3/4 produced
FRAMES_DIR         = "frames"
CAM_CLASSES_DIR    = "cam_classes"
CAM_VIS_DIR        = "cam_vis"
LOADING_SCREEN_STD = 8.0


def build_class_lut(vocab_path: Path, max_sid: int = 65535) -> np.ndarray:
    """
    Map block state IDs → class indices.
      0          → 0  (air)
      in-vocab   → 1..n_classes  (block classes shifted by 1 to free slot 0 for air)
      OOV / air  → -1            (ignored in CE loss)
    Returns (max_sid+1,) int64 array.
    """
    import json
    with open(vocab_path) as f:
        vocab = json.load(f)
    lut = np.full(max_sid + 1, -1, dtype=np.int64)
    lut[0] = 0   # air is class 0
    for sid_str, cls in vocab["sid_to_class"].items():
        sid = int(sid_str)
        if 0 < sid <= max_sid:
            lut[sid] = int(cls) + 1   # shift so 0 is reserved for air
    return lut


class ViBlock5Dataset(IterableDataset):
    """
    Iterable dataset for viblock5 (online SigLIP mode only).

    Yields (frame_rgb, cam_blocks, cam_vis) per tick.  cam_blocks and cam_vis are
    read from the precomputed cam_classes/ and cam_vis/ directories; both are in
    the same camera-relative (yaw-snapped) frame.
    """

    def __init__(self, session_names: list[str], vocab_path: Path,
                 shuffle: bool = True, require_vis: bool = True):
        self.session_names = list(session_names)
        self.shuffle       = shuffle
        self.class_lut     = build_class_lut(vocab_path)
        # If True, skip ticks whose cam_vis is missing rather than training on
        # them unmasked.  Keep True unless you have deliberately decided to fall
        # back to viblock4-style full-grid supervision.
        self.require_vis   = require_vis

    def __iter__(self):
        sessions = list(self.session_names)

        # ── T2 fix: deterministic disjoint shard BEFORE any shuffle ──────────────
        # sessions[wi.id::num_workers] gives each worker a unique, non-overlapping
        # slice that together cover every session exactly once.  Only after that do
        # we shuffle *within* the worker's own shard.
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            sessions = sessions[wi.id::wi.num_workers]
        if self.shuffle:
            np.random.shuffle(sessions)

        lut = self.class_lut

        for session in sessions:
            labels_dir  = SMELTED_ROOT / session / "labels"
            data_path   = labels_dir / STEM
            cam_cls_dir = labels_dir / CAM_CLASSES_DIR
            cam_vis_dir = labels_dir / CAM_VIS_DIR
            frames_dir  = labels_dir / FRAMES_DIR

            if not (data_path.exists() and cam_cls_dir.exists() and frames_dir.exists()):
                continue
            if self.require_vis and not cam_vis_dir.exists():
                log.warning("no cam_vis/ for %s — run precompute_cam_vis.py; skipping session",
                            session)
                continue

            d            = np.load(data_path)
            tick_indices = d["tick_indices"]

            order = np.arange(len(tick_indices))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                tick_idx = int(tick_indices[k])

                cls_path = cam_cls_dir / f"{tick_idx:06d}.npy"
                if not cls_path.exists():
                    continue
                cam_classes = np.load(str(cls_path)).astype(np.int64)   # (32,32,32)

                # ── Visibility mask (E10) ──────────────────────────────────────
                vis_path = cam_vis_dir / f"{tick_idx:06d}.npy"
                if vis_path.exists():
                    cam_vis = np.load(str(vis_path)).astype(np.uint8)   # (32,32,32)
                elif self.require_vis:
                    continue   # never fabricate an all-visible mask
                else:
                    cam_vis = np.ones((32, 32, 32), dtype=np.uint8)

                # ── Frame (C1 fix: precomputed JPEG only, no video fallback) ───
                jpg = frames_dir / f"{tick_idx:06d}.jpg"
                if not jpg.exists():
                    continue
                bgr = cv2.imread(str(jpg))
                if bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.std() < LOADING_SCREEN_STD:
                    continue   # loading screen / blank frame

                yield frame_rgb, cam_classes, cam_vis


def collate_fn(batch):
    """
    Online SigLIP mode collate.
      frames     : list[(H,W,3) uint8]   — kept as a list for the SigLIP processor
      cam_blocks : (B, 32, 32, 32) int64
      cam_vis    : (B, 32, 32, 32) uint8
    """
    frames     = [b[0] for b in batch]
    cam_blocks = torch.stack([torch.from_numpy(b[1]) for b in batch])
    cam_vis    = torch.stack([torch.from_numpy(b[2]) for b in batch])
    return frames, cam_blocks, cam_vis
