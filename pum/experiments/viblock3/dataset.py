"""
ViBlock3Dataset — image-only dense voxel prediction dataset.

For each tick yields:
  frame_rgb  : (H, W, 3)    uint8   — raw RGB frame (no normalisation here)
  cam_blocks : (32, 32, 32) int64   — camera-relative block class indices
               0 = air, 1..n_classes = block type, -1 = OOV (ignore in loss)

Camera-relative coordinates:
  The 32³ world block grid is rotated around the Y axis by the player's yaw
  angle so that the camera always faces the +Z direction of the output grid.
  This makes the image ↔ voxel correspondence consistent across all training
  examples regardless of where the player is looking.

  Rotation (nearest-neighbor, no pitch):
    x_cam = round( x_rel * cos(yaw) + z_rel * sin(yaw) ) + 15
    z_cam = round(-x_rel * sin(yaw) + z_rel * cos(yaw) ) + 15
    y_cam = yi                      (world Y unchanged — no pitch rotation)

  where x_rel = xi - 15, z_rel = zi - 15 (world-relative block offsets).
  parse_mcpr.js encodes cell i → world px-15+i, so player block is at i=15.

  Full raw yaw is used (cos/sin handle periodicity). The LOCAL orientation
  shown in diagnostics is the signed residual from the nearest cardinal
  (range [-45°, 45°]) — not the rotation angle itself.

  Yaw is an arbitrary real number in degrees (not bounded to [-180, 180]) —
  np.radians handles periodicity automatically via trig.
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
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT

STEM             = "viblock3_data.npz"
FRAMES_DIR       = "frames"
FEATS_DIR        = "siglip_feats"
CAM_CLASSES_DIR  = "cam_classes"
LOADING_SCREEN_STD = 8.0


def build_class_lut(vocab_path: Path, max_sid: int = 65535) -> np.ndarray:
    """
    Map block state IDs → class indices.
      0          → 0  (air)
      in-vocab   → 1..n_classes  (viblock2 classes shifted by 1 to free slot 0 for air)
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


def snap_yaw(yaw_deg: float) -> tuple[float, float]:
    """
    Snap raw yaw to nearest 90° cardinal and return (snap, local).

    snap  : 0 / 90 / 180 / 270 — used for exact block grid rotation
    local : residual in [-45°, +45°] — yaw offset the model must tolerate
    """
    s = float(round(yaw_deg / 90) * 90)
    return s, yaw_deg - s


def rotate_blocks_yaw(blocks: np.ndarray, yaw_deg: float) -> np.ndarray:
    """
    Rotate world-relative 32³ block grid to camera-relative coordinates.

    blocks   : (32, 32, 32) uint16  block-state IDs (0 = air)
    yaw_deg  : float                yaw in degrees — MUST be a multiple of 90
                                    (use snap_yaw() to get the snapped value)
    returns  : (32, 32, 32) uint16  camera-relative block-state IDs

    Minecraft coordinate system: X=east, Y=up, Z=south.
    Camera right = [cos(yaw), 0, sin(yaw)] in world (X, Y, Z).
    Camera forward (horizontal) = [-sin(yaw), 0, cos(yaw)].

    At exact 90° multiples this is a lossless permutation — no aliasing,
    no corner clipping.  Caller is responsible for snapping before calling.
    """
    yaw_rad = np.radians(yaw_deg)
    cos_y   = float(np.cos(yaw_rad))
    sin_y   = float(np.sin(yaw_rad))

    xi, yi, zi = np.where(blocks > 0)
    if len(xi) == 0:
        return np.zeros((32, 32, 32), dtype=np.uint16)

    sids  = blocks[xi, yi, zi]
    x_rel = xi.astype(np.float32) - 15.0
    z_rel = zi.astype(np.float32) - 15.0

    xi_cam = (np.round(x_rel * cos_y + z_rel * sin_y).astype(np.int32) + 15)
    zi_cam = (np.round(-x_rel * sin_y + z_rel * cos_y).astype(np.int32) + 15)
    # yi is unchanged (no pitch rotation)

    valid = (
        (xi_cam >= 0) & (xi_cam < 32) &
        (yi    >= 0) & (yi    < 32) &
        (zi_cam >= 0) & (zi_cam < 32)
    )

    cam_blocks = np.zeros((32, 32, 32), dtype=np.uint16)
    cam_blocks[xi_cam[valid], yi[valid], zi_cam[valid]] = sids[valid]
    return cam_blocks


class ViBlock3Dataset(IterableDataset):
    """
    Iterable dataset for viblock3.

    If precomputed SigLIP features exist (siglip_feats/<tick>.npy), yields:
      patch_grid : (22, 40, 768) float16  precomputed SigLIP features
      cam_blocks : (32, 32, 32)  int64    class indices (0=air, 1..N=block, -1=OOV)

    Otherwise falls back to raw frames (for online SigLIP mode):
      frame_rgb  : (H, W, 3)    uint8    raw RGB
      cam_blocks : (32, 32, 32) int64
    """

    def __init__(self, session_names: list[str], vocab_path: Path, shuffle: bool = True,
                 precomputed: bool = True):
        self.session_names = list(session_names)
        self.shuffle       = shuffle
        self.class_lut     = build_class_lut(vocab_path)
        self.precomputed   = precomputed  # if False, skip precomputed feats even if they exist

    def __iter__(self):
        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            per      = int(np.ceil(len(sessions) / wi.num_workers))
            sessions = sessions[wi.id * per: wi.id * per + per]

        lut = self.class_lut

        for session in sessions:
            labels_dir    = SMELTED_ROOT / session / "labels"
            data_path     = labels_dir / STEM
            feats_dir     = labels_dir / FEATS_DIR
            cam_cls_dir   = labels_dir / CAM_CLASSES_DIR
            frames_dir    = labels_dir / FRAMES_DIR
            video_path    = RAW_ROOT / session / "video.mp4"

            if not data_path.exists():
                continue

            d            = np.load(data_path)
            tick_indices = d["tick_indices"]
            tick_yaws    = d["tick_yaws"]

            use_cam_cls     = cam_cls_dir.exists()
            use_precomputed = self.precomputed and feats_dir.exists()

            # Only need world_states if cam_classes not precomputed
            blocks = None
            if not use_cam_cls:
                if not (labels_dir / "world_states.bin").exists():
                    continue
                from io_helpers import load_world_states  # type: ignore
                _, _, _, blocks = load_world_states(labels_dir)

            use_jpegs = frames_dir.exists()
            cap = None
            if not use_precomputed and not use_jpegs and video_path.exists():
                cap = cv2.VideoCapture(str(video_path))

            order = np.arange(len(tick_indices))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                tick_idx = int(tick_indices[k])
                yaw      = float(tick_yaws[k])

                # Get camera-relative class labels
                if use_cam_cls:
                    cls_path = cam_cls_dir / f"{tick_idx:06d}.npy"
                    if not cls_path.exists():
                        continue
                    cam_classes = np.load(str(cls_path)).astype(np.int64)  # (32,32,32)
                else:
                    if tick_idx >= len(blocks):
                        continue
                    snap, _ = snap_yaw(yaw)
                    cam_blocks_u16 = rotate_blocks_yaw(blocks[tick_idx], snap)
                    flat           = cam_blocks_u16.reshape(-1)
                    cam_classes    = lut[np.clip(flat, 0, len(lut) - 1)].reshape(32, 32, 32)

                if use_precomputed:
                    feat_path = feats_dir / f"{tick_idx:06d}.npy"
                    if not feat_path.exists():
                        continue
                    patch_grid = np.load(str(feat_path))   # (22, 40, 768) float16
                    yield patch_grid, cam_classes
                else:
                    frame_rgb = None
                    if use_jpegs:
                        jpg = frames_dir / f"{tick_idx:06d}.jpg"
                        if jpg.exists():
                            bgr = cv2.imread(str(jpg))
                            if bgr is not None:
                                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if frame_rgb is None and cap is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, tick_idx)
                        ok, bgr = cap.read()
                        if ok:
                            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if frame_rgb is None or frame_rgb.std() < LOADING_SCREEN_STD:
                        continue
                    yield frame_rgb, cam_classes

            if cap is not None:
                cap.release()


def collate_fn(batch):
    """
    Stack fixed-size targets. First element is either:
      - (22,40,768) float16 ndarray  → precomputed SigLIP features (stack into tensor)
      - (H,W,3)    uint8  ndarray    → raw frame (keep as list for SigLIP processor)
    """
    first = batch[0][0]
    cam_blocks = torch.stack([torch.from_numpy(b[1]) for b in batch])   # (B,32,32,32)
    if first.dtype == np.float16:
        # Precomputed mode: stack into (B, 22, 40, 768) float16 tensor
        patch_grids = torch.stack([torch.from_numpy(b[0]) for b in batch])
        return patch_grids, cam_blocks
    else:
        # Online SigLIP mode: keep frames as list
        frames = [b[0] for b in batch]
        return frames, cam_blocks
