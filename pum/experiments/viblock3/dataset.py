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
    x_cam = round( x_rel * cos(yaw) + z_rel * sin(yaw) ) + 16
    z_cam = round(-x_rel * sin(yaw) + z_rel * cos(yaw) ) + 16
    y_cam = yi                      (world Y unchanged — no pitch rotation)

  where x_rel = xi - 16, z_rel = zi - 16 (world-relative block offsets,
  player grid centre at (16, 16, 16)).

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

STEM       = "viblock3_data.npz"
FRAMES_DIR = "frames"
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


def rotate_blocks_yaw(blocks: np.ndarray, yaw_deg: float) -> np.ndarray:
    """
    Rotate world-relative 32³ block grid to camera-relative coordinates.

    blocks   : (32, 32, 32) uint16  block-state IDs (0 = air)
    yaw_deg  : float                player yaw in Minecraft degrees
    returns  : (32, 32, 32) uint16  camera-relative block-state IDs

    Minecraft coordinate system: X=east, Y=up, Z=south.
    Camera right = [cos(yaw), 0, sin(yaw)] in world (X, Y, Z).
    Camera forward (horizontal) = [-sin(yaw), 0, cos(yaw)].

    Non-90° yaws cause nearest-neighbor aliasing (some grid cells get two
    world blocks; the higher-index one wins).  This is acceptable noise for
    a training signal.
    """
    yaw_rad = np.radians(yaw_deg)
    cos_y   = float(np.cos(yaw_rad))
    sin_y   = float(np.sin(yaw_rad))

    xi, yi, zi = np.where(blocks > 0)
    if len(xi) == 0:
        return np.zeros((32, 32, 32), dtype=np.uint16)

    sids  = blocks[xi, yi, zi]
    x_rel = xi.astype(np.float32) - 16.0
    z_rel = zi.astype(np.float32) - 16.0

    xi_cam = (np.round(x_rel * cos_y + z_rel * sin_y).astype(np.int32) + 16)
    zi_cam = (np.round(-x_rel * sin_y + z_rel * cos_y).astype(np.int32) + 16)
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

    Yields (frame_rgb, cam_blocks):
      frame_rgb  : (H, W, 3)    uint8   raw RGB
      cam_blocks : (32, 32, 32) int64   class indices (0=air, 1..N=block, -1=OOV)
    """

    def __init__(self, session_names: list[str], vocab_path: Path, shuffle: bool = True):
        self.session_names = list(session_names)
        self.shuffle       = shuffle
        self.class_lut     = build_class_lut(vocab_path)

    def __iter__(self):
        from io_helpers import load_world_states  # type: ignore

        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            per      = int(np.ceil(len(sessions) / wi.num_workers))
            sessions = sessions[wi.id * per: wi.id * per + per]

        lut = self.class_lut

        for session in sessions:
            labels_dir = SMELTED_ROOT / session / "labels"
            data_path  = labels_dir / STEM
            frames_dir = labels_dir / FRAMES_DIR
            video_path = RAW_ROOT / session / "video.mp4"

            if not data_path.exists() or not (labels_dir / "world_states.bin").exists():
                continue

            _, _, _, blocks = load_world_states(labels_dir)

            d            = np.load(data_path)
            tick_indices = d["tick_indices"]
            tick_yaws    = d["tick_yaws"]

            use_jpegs = frames_dir.exists()
            cap = None
            if not use_jpegs and video_path.exists():
                cap = cv2.VideoCapture(str(video_path))

            order = np.arange(len(tick_indices))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                tick_idx = int(tick_indices[k])
                yaw      = float(tick_yaws[k])

                if tick_idx >= len(blocks):
                    continue

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

                cam_blocks_u16 = rotate_blocks_yaw(blocks[tick_idx], yaw)
                flat           = cam_blocks_u16.reshape(-1)
                cam_classes    = lut[np.clip(flat, 0, len(lut) - 1)].reshape(32, 32, 32)

                yield frame_rgb, cam_classes

            if cap is not None:
                cap.release()


def collate_fn(batch):
    """Stack fixed-size targets; keep frames as a list for SigLIP preprocessing."""
    frames     = [b[0] for b in batch]
    cam_blocks = torch.stack([torch.from_numpy(b[1]) for b in batch])  # (B, 32, 32, 32)
    return frames, cam_blocks
