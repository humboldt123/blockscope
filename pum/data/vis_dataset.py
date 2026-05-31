"""
vis_dataset.py — Visible-block recognition dataset.

Task: given a video frame, predict Furnace's 32³ binary visibility mask m.
m[i,j,k] = 1 if the non-air block at grid position (i,j,k) is visible
from the camera; 0 if it exists but is occluded or off-screen.

Ground truth comes from load_tick_npz (Furnace's raycaster output).
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset

log = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 360
WINDOW       = 32
EYE_HEIGHT   = 1.62
MIN_DIST     = 0.6

FURNACE_PYTHON = "/home/vvm33/blockscope/furnace/pipeline/src/python"
RAW_ROOT       = Path("/data/vvm33/BLOCKSCOPE_DATA")
SMELTED_ROOT   = Path("/data/vvm33/SMELTED_DATA")

DINO_PATCH = 14
PATCHES_W  = IMG_W // DINO_PATCH   # 45
PATCHES_H  = IMG_H // DINO_PATCH   # 25
_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TICK_STRIDE = 20


def _camera_basis(yaw_deg: float, pitch_deg: float):
    """(fwd, right, up) unit vectors in Minecraft world coords (X=east, Y=up, Z=south)."""
    yr = np.radians(yaw_deg)
    pr = np.radians(pitch_deg)
    fwd   = np.array([-np.sin(yr)*np.cos(pr), -np.sin(pr),  np.cos(yr)*np.cos(pr)])
    right = np.array([ np.cos(yr),              0.0,          np.sin(yr)])
    up    = np.cross(fwd, right)
    return fwd, right, up


def project_nonair_visibility(blocks_u16: np.ndarray, m: np.ndarray,
                               px: int, py: int, pz: int,
                               yaw: float, pitch: float,
                               fov_deg: float) -> tuple:
    """
    Project all non-air voxels into the camera frustum and return their
    screen UVs and Furnace binary visibility labels.

    Parameters
    ----------
    blocks_u16 : (32,32,32) uint16  block-state IDs (0 = air)
    m          : (32,32,32) uint8   Furnace visibility mask (1 = visible)
    px,py,pz   : int  player block coords (world, grid centre)
    yaw,pitch  : float  camera angles in Minecraft degrees
    fov_deg    : float  horizontal FOV in degrees

    Returns
    -------
    screen_uv  : (N, 2) float32  normalised screen coords in [0,1]x[0,1]
    vis_labels : (N,)   float32  binary: 1=visible, 0=occluded
    xi, yi, zi : (N,)   int32   grid indices (for m reconstruction at eval)
    block_sids : (N,)   int32   block state IDs at each position
    """
    xi, yi, zi = np.where(blocks_u16 > 0)
    if len(xi) == 0:
        empty_i = np.empty(0, dtype=np.int32)
        return (np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                empty_i, empty_i, empty_i, empty_i)

    wx = (xi - WINDOW // 2 + px + 0.5).astype(np.float64)
    wy = (yi - WINDOW // 2 + py + 0.5).astype(np.float64)
    wz = (zi - WINDOW // 2 + pz + 0.5).astype(np.float64)

    cx, cy, cz = px + 0.5, py + EYE_HEIGHT, pz + 0.5
    dx, dy, dz = wx - cx, wy - cy, wz - cz

    fwd, right, up = _camera_basis(yaw, pitch)
    dot_f = dx*fwd[0] + dy*fwd[1] + dz*fwd[2]
    dot_r = dx*right[0] + dy*right[1] + dz*right[2]
    dot_u = dx*up[0] + dy*up[1] + dz*up[2]

    fov_x  = np.radians(fov_deg)
    fov_y  = fov_x * (IMG_H / IMG_W)
    tan_x, tan_y = np.tan(fov_x / 2), np.tan(fov_y / 2)

    in_front = dot_f > MIN_DIST
    safe_f   = np.where(dot_f > 0, dot_f, 1e-9)
    ndc_x    = np.where(in_front, (dot_r / safe_f) / tan_x, 2.0)
    ndc_y    = np.where(in_front, -(dot_u / safe_f) / tan_y, 2.0)

    mask = in_front & (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)

    u    = ((ndc_x[mask] + 1) / 2).astype(np.float32)
    v    = ((ndc_y[mask] + 1) / 2).astype(np.float32)
    vis  = m[xi[mask], yi[mask], zi[mask]].astype(np.float32)
    sids = blocks_u16[xi[mask], yi[mask], zi[mask]].astype(np.int32)

    return (np.stack([u, v], axis=1), vis,
            xi[mask].astype(np.int32),
            yi[mask].astype(np.int32),
            zi[mask].astype(np.int32),
            sids)


def preprocess_frame_np(frame_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB → (3, H', W') float32 normalised, centre-cropped for DINOv2."""
    h_crop = PATCHES_H * DINO_PATCH   # 350
    w_crop = PATCHES_W * DINO_PATCH   # 630
    top  = (IMG_H - h_crop) // 2
    left = (IMG_W - w_crop) // 2
    crop = frame_rgb[top:top+h_crop, left:left+w_crop]
    t = crop.astype(np.float32) / 255.0
    t = (t - _DINO_MEAN) / _DINO_STD
    return t.transpose(2, 0, 1)   # (3, 350, 630)


def uv_to_patch_idx(screen_uvs: np.ndarray) -> tuple:
    """Map (N,2) UV coords → (row_idx, col_idx) into the (PATCHES_H, PATCHES_W) patch grid."""
    top_frac  = 5.0 / IMG_H
    left_frac = 5.0 / IMG_W
    h_frac    = (PATCHES_H * DINO_PATCH) / IMG_H
    w_frac    = (PATCHES_W * DINO_PATCH) / IMG_W
    u_crop = np.clip((screen_uvs[:, 0] - left_frac) / w_frac, 0.0, 1.0)
    v_crop = np.clip((screen_uvs[:, 1] - top_frac)  / h_frac, 0.0, 1.0)
    col = np.clip((u_crop * PATCHES_W).astype(np.int32), 0, PATCHES_W - 1)
    row = np.clip((v_crop * PATCHES_H).astype(np.int32), 0, PATCHES_H - 1)
    return row, col


VIS_STEM = "visibility_s20.npz"


class OnlineFrameDataset(IterableDataset):
    """
    Yields one tick at a time: (frame_tensor, screen_uvs, vis_labels).

    frame_tensor : (3, 350, 630) float32  DINOv2-normalised, centre-cropped
    screen_uvs   : (N, 2)        float32  screen coords for in-frustum non-air voxels
    vis_labels   : (N,)          float32  binary 1=visible, 0=occluded (Furnace m)

    Requires visibility_s20.npz in each session's labels/ dir
    (produced by precompute_visibility.py).
    """

    def __init__(self, session_names: list[str], shuffle: bool = True):
        self.session_names = session_names
        self.shuffle       = shuffle

    def __iter__(self):
        sys.path.insert(0, FURNACE_PYTHON)
        from io_helpers import load_world_states

        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        # Split sessions across DataLoader workers to avoid duplication
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(np.ceil(len(sessions) / worker_info.num_workers))
            start = worker_info.id * per_worker
            sessions = sessions[start: start + per_worker]

        for session in sessions:
            labels = SMELTED_ROOT / session / "labels"
            vis_path = labels / VIS_STEM
            if not vis_path.exists():
                log.warning("No visibility file for %s — run precompute_visibility.py", session)
                continue

            px_arr, py_arr, pz_arr, blocks = load_world_states(labels)

            vis_data    = np.load(vis_path)
            tick_indices = vis_data["tick_indices"].tolist()   # (K,) int
            visibility   = vis_data["visibility"]              # (K, 32, 32, 32) uint8

            raw = RAW_ROOT / session
            with open(raw / "ticks.jsonl") as f:
                ticks = [json.loads(line) for line in f]

            video_path = str(raw / "video.mp4")
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            order = list(range(len(tick_indices)))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                tick_idx = tick_indices[k]
                if tick_idx >= len(ticks) or tick_idx >= n_frames:
                    continue

                m      = visibility[k]                          # (32,32,32) uint8
                tick   = ticks[tick_idx]
                player = tick.get("player", tick)
                yaw    = float(player["yaw"])
                pitch  = float(player["pitch"])
                fov    = float(player.get("fov", 70.0))
                px     = int(px_arr[tick_idx])
                py     = int(py_arr[tick_idx])
                pz     = int(pz_arr[tick_idx])

                screen_uvs, vis_labels, _, _, _, block_sids = project_nonair_visibility(
                    blocks[tick_idx], m, px, py, pz, yaw, pitch, fov
                )
                if len(vis_labels) == 0:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, min(tick_idx, n_frames - 1))
                ok, bgr = cap.read()
                if not ok:
                    continue
                frame_t = preprocess_frame_np(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                yield (torch.from_numpy(frame_t), screen_uvs, vis_labels, block_sids)

            cap.release()


def online_collate(batch):
    """Collate variable-N ticks: stack frames, keep uv/vis/sids as lists."""
    frames   = torch.stack([b[0] for b in batch])   # (B, 3, 350, 630)
    uvs_list = [b[1] for b in batch]                # list of (N_i, 2) float32
    vis_list = [b[2] for b in batch]                # list of (N_i,) float32
    sid_list = [b[3] for b in batch]                # list of (N_i,) int32
    return frames, uvs_list, vis_list, sid_list


class VisibleBlockDataset(OnlineFrameDataset):
    """
    Wraps OnlineFrameDataset, filtering to only the visible blocks (m=1).

    Yields (frame_tensor, screen_uvs, block_sids) where screen_uvs and
    block_sids contain only positions where Furnace's raycaster says m=1.
    block_sids are the classification targets for block-type prediction.
    """

    def __iter__(self):
        for frame_t, screen_uvs, vis_labels, block_sids in super().__iter__():
            vis_mask = vis_labels == 1.0
            if vis_mask.sum() == 0:
                continue
            yield frame_t, screen_uvs[vis_mask], block_sids[vis_mask]


def visible_block_collate(batch):
    """Collate VisibleBlockDataset items: stack frames, keep uvs/sids as lists."""
    frames   = torch.stack([b[0] for b in batch])   # (B, 3, 350, 630)
    uvs_list = [b[1] for b in batch]                # list of (N_i, 2) float32
    sid_list = [b[2] for b in batch]                # list of (N_i,) int32
    return frames, uvs_list, sid_list


# ── Legacy: frustum_project (block-type classification, kept for reference) ───

def frustum_project(blocks_u16: np.ndarray, s2t: dict,
                    px: int, py: int, pz: int,
                    yaw: float, pitch: float, fov_deg: float) -> tuple:
    """LEGACY: returns screen_uv and token_ids for visible non-air blocks."""
    max_state = int(blocks_u16.max()) if blocks_u16.size > 0 else 0
    lut = np.zeros(max_state + 1, dtype=np.int32)
    for sid in range(max_state + 1):
        lut[sid] = s2t.get(sid, 0)
    tokens = lut[blocks_u16]

    xi, yi, zi = np.where(tokens > 0)
    if len(xi) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)

    wx = (xi - WINDOW // 2 + px + 0.5).astype(np.float64)
    wy = (yi - WINDOW // 2 + py + 0.5).astype(np.float64)
    wz = (zi - WINDOW // 2 + pz + 0.5).astype(np.float64)

    cx, cy, cz = px + 0.5, py + EYE_HEIGHT, pz + 0.5
    dx, dy, dz = wx - cx, wy - cy, wz - cz

    fwd, right, up = _camera_basis(yaw, pitch)
    dot_f = dx*fwd[0] + dy*fwd[1] + dz*fwd[2]
    dot_r = dx*right[0] + dy*right[1] + dz*right[2]
    dot_u = dx*up[0] + dy*up[1] + dz*up[2]

    fov_x  = np.radians(fov_deg)
    fov_y  = fov_x * (IMG_H / IMG_W)
    tan_x, tan_y = np.tan(fov_x / 2), np.tan(fov_y / 2)

    in_front = dot_f > MIN_DIST
    safe_f   = np.where(dot_f > 0, dot_f, 1e-9)
    ndc_x    = np.where(in_front, (dot_r / safe_f) / tan_x, 2.0)
    ndc_y    = np.where(in_front, -(dot_u / safe_f) / tan_y, 2.0)
    mask = in_front & (np.abs(ndc_x) <= 1.0) & (np.abs(ndc_y) <= 1.0)

    u = ((ndc_x[mask] + 1) / 2).astype(np.float32)
    v = ((ndc_y[mask] + 1) / 2).astype(np.float32)
    return np.stack([u, v], axis=1), tokens[xi[mask], yi[mask], zi[mask]].astype(np.int32)


# ── Legacy FeatureDataset (pre-extracted features, kept for reference) ────────

class FeatureDataset(IterableDataset):
    def __init__(self, feature_dir: Path, split: str = "train", shuffle: bool = True):
        self.npz_paths = sorted((Path(feature_dir) / split).glob("*.npz"))
        self.shuffle   = shuffle
        self._n = sum(len(np.load(p)["token_ids"]) for p in self.npz_paths)
        log.info("FeatureDataset[%s]: %d items from %d files",
                 split, self._n, len(self.npz_paths))
        self.items = [(p, None) for p in self.npz_paths]

    def _load(self, path):
        return np.load(path)

    def __len__(self):
        return self._n

    def __iter__(self):
        paths = list(self.npz_paths)
        if self.shuffle:
            np.random.shuffle(paths)
        for path in paths:
            d     = np.load(path)
            feats = d["patch_feats"]
            toks  = d["token_ids"]
            idx   = np.arange(len(toks))
            if self.shuffle:
                np.random.shuffle(idx)
            for i in idx:
                yield torch.from_numpy(feats[i].copy()), int(toks[i])
