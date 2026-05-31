"""
precompute_siglip_feats.py — Extract and cache frozen SigLIP patch features.

WHY: SigLIP forward (92M params, frozen) dominates epoch time at ~200ms/batch.
Precomputing once eliminates all video decode and SigLIP cost from training,
cutting epoch time from ~22 min → ~2 min (pure MLP head over cached features).

For each session, reads viblock2_data.npz (tick_indices, block_uvs, block_offsets)
and the raw video. Runs SigLIP on frames in GPU batches, extracts the per-block
patch feature via UV lookup, saves float16 to viblock2_feats.npz.

Output per session: {session}/labels/viblock2_feats.npz
  block_feats  (N_total, 768) float16  — SigLIP patch feat for each visible block
  (N_total matches block_uvs in viblock2_data.npz)

Run once on Brev BEFORE training (needs GPU):
  PYTHONPATH=/home/vvm33/blockscope python precompute_siglip_feats.py [--device cuda:0]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SIGLIP_W = 40
SIGLIP_H = 22
DATA_STEM = "viblock2_data.npz"
FEAT_STEM = "viblock2_feats.npz"
GPU_BATCH  = 32   # frames per SigLIP forward — fits easily in 80GB


def load_siglip(device: torch.device):
    from transformers import AutoModel, AutoProcessor
    backbone = "google/siglip2-base-patch16-naflex"
    log.info("Loading SigLIP %s", backbone)
    model  = AutoModel.from_pretrained(backbone, cache_dir="/data/vvm33/hf_cache")
    vision = model.vision_model.eval().to(device)
    for p in vision.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(backbone, cache_dir="/data/vvm33/hf_cache")
    log.info("Loaded (%d params)", sum(p.numel() for p in vision.parameters()))
    return vision, proc


@torch.no_grad()
def siglip_grid_batch(vision, proc, frames_bgr: list, device: torch.device) -> torch.Tensor:
    """List of (H,W,3) BGR uint8 → (B, 22, 40, 768) float32 on CPU."""
    from PIL import Image
    pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    inp  = proc(images=pil, return_tensors="pt", max_num_patches=880)
    out  = vision(
        pixel_values        = inp["pixel_values"].to(device),
        attention_mask      = inp["pixel_attention_mask"].to(device),
        spatial_shapes      = inp["spatial_shapes"].to(device),
    )
    h = out.last_hidden_state   # (B, 880, 768)
    B = h.shape[0]
    return h.view(B, SIGLIP_H, SIGLIP_W, 768).cpu()


def uv_to_patch(uvs: np.ndarray):
    col = np.floor(uvs[:, 0] * SIGLIP_W).astype(np.int32).clip(0, SIGLIP_W - 1)
    row = np.floor(uvs[:, 1] * SIGLIP_H).astype(np.int32).clip(0, SIGLIP_H - 1)
    return row, col


def precompute_session(session: str, vision, proc, device: torch.device) -> bool:
    labels    = SMELTED_ROOT / session / "labels"
    data_path = labels / DATA_STEM
    feat_path = labels / FEAT_STEM
    video_path = RAW_ROOT / session / "video.mp4"

    if feat_path.exists():
        log.info("skip %s (feats exist)", session)
        return True
    if not data_path.exists() or not video_path.exists():
        log.warning("skip %s (missing data/video)", session)
        return False

    data         = np.load(data_path)
    tick_indices = data["tick_indices"]   # (K,)
    offsets      = data["block_offsets"]  # (K+1,)
    block_uvs    = data["block_uvs"]      # (N, 2)
    K            = len(tick_indices)

    cap      = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all needed frames in one sequential-ish pass
    # (random seek per tick — unavoidable with shuffled ticks)
    frames_by_k: dict[int, np.ndarray] = {}
    for k in range(K):
        tidx = int(tick_indices[k])
        if tidx >= n_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, tidx)
        ok, bgr = cap.read()
        if ok:
            frames_by_k[k] = bgr
    cap.release()

    if not frames_by_k:
        log.warning("skip %s (no readable frames)", session)
        return False

    # Total visible blocks
    N_total     = int(offsets[-1])
    all_feats   = np.zeros((N_total, 768), dtype=np.float16)
    valid_ks    = sorted(frames_by_k.keys())

    # Process in GPU batches
    for batch_start in range(0, len(valid_ks), GPU_BATCH):
        batch_ks   = valid_ks[batch_start: batch_start + GPU_BATCH]
        frames_bgr = [frames_by_k[k] for k in batch_ks]

        grids = siglip_grid_batch(vision, proc, frames_bgr, device)  # (B, 22, 40, 768)

        for i, k in enumerate(batch_ks):
            s, e = int(offsets[k]), int(offsets[k + 1])
            if s >= e:
                continue
            uvs      = block_uvs[s:e]              # (n, 2)
            rows, cols = uv_to_patch(uvs)
            feats    = grids[i, rows, cols, :]     # (n, 768) float32
            all_feats[s:e] = feats.numpy().astype(np.float16)

    np.savez_compressed(feat_path, block_feats=all_feats)
    log.info("%s: saved %d block feats (%.1f MB)",
             session, N_total, all_feats.nbytes / 1e6)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device   = torch.device(args.device)
    vision, proc = load_siglip(device)

    sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / DATA_STEM).exists()
    )
    log.info("Processing %d sessions", len(sessions))

    ok = 0
    for s in sessions:
        ok += precompute_session(s, vision, proc, device)
    log.info("Done: %d/%d", ok, len(sessions))


if __name__ == "__main__":
    main()
