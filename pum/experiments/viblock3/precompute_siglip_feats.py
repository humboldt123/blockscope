"""
Precompute SigLIP patch grids for all viblock3 sessions.

Saves (22, 40, 768) float16 per tick to:
  SMELTED_ROOT/<session>/labels/siglip_feats/<tick_idx:06d>.npy

Run in parallel across 6 GPUs — each process handles a shard of sessions:
  for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=/home/vvm33/blockscope \
      python precompute_siglip_feats.py --gpu_shard $i --n_shards 6 &
  done
  wait

Or single GPU:
  PYTHONPATH=/home/vvm33/blockscope python precompute_siglip_feats.py
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT
from pum.experiments.viblock3.dataset import STEM, FRAMES_DIR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATS_DIR  = "siglip_feats"
SIGLIP_H   = 22
SIGLIP_W   = 40
BATCH_SIZE = 16
BACKBONE   = "google/siglip2-base-patch16-naflex"


def load_siglip(device):
    from transformers import AutoModel, AutoProcessor
    log.info("Loading SigLIP on %s", device)
    model = AutoModel.from_pretrained(BACKBONE, cache_dir="/data/vvm33/hf_cache")
    vision = model.vision_model.to(device).eval()
    proc   = AutoProcessor.from_pretrained(BACKBONE, cache_dir="/data/vvm33/hf_cache")
    for p in vision.parameters():
        p.requires_grad_(False)
    return vision, proc


def run_siglip_batch(vision, proc, frames_rgb: list, device) -> np.ndarray:
    """Run SigLIP on a list of RGB frames → (B, 22, 40, 768) float16."""
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames_rgb]
    inp = proc(images=pil, return_tensors="pt", max_num_patches=880)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = vision(
            pixel_values   = inp["pixel_values"].to(device),
            attention_mask = inp["pixel_attention_mask"].to(device),
            spatial_shapes = inp["spatial_shapes"].to(device),
        )
    grid = out.last_hidden_state.view(len(frames_rgb), SIGLIP_H, SIGLIP_W, 768)
    return grid.to(torch.float16).cpu().numpy()


def precompute_session(session: str, vision, proc, device, force: bool = False) -> int:
    labels_dir = SMELTED_ROOT / session / "labels"
    feats_dir  = labels_dir / FEATS_DIR
    data_path  = labels_dir / STEM
    frames_dir = labels_dir / FRAMES_DIR
    video_path = RAW_ROOT / session / "video.mp4"

    if not data_path.exists():
        return 0

    feats_dir.mkdir(exist_ok=True)

    d            = np.load(data_path)
    tick_indices = d["tick_indices"]

    # Find which ticks still need processing
    pending = [
        int(ti) for ti in tick_indices
        if force or not (feats_dir / f"{int(ti):06d}.npy").exists()
    ]
    if not pending:
        log.info("skip %s (all %d feats exist)", session, len(tick_indices))
        return 0

    # Open video if needed (fallback when JPEG cache missing)
    use_jpegs = frames_dir.exists()
    cap = None
    if not use_jpegs and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))

    batch_frames = []
    batch_ticks  = []
    n_saved = 0

    def flush():
        nonlocal n_saved
        if not batch_frames:
            return
        grids = run_siglip_batch(vision, proc, batch_frames, device)
        for idx, ti in enumerate(batch_ticks):
            np.save(str(feats_dir / f"{ti:06d}.npy"), grids[idx])
            n_saved += 1
        batch_frames.clear()
        batch_ticks.clear()

    for ti in pending:
        frame_rgb = None
        if use_jpegs:
            jpg = frames_dir / f"{ti:06d}.jpg"
            if jpg.exists():
                bgr = cv2.imread(str(jpg))
                if bgr is not None:
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb is None and cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ti)
            ok, bgr = cap.read()
            if ok:
                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb is None:
            continue

        batch_frames.append(frame_rgb)
        batch_ticks.append(ti)
        if len(batch_frames) >= BATCH_SIZE:
            flush()

    flush()
    if cap is not None:
        cap.release()

    log.info("%s: saved %d/%d feats", session, n_saved, len(pending))
    return n_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_shard", type=int, default=0)
    ap.add_argument("--n_shards",  type=int, default=1)
    ap.add_argument("--force",     action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    vision, proc = load_siglip(device)

    sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / STEM).exists()
    )
    # Take this shard
    shard = sessions[args.gpu_shard::args.n_shards]
    log.info("Shard %d/%d: %d sessions", args.gpu_shard, args.n_shards, len(shard))

    total = sum(precompute_session(s, vision, proc, device, args.force) for s in shard)
    log.info("Done: %d feats saved", total)


if __name__ == "__main__":
    main()
