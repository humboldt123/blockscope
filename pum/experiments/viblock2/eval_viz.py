"""
eval_viz.py — Visualize viblock2 predictions vs ground truth.

Three-panel output per tick:
  Left   : actual video frame
  Center : Furnace ground-truth render (real block textures)
  Right  : Furnace predicted render (predicted block types, same textures)

Uses precomputed SigLIP features (viblock2_feats.npz) — no GPU required for
backbone; only the MLP head runs at eval time.

Usage:
  PYTHONPATH=/home/vvm33/blockscope python eval_viz.py \\
    [--checkpoint /data/vvm33/pum_checkpoints/viblock2/best_finetune.pt] \\
    [--session session_XXXXXXXX]   (default: first val session) \\
    [--n_ticks 8]                  (how many ticks to render) \\
    [--out_dir /tmp/viblock2_eval]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT
from pum.experiments.viblock2.train import ViBlock2Head, load_vocab, get_session_names

from visualize import render_perspective_view, IMG_W, IMG_H
from io_helpers import load_world_states

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_STEM  = "viblock2_data.npz"
FEATS_STEM = "viblock2_feats.npz"
VIS_STEM   = "visibility_s20.npz"


def build_class_to_sid(vocab_path: Path) -> dict[int, int]:
    """Invert sid_to_class: for each class pick the smallest matching SID."""
    with open(vocab_path) as f:
        v = json.load(f)
    class_to_sid: dict[int, int] = {}
    for sid_str, cls in v["sid_to_class"].items():
        sid = int(sid_str)
        cls = int(cls)
        if cls not in class_to_sid or sid < class_to_sid[cls]:
            class_to_sid[cls] = sid
    return class_to_sid


def load_ticks_jsonl(session: str) -> dict[int, dict]:
    path = RAW_ROOT / session / "ticks.jsonl"
    ticks = {}
    for line in path.read_text().splitlines():
        if line:
            t = json.loads(line)
            ticks[int(t["tick"])] = t
    return ticks


def extract_video_frame(session: str, tick_num: int) -> np.ndarray | None:
    """Extract the video frame mapped to tick_num (sequential decode fallback)."""
    try:
        import av
        mapping_path = RAW_ROOT / session / "frame_mapping.jsonl"
        tick_to_frame = {}
        for line in mapping_path.read_text().splitlines():
            if line:
                obj = json.loads(line)
                corrected = max(0, obj["tick"] - 1)
                tick_to_frame[corrected] = obj["frame"]
        frame_num = tick_to_frame.get(tick_num)
        if frame_num is None:
            return None
        video_path = str(RAW_ROOT / session / "video.mp4")
        container = av.open(video_path)
        for idx, frm in enumerate(container.decode(video=0)):
            if idx == frame_num:
                img = frm.to_ndarray(format="rgb24")
                container.close()
                return img.astype(np.uint8)
        container.close()
        return None
    except Exception as e:
        log.warning("Frame extract failed for tick %d: %s", tick_num, e)
        return None


def make_three_panel(left: np.ndarray, center: np.ndarray, right: np.ndarray,
                     labels: tuple[str, str, str]) -> np.ndarray:
    """Stack three (H, W, 3) uint8 images horizontally with text labels."""
    h, w = IMG_H, IMG_W

    def resize(img):
        if img is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if img.shape[:2] != (h, w):
            img = np.array(Image.fromarray(img).resize((w, h)))
        return img

    left   = resize(left)
    center = resize(center)
    right  = resize(right)

    out = np.concatenate([left, center, right], axis=1)

    # Simple text labels burned into top-left of each panel
    try:
        import cv2
        for i, txt in enumerate(labels):
            cv2.putText(out, txt, (i * w + 6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    except ImportError:
        pass  # cv2 not available — skip labels

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/data/vvm33/pum_checkpoints/viblock2/best_finetune.pt"))
    ap.add_argument("--session",    type=str, default=None,
                    help="Session name (default: first val session)")
    ap.add_argument("--n_ticks",    type=int, default=8)
    ap.add_argument("--out_dir",    type=Path, default=Path("/tmp/viblock2_eval"))
    ap.add_argument("--device",     default="cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_path = Path(cfg["data"]["vocab_path"])
    class_lut, n_classes, class_names = load_vocab(vocab_path)
    class_to_sid = build_class_to_sid(vocab_path)
    log.info("Vocab: %d classes", n_classes)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    m_cfg  = cfg["model"]
    head   = ViBlock2Head(m_cfg["feat_dim"], m_cfg["xyz_dim"],
                          m_cfg["hidden_dim"], n_classes, m_cfg["dropout"]).to(device)

    if not args.checkpoint.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    log.info("Loaded checkpoint: top1=%.4f  epoch=%d",
             ckpt.get("top1", 0), ckpt.get("epoch", 0))

    # ── Pick session ──────────────────────────────────────────────────────────
    val_sessions = get_session_names("val", cfg["data"]["n_train"])
    if not val_sessions:
        log.error("No val sessions found")
        sys.exit(1)

    session = args.session or val_sessions[0]
    log.info("Session: %s", session)

    labels_dir = SMELTED_ROOT / session / "labels"
    if not (labels_dir / DATA_STEM).exists():
        log.error("viblock2_data.npz not found in %s", labels_dir)
        sys.exit(1)
    if not (labels_dir / FEATS_STEM).exists():
        log.error("viblock2_feats.npz not found in %s", labels_dir)
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading viblock2 data and features...")
    d = np.load(labels_dir / DATA_STEM)
    tick_indices  = d["tick_indices"]   # (K,) actual tick numbers
    block_offsets = d["block_offsets"]  # (K+1,)
    block_xyz     = d["block_xyz"]      # (N, 3) normalized (xi-16)/16
    block_sids_gt = d["block_sids"]     # (N,) uint16 GT block state IDs

    f = np.load(labels_dir / FEATS_STEM)
    block_feats = f["block_feats"]     # (N, 768) float16

    vis_data = np.load(labels_dir / VIS_STEM, allow_pickle=True)
    vis_tick_idx  = vis_data["tick_indices"]  # (V,)
    vis_masks     = vis_data["visibility"]    # (V,) each is (32,32,32) uint8

    log.info("Loading world states (may take a moment)...")
    px, py, pz, ws_blocks = load_world_states(labels_dir)
    # ws_blocks[tick_num] is the GT world state for that tick number

    ticks_jsonl = load_ticks_jsonl(session)
    log.info("Loaded %d ticks from ticks.jsonl", len(ticks_jsonl))

    # Build visibility lookup: tick_num → mask
    vis_lookup = {int(vis_tick_idx[i]): vis_masks[i] for i in range(len(vis_tick_idx))}

    # ── Sample ticks evenly ───────────────────────────────────────────────────
    K = len(tick_indices)
    step = max(1, K // args.n_ticks)
    sample_idxs = list(range(0, K, step))[:args.n_ticks]
    log.info("Rendering %d ticks from %d total precomputed ticks", len(sample_idxs), K)

    # ── Render ────────────────────────────────────────────────────────────────
    for sample_i, ki in enumerate(sample_idxs):
        tick_num = int(tick_indices[ki])
        log.info("[%d/%d] tick=%d", sample_i + 1, len(sample_idxs), tick_num)

        # Features for this tick
        f_start = int(block_offsets[ki])
        f_end   = int(block_offsets[ki + 1])
        if f_end == f_start:
            log.warning("  No blocks for tick %d, skipping", tick_num)
            continue

        feats_np = block_feats[f_start:f_end].astype(np.float32)   # (M, 768)
        xyz_np   = block_xyz[f_start:f_end]                          # (M, 3)
        sids_gt  = block_sids_gt[f_start:f_end].astype(np.int32)    # (M,)

        # Run inference
        with torch.no_grad():
            feats_t  = torch.from_numpy(feats_np).to(device)
            xyz_t    = torch.from_numpy(xyz_np).to(device)
            logits   = head(feats_t, xyz_t)
            pred_cls = logits.argmax(dim=1).cpu().numpy()  # (M,)

        # GT world state for this tick
        if tick_num >= len(ws_blocks):
            log.warning("  tick_num %d >= ws_blocks length %d", tick_num, len(ws_blocks))
            continue
        b_gt = ws_blocks[tick_num].astype(np.int32)  # (32,32,32)

        # Visibility mask
        m_vis = vis_lookup.get(tick_num)
        if m_vis is None:
            # Find nearest available visibility tick
            nearest = min(vis_lookup.keys(), key=lambda t: abs(t - tick_num))
            m_vis = vis_lookup[nearest]
        m_vis = m_vis.astype(bool)

        # Build predicted world state: copy GT, replace visible block positions
        b_pred = b_gt.copy()
        for bi in range(len(pred_cls)):
            # Grid indices from normalized xyz
            gi = int(round(float(xyz_np[bi, 0]) * 16 + 16))
            gj = int(round(float(xyz_np[bi, 1]) * 16 + 16))
            gk = int(round(float(xyz_np[bi, 2]) * 16 + 16))
            if not (0 <= gi < 32 and 0 <= gj < 32 and 0 <= gk < 32):
                continue
            rep_sid = class_to_sid.get(int(pred_cls[bi]), 0)
            b_pred[gi, gj, gk] = rep_sid

        # Pose for camera
        tick_data = ticks_jsonl.get(tick_num, {})
        player    = tick_data.get("player", {})
        pose = {
            "x":     player.get("x",   0.0),
            "y":     player.get("y",   0.0),
            "z":     player.get("z",   0.0),
            "yaw":   player.get("yaw", 0.0),
            "pitch": player.get("pitch", 0.0),
            "fov":   player.get("fov",  70.0),
        }

        # Render
        log.info("  Rendering GT...")
        render_gt   = render_perspective_view(m_vis, b_gt,   pose, fullbright=True)
        log.info("  Rendering predicted...")
        render_pred = render_perspective_view(m_vis, b_pred, pose, fullbright=True)

        # Video frame
        log.info("  Extracting video frame...")
        video_frame = extract_video_frame(session, tick_num)

        # Compute accuracy for this tick
        valid = class_lut[np.clip(sids_gt, 0, len(class_lut) - 1)] >= 0
        if valid.sum() > 0:
            gt_cls  = class_lut[np.clip(sids_gt[valid], 0, len(class_lut) - 1)]
            top1_ok = (pred_cls[valid] == gt_cls).mean()
            acc_str = f"top1={top1_ok:.1%} ({valid.sum()}/{len(sids_gt)} in-vocab)"
        else:
            acc_str = "all OOV"

        # 3-panel image
        panel = make_three_panel(
            video_frame, render_gt, render_pred,
            ("Video", f"GT | {acc_str}", "Predicted"),
        )

        out_path = args.out_dir / f"tick_{tick_num:05d}.png"
        Image.fromarray(panel).save(str(out_path))
        log.info("  Saved: %s  (%s)", out_path, acc_str)

    log.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
