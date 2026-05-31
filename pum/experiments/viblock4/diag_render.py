"""
diag_render.py — verify render_wandb_samples produces non-empty images.
Runs on CPU (no GPU needed), saves strips to /tmp/diag_render_{i}.jpg.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/vvm33/blockscope")
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT
from pum.experiments.viblock3.dataset import STEM
from pum.experiments.viblock4.train_finetune import _build_class_to_sid
from visualize import render_perspective_view, IMG_W, IMG_H  # type: ignore

VOCAB_PATH = Path("/data/vvm33/pum_checkpoints/blocktype/vocab.json")

cls_to_sid = _build_class_to_sid(VOCAB_PATH)
max_cls = max(cls_to_sid.keys(), default=0)
sid_lut = np.zeros(max_cls + 1, dtype=np.int32)
for cls, sid in cls_to_sid.items():
    if cls <= max_cls:
        sid_lut[cls] = sid

sessions = sorted(
    d.name for d in SMELTED_ROOT.iterdir()
    if d.is_dir() and (d / "labels" / STEM).exists()
)
print(f"Found {len(sessions)} sessions")

saved = 0
for session in sessions:
    if saved >= 3:
        break
    labels_dir = SMELTED_ROOT / session / "labels"
    cc_dir     = labels_dir / "cam_classes"
    frames_dir = labels_dir / "frames"
    data_path  = labels_dir / STEM
    if not (data_path.exists() and cc_dir.exists() and frames_dir.exists()):
        continue

    d            = np.load(data_path)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]
    try:
        tick_pitches = d["tick_pitches"]
    except KeyError:
        tick_pitches = np.zeros_like(tick_yaws)

    candidates = [k for k in range(len(tick_indices))
                  if (frames_dir / f"{int(tick_indices[k]):06d}.jpg").exists()
                  and (cc_dir / f"{int(tick_indices[k]):06d}.npy").exists()]
    if not candidates:
        print(f"  {session}: no valid ticks")
        continue

    k = None
    for ck in candidates[::max(1, len(candidates)//20)][:20]:
        t = int(tick_indices[ck])
        if np.load(cc_dir / f"{t:06d}.npy").astype(np.int32).clip(0, None).sum() > 0:
            k = ck
            break
    if k is None:
        print(f"  {session}: all sampled ticks have 0 GT blocks, skipping")
        continue
    tick_idx = int(tick_indices[k])
    yaw      = float(tick_yaws[k])
    pitch    = float(tick_pitches[k])

    frame_rgb  = np.array(Image.open(frames_dir / f"{tick_idx:06d}.jpg").convert("RGB"))
    gt_cam_cls = np.load(cc_dir / f"{tick_idx:06d}.npy").astype(np.int32)
    gt_cam_sid = sid_lut[np.clip(gt_cam_cls, 0, max_cls)]
    gt_cam_sid[gt_cam_cls < 0] = 0

    n_gt_blocks = int((gt_cam_sid > 0).sum())
    snap_y  = round(yaw / 90) * 90
    local_y = yaw - snap_y
    pose    = {"x": 15.5, "y": 15.62, "z": 15.5, "yaw": local_y, "pitch": pitch, "fov": 70.0}

    print(f"  {session} tick={tick_idx} yaw={yaw:.1f} local={local_y:+.1f} pitch={pitch:.1f} GT blocks={n_gt_blocks}")

    r_gt   = render_perspective_view((gt_cam_sid > 0), gt_cam_sid, pose, fullbright=True)
    # Fake pred: all air (worst case — should show empty render)
    r_pred = render_perspective_view(np.zeros((32,32,32), bool), np.zeros((32,32,32), np.int32), pose, fullbright=True)

    f_vid  = np.array(Image.fromarray(frame_rgb).resize((IMG_W, IMG_H)))
    strip  = np.concatenate([f_vid, r_gt, r_pred], axis=1)

    n_gt_px   = int((r_gt   > 20).any(axis=2).sum())
    n_pred_px = int((r_pred > 20).any(axis=2).sum())
    print(f"    GT render non-black pixels: {n_gt_px}  pred render: {n_pred_px}")

    out = f"/tmp/diag_render_{saved}.jpg"
    Image.fromarray(strip).save(out, quality=88)
    print(f"    Saved {out}")
    saved += 1

print(f"\nDone. {saved} strips saved to /tmp/diag_render_*.jpg")
