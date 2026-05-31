"""
render_html.py — sample random train + val ticks, render MC frame / GT / pred, upload HTML.

Usage:
    python pum/experiments/viblock4/render_html.py \
        --checkpoint /data/vvm33/pum_checkpoints/viblock4/best_viblock4.pt \
        --n_train 4 --n_val 4
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[3]))          # blockscope root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT
from pum.experiments.viblock3.dataset import STEM, FRAMES_DIR
from pum.experiments.viblock4.train_finetune import get_session_names, _build_class_to_sid
from visualize import render_perspective_view, IMG_W, IMG_H  # type: ignore
from io_helpers import load_world_states                                           # type: ignore


VOCAB_PATH = Path("/data/vvm33/pum_checkpoints/blocktype/vocab.json")

# Minecraft yaw → cardinal: 0=South(+Z), 90=West(-X), 180=North(-Z), 270=East(+X)
def _cardinal(yaw_deg: float) -> str:
    n = yaw_deg % 360
    if n < 45 or n >= 315:
        return "S(+Z)"
    elif n < 135:
        return "W(-X)"
    elif n < 225:
        return "N(-Z)"
    else:
        return "E(+X)"



def sample_tick(session: str):
    """Return (frame_rgb, b_gt, m_vis, pose, tick_idx, yaw, pitch, session) or None."""
    labels_dir = SMELTED_ROOT / session / "labels"
    data_path  = labels_dir / STEM
    ws_path    = labels_dir / "world_states.bin"
    if not data_path.exists() or not ws_path.exists():
        return None

    d            = np.load(data_path)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]
    try:
        tick_pitches = d["tick_pitches"]
    except KeyError:
        tick_pitches = np.zeros_like(tick_yaws)

    # ticks.jsonl — needed only for camera pose (x/y/z/yaw/pitch).
    ticks_path = RAW_ROOT / session / "ticks.jsonl"
    if not ticks_path.exists():
        return None
    with open(ticks_path) as f:
        ticks_jsonl = [json.loads(l) for l in f]

    px, py, pz, blocks = load_world_states(labels_dir)

    vis       = np.load(labels_dir / "visibility_s20.npz")
    vis_ticks = vis["tick_indices"]
    vis_masks = vis["visibility"]

    k        = random.randint(0, len(tick_indices) - 1)
    tick_idx = int(tick_indices[k])
    yaw      = float(tick_yaws[k])
    pitch    = float(tick_pitches[k])

    # Load pre-extracted frame from smelted labels/frames/ (correct sync via frame_mapping).
    jpg_path = labels_dir / "frames" / f"{tick_idx:06d}.jpg"
    if not jpg_path.exists():
        return None
    frame_rgb = np.array(Image.open(jpg_path).convert("RGB"))

    b_gt = blocks[tick_idx].astype(np.int32)

    closest_vi = int(np.argmin(np.abs(vis_ticks - tick_idx)))
    m_vis = vis_masks[closest_vi].astype(bool)

    # Real camera coords (float) from ticks.jsonl.
    tick_entry = ticks_jsonl[tick_idx]
    player_e   = tick_entry.get("player", tick_entry)
    cam_e      = player_e.get("camera", player_e)
    cam_x = float(cam_e.get("x", player_e.get("x", float(px[tick_idx]))))
    cam_y = float(cam_e.get("y", player_e.get("y", float(py[tick_idx]))))
    cam_z = float(cam_e.get("z", player_e.get("z", float(pz[tick_idx]))))

    pose = {
        "x": float(px[tick_idx]), "y": float(py[tick_idx]), "z": float(pz[tick_idx]),
        "yaw": yaw, "pitch": pitch, "fov": 70.0,
    }

    return frame_rgb, b_gt, m_vis, pose, tick_idx, yaw, pitch, session, (cam_x, cam_y, cam_z)


def render_sample(sample, sid_lut):
    frame_rgb, b_gt_world, m_vis, pose, tick_idx, yaw, pitch, session, cam_xyz = sample

    # Load cam-space GT classes (rotated by raw yaw so player always faces +Z in grid).
    labels_dir = SMELTED_ROOT / session / "labels"
    cc_path    = labels_dir / "cam_classes" / f"{tick_idx:06d}.npy"
    if not cc_path.exists():
        return None
    gt_cam_cls = np.load(cc_path).astype(np.int32)
    gt_cam_sid = sid_lut[np.clip(gt_cam_cls, 0, len(sid_lut) - 1)]
    gt_cam_sid[gt_cam_cls < 0] = 0

    # Camera grid position: frac offset from the floor anchor (px,py,pz).
    cam_x, cam_y, cam_z = cam_xyz
    px_floor, py_floor, pz_floor = pose["x"], pose["y"], pose["z"]
    cam_gx = cam_x - px_floor + 15.0   # 15 + frac(cam.x)
    cam_gy = cam_y - py_floor + 15.0   # 15 + frac(cam.y) ≈ 15.62
    cam_gz = cam_z - pz_floor + 15.0   # 15 + frac(cam.z)

    # Panel 2 — pre-localization: world-space blocks rendered from actual camera pose.
    # Exposes whether the world state itself matches the video (before any rotation).
    world_pose = {"x": cam_gx, "y": cam_gy, "z": cam_gz,
                  "yaw": yaw, "pitch": pitch, "fov": 70.0}
    render_prelocal = render_perspective_view(
        (b_gt_world > 0), b_gt_world, world_pose, fullbright=True
    )

    # Panel 3 — GT cam-space: cam_classes rendered at local_yaw (residual after 90° snap).
    # Camera position is also snap-rotated so panel 3 matches panel 2 geometrically.
    cam_grid_y = cam_gy
    snap_y     = round(yaw / 90) * 90
    local_y    = yaw - snap_y          # in [-45°, +45°]

    # Rotate the fractional camera offset (cam_gx-15, cam_gz-15) by snap_y into cam-space.
    snap_rad = np.radians(snap_y)
    cs, sn = float(np.cos(snap_rad)), float(np.sin(snap_rad))
    dx, dz  = cam_gx - 15.0, cam_gz - 15.0
    cam_cx  = dx * cs + dz * sn + 15.0
    cam_cz  = -dx * sn + dz * cs + 15.0

    gt_pose = {"x": cam_cx, "y": cam_gy, "z": cam_cz, "yaw": local_y, "pitch": pitch, "fov": 70.0}
    render_gt = render_perspective_view(
        (gt_cam_sid > 0), gt_cam_sid.astype(np.int32), gt_pose, fullbright=True
    )

    # Cross layout: F-3..F+3 in left column; world-furnace + GT extend right only at F.
    offsets = [-3, -2, -1, 0, 1, 2, 3]
    main_row = offsets.index(0)
    canvas = np.zeros((len(offsets) * IMG_H, 3 * IMG_W, 3), dtype=np.uint8)
    frames_dir = SMELTED_ROOT / session / "labels" / "frames"
    for i, dt in enumerate(offsets):
        adj_jpg = frames_dir / f"{tick_idx + dt:06d}.jpg"
        if adj_jpg.exists():
            f = np.array(Image.open(adj_jpg).convert("RGB"))
        else:
            f = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        canvas[i*IMG_H:(i+1)*IMG_H, :IMG_W] = np.array(Image.fromarray(f).resize((IMG_W, IMG_H)))
    canvas[main_row*IMG_H:(main_row+1)*IMG_H, IMG_W:2*IMG_W] = render_prelocal
    canvas[main_row*IMG_H:(main_row+1)*IMG_H, 2*IMG_W:] = render_gt
    panel = canvas

    buf = io.BytesIO()
    Image.fromarray(panel).save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()

    caption = (
        f"<b>{session}</b>  tick={tick_idx} &nbsp;|&nbsp; "
        f"<span style='color:#fa8'>video F-3..F+3 (left col)</span> &nbsp; "
        f"<span style='color:#afc'>world-furnace (actual pose)</span> &nbsp; "
        f"<span style='color:#7af'>cam-furnace GT (both right of F)</span><br>"
        f"<span style='color:#aaa'>"
        f"RAW &nbsp; x={cam_x:.2f} y={cam_y:.2f} z={cam_z:.2f} &nbsp;|&nbsp; "
        f"grid ({cam_gx:.2f}, {cam_gy:.2f}, {cam_gz:.2f}) &nbsp;|&nbsp; "
        f"yaw={yaw:.1f}°  snap={snap_y:.0f}°  local={local_y:+.1f}° &nbsp; pitch={pitch:.1f}° &nbsp;|&nbsp; "
        f"facing {_cardinal(yaw)}"
        f"</span><br>"
        f"<span style='color:#7af'>"
        f"CAM-SPACE  x={cam_cx:.2f} y={cam_gy:.2f} z={cam_cz:.2f} &nbsp;|&nbsp; "
        f"render yaw={local_y:+.1f}° pitch={pitch:.1f}°"
        f"</span><br>"
        f"world blocks={int((b_gt_world>0).sum())}  "
        f"GT cam blocks={int((gt_cam_sid>0).sum())}"
    )
    return b64, caption


def build_html(train_rows, val_rows, ckpt_name: str) -> str:
    def section(title, rows, color):
        cards = ""
        for item in rows:
            if item is None:
                continue
            b64, caption = item
            cards += f"""
            <div class="card">
              <img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:6px"/>
              <p style="font-size:11px;color:#ccc;margin:6px 0 0;line-height:1.6">{caption}</p>
            </div>"""
        return f"""
        <h2 style="color:{color};border-bottom:2px solid {color};padding-bottom:6px">{title}</h2>
        <div class="grid">{cards}</div>"""

    body = section("Train samples", train_rows, "#4caf50") + section("Val samples", val_rows, "#2196f3")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>viblock4 renders — {ckpt_name}</title>
<style>
  body{{background:#111;color:#eee;font-family:monospace;padding:16px;max-width:1600px;margin:0 auto}}
  h1{{color:#fff;font-size:18px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(720px,1fr));gap:16px;margin-bottom:32px}}
  .card{{background:#1e1e1e;border-radius:8px;padding:10px}}
  .legend{{font-size:11px;color:#888;margin-bottom:24px}}
</style>
</head>
<body>
<h1>viblock4 renders — {ckpt_name}</h1>
<p class="legend">Each card: left column = video frames F-3..F+3 | right of main frame = <span style="color:#7af">cam-furnace GT (snap-rotated, local yaw)</span></p>
{body}
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=4)
    ap.add_argument("--n_val",   type=int, default=4)
    ap.add_argument("--n_train_cfg", type=int, default=73)
    args = ap.parse_args()

    cls_to_sid = _build_class_to_sid(VOCAB_PATH)
    sid_lut    = np.zeros(max(cls_to_sid.values()) + 1, dtype=np.uint16)
    for cls, sid in cls_to_sid.items():
        if cls < len(sid_lut):
            sid_lut[cls] = sid

    train_sessions = get_session_names("train", args.n_train_cfg)
    val_sessions   = get_session_names("val",   args.n_train_cfg)

    random.shuffle(train_sessions)
    random.shuffle(val_sessions)

    def collect(sessions, n):
        rows = []
        for s in sessions:
            if len(rows) >= n:
                break
            sample = sample_tick(s)
            if sample is None:
                continue
            print(f"  rendering {s} tick={sample[4]} pitch={sample[6]:.1f}°...")
            result = render_sample(sample, sid_lut)
            if result is not None:
                rows.append(result)
        return rows

    print(f"Rendering {args.n_train} train samples...")
    train_rows = collect(train_sessions, args.n_train)
    print(f"Rendering {args.n_val} val samples...")
    val_rows   = collect(val_sessions,   args.n_val)

    html = build_html(train_rows, val_rows, "no-model")
    out  = Path("/tmp/viblock4_renders.html")
    out.write_text(html)
    print(f"Saved {out} ({out.stat().st_size // 1024} KB)")

    import subprocess
    result = subprocess.run(
        ["curl", "-s",
         "-F", "reqtype=fileupload",
         "-F", "time=72h",
         "-F", f"fileToUpload=@{out}",
         "https://litterbox.catbox.moe/resources/internals/api.php"],
        capture_output=True, text=True
    )
    url = result.stdout.strip()
    if url.startswith("http"):
        print(f"\n✅ View on your phone: {url}\n")
    else:
        print(f"Upload failed: {result.stderr}")
        print(f"File at: {out}")


if __name__ == "__main__":
    main()
