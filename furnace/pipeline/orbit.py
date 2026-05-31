"""
orbit.py — diagnostic orbit video for a single tick.

Renders a full 360° orbit around the player's camera position for one tick,
showing either all visible blocks (default) or all blocks (--show-invisible).

Usage:
  uv run python -m pipeline.orbit --session path/to/session --tick 1234 --output orbit.mp4
      [--radius 15] [--elevation 5] [--frames 72] [--fps 24] [--show-invisible]
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
MAGIC = b"WSBIN001"
WINDOW = 32


# ── Orbit camera path ──────────────────────────────────────────────────────────

def _yaw_pitch_from_lookat(
    from_xyz: tuple[float, float, float],
    to_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    """
    Return (yaw, pitch) in Minecraft conventions (degrees):
      yaw=0 → south (+Z), yaw increases clockwise looking down.
      pitch > 0 → looking down.
    """
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dz = to_xyz[2] - from_xyz[2]
    horiz = math.sqrt(dx * dx + dz * dz)
    pitch = math.degrees(-math.atan2(dy, horiz))   # look down → positive
    yaw = math.degrees(math.atan2(-dx, dz))
    return yaw, pitch


def orbit_cameras(
    cx: float, cy: float, cz: float,
    radius: float, elevation: float,
    n_frames: int,
) -> list[dict]:
    """Generate n_frames orbit camera dicts looking at (cx, cy, cz)."""
    cams = []
    for i in range(n_frames):
        theta = 2 * math.pi * i / n_frames
        cam_x = cx + radius * math.cos(theta)
        cam_y = cy + elevation
        cam_z = cz + radius * math.sin(theta)
        yaw, pitch = _yaw_pitch_from_lookat((cam_x, cam_y, cam_z), (cx, cy, cz))
        cams.append({
            "x": cam_x, "y": cam_y, "z": cam_z,
            "yaw": yaw, "pitch": pitch, "fov": 70.0,
        })
    return cams


# ── world_states.bin: single-tick loader ──────────────────────────────────────

def _load_one_tick(labels_dir: Path, tick_idx: int):
    path = labels_dir / "world_states.bin"
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        tick_count = struct.unpack("<I", f.read(4))[0]
        if tick_idx >= tick_count:
            raise IndexError(
                f"Tick {tick_idx} out of range (file has {tick_count} ticks)"
            )
        px = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        py = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        pz = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        # Seek to the right block slice
        f.seek(tick_idx * WINDOW ** 3 * 2, 1)  # from current position
        raw = f.read(WINDOW ** 3 * 2)
        world_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(
            WINDOW, WINDOW, WINDOW
        ).copy()
    return int(px[tick_idx]), int(py[tick_idx]), int(pz[tick_idx]), world_u16


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Render diagnostic orbit video.")
    parser.add_argument("--session",  required=True, type=Path)
    parser.add_argument("--tick",     required=True, type=int)
    parser.add_argument("--output",   default="orbit.mp4")
    parser.add_argument("--radius",   type=float, default=15.0)
    parser.add_argument("--elevation",type=float, default=5.0)
    parser.add_argument("--frames",   type=int,   default=72)
    parser.add_argument("--fps",      type=int,   default=24)
    parser.add_argument("--show-invisible", action="store_true",
                        help="Render all blocks, not just visible ones")
    parser.add_argument("--version",  default="1.19.4")
    args = parser.parse_args()

    session_dir = Path(args.session)
    labels_dir  = session_dir / "labels"

    # ── Load tick pose ─────────────────────────────────────────────────────────
    ticks = []
    for line in (session_dir / "ticks.jsonl").read_text().splitlines():
        if line.strip():
            ticks.append(json.loads(line))

    if args.tick >= len(ticks):
        raise SystemExit(f"Tick {args.tick} out of range (max {len(ticks) - 1})")

    pose = ticks[args.tick]["player"]

    # ── Load world state ───────────────────────────────────────────────────────
    pbp_x, pbp_y, pbp_z, world_u16 = _load_one_tick(labels_dir, args.tick)
    pbp = (pbp_x, pbp_y, pbp_z)

    # ── Setup renderer ─────────────────────────────────────────────────────────
    cache_dir = _ROOT / "cache"
    from .renderer import Renderer
    renderer = Renderer(cache_dir, args.version)

    # ── Get or compute visibility mask ─────────────────────────────────────────
    npz_path = labels_dir / f"tick_{args.tick:05d}.npz"
    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        m = data["m"]
        print(f"Loaded visibility mask from {npz_path}")
    else:
        print("Computing visibility mask ...")
        m = renderer.compute_visibility(world_u16, pose, pbp)

    # ── Build orbit cameras ────────────────────────────────────────────────────
    cam = pose.get("camera", pose)
    cx, cy, cz = float(cam["x"]), float(cam["y"]), float(cam["z"])
    orbit_cams = orbit_cameras(cx, cy, cz, args.radius, args.elevation, args.frames)

    # ── Render frames ──────────────────────────────────────────────────────────
    print(f"Rendering {args.frames} frames ...")
    render_mask = None if args.show_invisible else m
    frames = []
    for i, orbit_cam in enumerate(orbit_cams):
        rgb = renderer.render_rgb(world_u16, pose, pbp,
                                  mask=render_mask, orbit_cam=orbit_cam)
        frames.append(rgb)
        if (i + 1) % 12 == 0:
            print(f"  {i + 1} / {args.frames}")

    renderer.close()

    # ── Encode to MP4 via PyAV ─────────────────────────────────────────────────
    import av
    output_path = str(args.output)
    container = av.open(output_path, "w")
    stream = container.add_stream("h264", rate=args.fps)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    for rgb in frames:
        av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)

    for pkt in stream.encode():
        container.mux(pkt)

    container.close()
    print(f"Saved orbit video: {output_path}")


if __name__ == "__main__":
    main()
