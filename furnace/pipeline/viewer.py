"""
viewer.py — side-by-side visual validation (video frame vs GPU render).

Top row:    VIDEO (left) | RECONSTR (right)
Bottom row: ORBIT — hold O to spin orbit camera one frame at a time.

Usage:
  uv run python -m pipeline.viewer --session path/to/session [--tick-offset 1]

Keys:
  ← / →          step 1 tick
  Shift+← / →    step 50 ticks
  Page Up/Down    step 50 ticks
  O (hold)        advance orbit view one frame per held tick
  Q / Esc         quit

--- Attributions ---
Frame extraction logic (seeking + sequential fallback):
    Adapted from furnace/pipeline/src/python/visualize.py :: extract_video_frame()

world_states.bin loading:
    Copied verbatim from furnace/pipeline/src/python/io_helpers.py :: load_world_states()
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Optional

import av
import numpy as np
import pygame
from PIL import Image

_ROOT = Path(__file__).resolve().parent
WIN_W, WIN_H = 1280, 720   # two rows of 360
VID_W, VID_H = 640, 360

ORBIT_FRAMES   = 72          # frames per full revolution
ORBIT_RADIUS   = 15.0
ORBIT_ELEVATION = 5.0
MAGIC = b"WSBIN001"
WINDOW = 32


# ── Video reader — sequential decode with cache ────────────────────────────────

class VideoReader:
    """
    Decodes a video sequentially, caching every frame seen.
    Forward navigation is O(1) amortised (decode the next frame in sequence).
    Backward navigation hits the cache instantly.
    Reopens from the beginning only when seeking backward past the cached range.
    """
    def __init__(self, path: Path):
        self._path = str(path)
        self._cache: dict[int, np.ndarray] = {}  # frame_num → uint8 (H,W,3)
        self._container = None
        self._cursor = -1   # last frame decoded
        self._open()

    def _open(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = av.open(self._path)
        self._decoder = self._container.decode(video=0)
        self._cursor = -1

    def get(self, frame_num: int) -> Optional[np.ndarray]:
        if frame_num in self._cache:
            return self._cache[frame_num]
        # If we need to go backward past what's cached, reopen from start
        if frame_num <= self._cursor:
            self._open()
        # Decode forward until we reach frame_num
        while self._cursor < frame_num:
            try:
                frm = next(self._decoder)
                self._cursor += 1
                self._cache[self._cursor] = frm.to_ndarray(format="rgb24").astype(np.uint8)
            except StopIteration:
                return None
        return self._cache.get(frame_num)

    def close(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass


# ── Frame mapping ──────────────────────────────────────────────────────────────

def _load_frame_mapping(session_dir: Path, tick_offset: int = 1) -> dict:
    """Maps tick array index → video frame number."""
    mapping = {}
    for line in (session_dir / "frame_mapping.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        corrected = max(0, obj["tick"] - tick_offset)
        mapping[corrected] = obj["frame"]
    return mapping


# ── world_states.bin loader ────────────────────────────────────────────────────

def load_world_states(labels_dir: Path):
    path = labels_dir / "world_states.bin"
    with open(path, "rb") as f:
        f.read(8)
        tick_count = struct.unpack("<I", f.read(4))[0]
        px = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        py = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        pz = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        raw = f.read(tick_count * WINDOW ** 3 * 2)
        blocks = np.frombuffer(raw, dtype=np.uint16).reshape(
            tick_count, WINDOW, WINDOW, WINDOW
        ).copy()
    return px, py, pz, blocks


# ── Pygame surface helpers ─────────────────────────────────────────────────────

def _rgb_to_surface(rgb: np.ndarray) -> pygame.Surface:
    h, w = rgb.shape[:2]
    if w != VID_W or h != VID_H:
        rgb = np.array(Image.fromarray(rgb).resize((VID_W, VID_H)))
    return pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))


def _blank_surface(color=(20, 20, 30)) -> pygame.Surface:
    surf = pygame.Surface((VID_W, VID_H))
    surf.fill(color)
    return surf


# ── Main ──────────────────────────────────────────────────────────────────────

def _orbit_cam(cx: float, cy: float, cz: float, frame: int) -> dict:
    theta = 2 * math.pi * frame / ORBIT_FRAMES
    ox = cx + ORBIT_RADIUS * math.cos(theta)
    oy = cy + ORBIT_ELEVATION
    oz = cz + ORBIT_RADIUS * math.sin(theta)
    dx, dy, dz = cx - ox, cy - oy, cz - oz
    horiz = math.sqrt(dx*dx + dz*dz)
    pitch = math.degrees(-math.atan2(dy, horiz))
    yaw   = math.degrees(math.atan2(-dx, dz))
    return {"x": ox, "y": oy, "z": oz, "yaw": yaw, "pitch": pitch, "fov": 70.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline side-by-side viewer.")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--tick-offset", type=int, default=1)
    parser.add_argument("--version", default="1.19.4")
    args = parser.parse_args()

    session_dir = Path(args.session)
    labels_dir  = session_dir / "labels"

    ticks = [json.loads(l) for l in (session_dir/"ticks.jsonl").read_text().splitlines() if l.strip()]
    frame_mapping = _load_frame_mapping(session_dir, args.tick_offset)

    ws_bin = labels_dir / "world_states.bin"
    has_world = ws_bin.exists()
    if has_world:
        px, py, pz, blocks = load_world_states(labels_dir)
        tick_count = len(px)
    else:
        px = py = pz = blocks = None
        tick_count = len(ticks)

    cache_dir = _ROOT / "cache"
    geo_cache  = cache_dir / args.version / f"geometry_{args.version}.npz"
    renderer   = None
    if has_world and geo_cache.exists():
        from .renderer import Renderer
        renderer = Renderer(cache_dir, args.version)

    video_path = session_dir / "video.mp4"
    video = VideoReader(video_path) if video_path.exists() else None

    pygame.init()
    pygame.key.set_repeat(120, 40)   # hold-key repeat: 120ms delay, 40ms interval
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("pipeline viewer")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 24)

    tick_idx    = 0
    orbit_frame = 0
    orbit_surf  = None   # cached bottom-panel surface; None = show hint

    # ── top-row render ────────────────────────────────────────────────────────

    def _get_tick_data(tidx):
        """Returns (pose, pbp, mask_or_None, world_u16_or_None, is_localized)."""
        if not has_world or tidx >= tick_count:
            return None, None, None, None, False
        pose = ticks[tidx]["player"]
        pbp  = (int(px[tidx]), int(py[tidx]), int(pz[tidx]))
        npz  = labels_dir / f"tick_{tidx:05d}.npz"
        if npz.exists():
            d = np.load(str(npz), allow_pickle=True)
            if "local_yaw" in d:
                # Localized smelted data: b/m already in cam space (player faces +Z).
                # Render from fixed grid-centre camera with the residual local yaw.
                local_yaw = float(d["local_yaw"])
                cam   = pose.get("camera", pose)
                pitch = float(cam.get("pitch", pose.get("pitch", 0.0)))
                fov   = float(cam.get("fov",   pose.get("fov",   70.0)))
                lpose = {"x": 15.5, "y": 15.62, "z": 15.5,
                         "yaw": local_yaw, "pitch": pitch, "fov": fov,
                         "camera": {"x": 15.5, "y": 15.62, "z": 15.5,
                                    "yaw": local_yaw, "pitch": pitch, "fov": fov}}
                return lpose, (15, 15, 15), d["m"], d["b"].astype(np.uint16), True
            return pose, pbp, d["m"], d["b"].astype(np.uint16), False
        return pose, pbp, None, blocks[tidx] if blocks is not None else None, False

    def render_top(tidx: int) -> None:
        frame_num = frame_mapping.get(tidx)
        vf   = video.get(frame_num) if (video is not None and frame_num is not None) else None
        left = _rgb_to_surface(vf) if vf is not None else _blank_surface((40, 40, 40))
        right = _blank_surface()

        if renderer is not None:
            pose, pbp, mask, world, is_local = _get_tick_data(tidx)
            if world is not None:
                try:
                    rgb   = renderer.render_rgb(world, pose, pbp, mask=mask)
                    right = _rgb_to_surface(rgb)
                except Exception as exc:
                    right.blit(font.render(str(exc)[:70], True, (255, 80, 80)),
                               (10, VID_H // 2 - 12))
        elif renderer is None:
            is_local = False
            right.blit(font.render("No geometry cache — run baker first",
                                   True, (200, 200, 80)), (10, VID_H // 2 - 12))
        else:
            is_local = False

        screen.blit(left,  (0,     0))
        screen.blit(right, (VID_W, 0))
        loc_tag = "  [LOCALIZED]" if is_local else ""
        pygame.display.set_caption(
            f"pipeline viewer  —  tick {tidx}/{tick_count-1}  "
            f"frame {frame_num}  orbit-frame {orbit_frame}/{ORBIT_FRAMES-1}{loc_tag}"
        )

    # ── bottom-row orbit panel ────────────────────────────────────────────────

    def render_bottom() -> None:
        """Draw whatever is in orbit_surf (or a hint) into the bottom half."""
        screen.fill((10, 10, 10), (0, VID_H, WIN_W, VID_H))
        if orbit_surf is not None:
            # centre the 640×360 orbit surface in the 1280-wide bottom row
            screen.blit(orbit_surf, ((WIN_W - VID_W) // 2, VID_H))
        else:
            hint = font.render("Hold O — orbit view", True, (100, 100, 100))
            screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2,
                               VID_H + VID_H // 2 - hint.get_height() // 2))

    def advance_orbit(tidx: int) -> None:
        nonlocal orbit_frame, orbit_surf
        if renderer is None:
            return
        pose, pbp, mask, world, _is_local = _get_tick_data(tidx)
        if world is None:
            return
        cam = pose.get("camera", pose)
        cx, cy, cz = float(cam["x"]), float(cam["y"]), float(cam["z"])
        ocam = _orbit_cam(cx, cy, cz, orbit_frame)
        try:
            rgb = renderer.render_rgb(world, pose, pbp, mask=mask, orbit_cam=ocam)
            orbit_surf = _rgb_to_surface(rgb)
        except Exception:
            pass
        orbit_frame = (orbit_frame + 1) % ORBIT_FRAMES

    # ── initial draw ──────────────────────────────────────────────────────────

    render_top(tick_idx)
    render_bottom()
    pygame.display.flip()

    running = True
    while running:
        dirty = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_LEFT:
                    step = 50 if (event.mod & pygame.KMOD_SHIFT) else 1
                    tick_idx    = max(0, tick_idx - step)
                    orbit_frame = 0;  orbit_surf = None
                    dirty = True
                elif event.key == pygame.K_RIGHT:
                    step = 50 if (event.mod & pygame.KMOD_SHIFT) else 1
                    tick_idx    = min(tick_count - 1, tick_idx + step)
                    orbit_frame = 0;  orbit_surf = None
                    dirty = True
                elif event.key == pygame.K_PAGEDOWN:
                    tick_idx    = min(tick_count - 1, tick_idx + 50)
                    orbit_frame = 0;  orbit_surf = None
                    dirty = True
                elif event.key == pygame.K_PAGEUP:
                    tick_idx    = max(0, tick_idx - 50)
                    orbit_frame = 0;  orbit_surf = None
                    dirty = True

        # O held down — advance orbit by one frame per game tick
        if pygame.key.get_pressed()[pygame.K_o]:
            advance_orbit(tick_idx)
            dirty = True

        if dirty:
            render_top(tick_idx)
            render_bottom()
            pygame.display.flip()

        clock.tick(30)

    pygame.quit()
    if renderer is not None:
        renderer.close()
    if video is not None:
        video.close()


if __name__ == "__main__":
    main()
