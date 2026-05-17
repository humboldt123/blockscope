"""
tests/test_session.py — integration test.

Run: python -m pipeline.tests.test_session [session_dir]

What this does:
1. Find a session in BLOCKSCOPE_DATA (default: session_1777850024, else alphabetically first).
2. Run the full pipeline end-to-end on it.
3. For hand-picked ticks (first, mid, last, first-third-person), produce:
   - side-by-side image: actual video frame (left) + voxel render (right)
   - saved to <session>/labels/test_renders/tick_<N>.png
4. Print sanity-check results (non-fatal INFO lines).

Pass/fail = you look at the renders and decide.
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_ROOT    = Path(os.environ.get("DATA_DIR", Path.home() / "blockscope_data"))
PREFER_SESSION = "session_1777850024"

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src" / "python"))


# ── helpers ──────────────────────────────────────────────────────────────────

def find_session() -> Path:
    sessions = sorted(DATA_ROOT.glob("session_*"))
    if not sessions:
        raise FileNotFoundError(f"No session_* dirs in {DATA_ROOT}")
    preferred = DATA_ROOT / PREFER_SESSION
    if preferred.exists():
        return preferred
    return sessions[0]


def load_ticks(session_dir: Path) -> list:
    return [json.loads(l) for l in
            (session_dir / "ticks.jsonl").read_text().splitlines() if l]


def pick_tick_indices(ticks: list) -> list[dict]:
    """Return a list of {idx, label} dicts for the ticks to render."""
    T = len(ticks)
    chosen = [
        {"idx": 0,           "label": "first"},
        {"idx": T // 2,      "label": "mid"},
        {"idx": T - 1,       "label": "last"},
    ]
    # First *stable* third_person_back tick (skip transition frame where camera
    # hasn't moved yet — detectable by camera.x/z == player.x/z)
    for i, t in enumerate(ticks):
        if t["player"].get("cameraPerspective", "") == "third_person_back":
            cam = t["player"]["camera"]
            dx  = abs(cam["x"] - t["player"]["x"])
            dz  = abs(cam["z"] - t["player"]["z"])
            if dx > 0.05 or dz > 0.05:   # camera has actually pulled back
                chosen.append({"idx": i, "label": "third_person"})
                break
    # Deduplicate by idx
    seen = set()
    out  = []
    for c in chosen:
        if c["idx"] not in seen:
            seen.add(c["idx"])
            out.append(c)
    return out


# ── sanity checks (printed as INFO, never raise) ─────────────────────────────

def sanity_shape_dtype(m, b, s, tick_idx):
    shape_ok   = (m.shape == b.shape == s.shape == (32, 32, 32))
    dtype_m_ok = m.dtype == np.uint8
    dtype_b_ok = b.dtype == np.int32
    dtype_s_ok = s.dtype == np.uint8
    log.info("[sanity tick %d] shapes=(32,32,32) %s  dtypes m=%s b=%s s=%s",
             tick_idx, "OK" if shape_ok else "FAIL",
             "OK" if dtype_m_ok else "FAIL",
             "OK" if dtype_b_ok else "FAIL",
             "OK" if dtype_s_ok else "FAIL")


def sanity_tick0_seen_before(s0):
    all_zero = s0.sum() == 0
    log.info("[sanity tick 0] s==0 everywhere: %s  (sum=%d)",
             "OK" if all_zero else "FAIL", int(s0.sum()))


def sanity_consecutive_visibility(m_a, m_b, tick_a, tick_b,
                                  cam_a, cam_b, threshold=0.50):
    """
    If camera barely moved, ≥50% of visible cells at tick_a should still be visible at tick_b.
    """
    delta = math.sqrt(
        (cam_a["x"] - cam_b["x"])**2 +
        (cam_a["y"] - cam_b["y"])**2 +
        (cam_a["z"] - cam_b["z"])**2
    )
    if delta >= 0.1:
        log.info("[sanity ticks %d→%d] camera moved %.3f blocks — skipping overlap check",
                 tick_a, tick_b, delta)
        return
    vis_a = m_a.sum()
    if vis_a == 0:
        log.info("[sanity ticks %d→%d] tick_a has 0 visible cells — skipping", tick_a, tick_b)
        return
    overlap = float((m_a & m_b).sum()) / float(vis_a)
    status  = "OK" if overlap >= threshold else "WARN"
    log.info("[sanity ticks %d→%d] visible overlap=%.2f (threshold=%.2f) [%s]",
             tick_a, tick_b, overlap, threshold, status)


def sanity_f5_occlusion(m, tick_idx, img_w=640, img_h=360):
    """
    In third-person, central screen region should have fewer visible cells
    than surrounding region (player model occludes centre pixels).
    """
    from io_helpers import camera_vectors, player_screen_bbox
    tick_data = None   # checked by caller before passing 3P ticks
    # We can't re-derive screen coords here without tick data — just print counts.
    central_slice  = m[12:20, 12:20, 12:20]
    outer = m.sum() - central_slice.sum()
    log.info("[sanity f5 tick %d] central(12:20)^3 visible=%d  outer visible=%d",
             tick_idx, int(central_slice.sum()), int(outer))


def sanity_visible_distribution(m, b, tick_idx):
    vis = np.where(m)
    total = len(vis[0])
    states = b[vis].astype(np.int32)
    unique, counts = np.unique(states, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    log.info("[sanity tick %d] visible cells=%d  top-5 state_ids(count): %s",
             tick_idx, total,
             ", ".join(f"{sid}({cnt})" for cnt, sid in top5))


# ── main test ────────────────────────────────────────────────────────────────

def run_test():
    session_dir = find_session()
    log.info("Test session: %s", session_dir)

    # ── Stage 1 + 2 via pipeline orchestrator ────────────────────────────────
    import pipeline as pipe_mod
    import importlib
    importlib.reload(pipe_mod)

    t_pipeline = time.time()
    pipe_mod.process_session(session_dir)
    log.info("Pipeline total wall time: %.1f s", time.time() - t_pipeline)

    # ── Load results ──────────────────────────────────────────────────────────
    from io_helpers import load_world_states, load_blockstate_table, load_tick_npz

    labels_dir = session_dir / "labels"
    ticks      = load_ticks(session_dir)
    T          = len(ticks)

    px_arr, py_arr, pz_arr, blocks = load_world_states(labels_dir)
    opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs = load_blockstate_table(PIPELINE_ROOT)

    # ── Global sanity: tick-0 s==0 ────────────────────────────────────────────
    m0, b0, s0, _, _ = load_tick_npz(labels_dir, ticks[0]["tick"])
    sanity_tick0_seen_before(s0)

    # ── Consecutive-tick overlap sanity: find a barely-moving pair with visible cells
    found_consec = False
    for i in range(T - 1):
        cam_a = ticks[i]["player"]["camera"]
        cam_b = ticks[i + 1]["player"]["camera"]
        delta = math.sqrt((cam_a["x"]-cam_b["x"])**2 +
                          (cam_a["y"]-cam_b["y"])**2 +
                          (cam_a["z"]-cam_b["z"])**2)
        if delta < 0.1:
            ma, ba, sa, _, _ = load_tick_npz(labels_dir, ticks[i]["tick"])
            if ma.sum() == 0:
                continue   # no visible cells yet — world not loaded
            mb, bb, sb, _, _ = load_tick_npz(labels_dir, ticks[i + 1]["tick"])
            sanity_consecutive_visibility(
                ma, mb, ticks[i]["tick"], ticks[i+1]["tick"], cam_a, cam_b
            )
            found_consec = True
            break
    if not found_consec:
        log.info("[sanity] No usable barely-moving consecutive tick pair found")

    # ── Render chosen ticks ────────────────────────────────────────────────────
    from visualize import (extract_video_frame, render_perspective_view,
                            make_side_by_side, save_render)
    renders_dir = labels_dir / "test_renders"
    renders_dir.mkdir(exist_ok=True)

    chosen = pick_tick_indices(ticks)
    log.info("Rendering %d ticks: %s", len(chosen),
             [f"{c['label']}(tick {ticks[c['idx']]['tick']})" for c in chosen])

    for c in chosen:
        t_idx  = c["idx"]
        tick   = ticks[t_idx]
        t_num  = tick["tick"]
        label  = c["label"]

        m, b, s, inventory, pose = load_tick_npz(labels_dir, t_num)

        # Per-tick sanity
        sanity_shape_dtype(m, b, s, t_num)
        sanity_visible_distribution(m, b, t_num)
        if "third_person" in label:
            sanity_f5_occlusion(m, t_num)

        # Render
        video_frame = extract_video_frame(session_dir, t_num)
        voxel_frame = render_perspective_view(m, b, pose)
        composite   = make_side_by_side(video_frame, voxel_frame,
                                         label=f"{label} tick={t_num}")
        out_path    = renders_dir / f"tick_{t_num:05d}_{label}.png"
        save_render(out_path, composite)

    log.info("=== Test complete ===")
    log.info("Renders saved to: %s", renders_dir)
    log.info("Open the PNGs and verify: left=video frame, right=voxel visibility.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA_ROOT = Path(sys.argv[1]).parent
        PREFER_SESSION = Path(sys.argv[1]).name
    run_test()
