"""
diag_gt_sync.py — compare precomputed cam_classes vs on-the-fly rotation from world_states.
Reports per-tick agreement, and if mismatched, saves a side-by-side render to /tmp/.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/vvm33/blockscope")
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT
from pum.experiments.viblock3.dataset import STEM, rotate_blocks_yaw, build_class_lut

def snap_yaw(yaw_deg):
    snap = float(round(yaw_deg / 90) * 90)
    return snap, yaw_deg - snap
from io_helpers import load_world_states  # type: ignore

VOCAB_PATH = Path("/data/vvm33/pum_checkpoints/blocktype/vocab.json")
class_lut  = build_class_lut(VOCAB_PATH)   # SID → class_index

sessions = sorted(
    d.name for d in SMELTED_ROOT.iterdir()
    if d.is_dir() and (d / "labels" / STEM).exists()
)[:10]  # first 10 sessions

total_ticks  = 0
exact_match  = 0
mismatch_ex  = []

for session in sessions:
    labels_dir = SMELTED_ROOT / session / "labels"
    cc_dir     = labels_dir / "cam_classes"
    if not cc_dir.exists():
        print(f"  {session}: no cam_classes dir, skipping")
        continue

    try:
        _, _, _, blocks = load_world_states(labels_dir)
    except Exception as e:
        print(f"  {session}: load_world_states failed: {e}")
        continue

    d            = np.load(labels_dir / STEM)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]

    n_checked = 0
    for k in range(min(len(tick_indices), 50)):   # check up to 50 ticks per session
        tick_idx = int(tick_indices[k])
        yaw      = float(tick_yaws[k])

        cc_path = cc_dir / f"{tick_idx:06d}.npy"
        if not cc_path.exists() or tick_idx >= len(blocks):
            continue

        precomp = np.load(cc_path).astype(np.int32)   # (32,32,32) class indices

        snap_y, _ = snap_yaw(yaw)
        cam_u16   = rotate_blocks_yaw(blocks[tick_idx], snap_y)
        flat      = cam_u16.reshape(-1)
        onthefly  = class_lut[np.clip(flat, 0, len(class_lut)-1)].reshape(32,32,32).astype(np.int32)

        total_ticks += 1
        n_checked   += 1

        match = np.array_equal(precomp, onthefly)
        if match:
            exact_match += 1
        else:
            diff = (precomp != onthefly)
            n_diff = diff.sum()
            # only save first few mismatches for inspection
            if len(mismatch_ex) < 3:
                mismatch_ex.append({
                    "session": session,
                    "tick": tick_idx,
                    "yaw": yaw,
                    "snap": snap_y,
                    "n_diff": n_diff,
                    "precomp_nonzero": int((precomp > 0).sum()),
                    "otf_nonzero":     int((onthefly > 0).sum()),
                    # check if offset by 1 tick matches
                })

    print(f"  {session}: checked {n_checked} ticks")

print(f"\n=== RESULTS ===")
print(f"Total ticks checked : {total_ticks}")
print(f"Exact matches       : {exact_match} ({100*exact_match/max(total_ticks,1):.1f}%)")
print(f"Mismatches          : {total_ticks - exact_match}")

if mismatch_ex:
    print(f"\nFirst {len(mismatch_ex)} mismatches:")
    for m in mismatch_ex:
        print(f"  {m['session']} tick={m['tick']} yaw={m['yaw']:.1f}° snap={m['snap']:.0f}°"
              f"  diff_cells={m['n_diff']}  precomp_blocks={m['precomp_nonzero']}  otf_blocks={m['otf_nonzero']}")

# Also check: does tick+1 or tick-1 world state match the precomputed cam_classes?
if mismatch_ex and total_ticks > exact_match:
    print("\n--- Offset check: does precomp match on-the-fly at tick±1? ---")
    for m in mismatch_ex[:2]:
        session  = m["session"]
        tick_idx = m["tick"]
        yaw      = m["yaw"]
        snap_y, _ = snap_yaw(yaw)
        labels_dir = SMELTED_ROOT / session / "labels"
        cc_dir     = labels_dir / "cam_classes"
        _, _, _, blocks = load_world_states(labels_dir)

        precomp = np.load(cc_dir / f"{tick_idx:06d}.npy").astype(np.int32)
        for dt in [-1, +1]:
            t2 = tick_idx + dt
            if t2 < 0 or t2 >= len(blocks):
                continue
            cam_u16  = rotate_blocks_yaw(blocks[t2], snap_y)
            onthefly = class_lut[np.clip(cam_u16.reshape(-1), 0, len(class_lut)-1)].reshape(32,32,32).astype(np.int32)
            match    = np.array_equal(precomp, onthefly)
            print(f"  {session} precomp[{tick_idx}] == otf[{t2}]: {match}  (diff={int((precomp!=onthefly).sum())})")
