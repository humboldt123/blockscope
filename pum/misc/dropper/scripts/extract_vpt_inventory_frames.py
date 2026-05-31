#!/usr/bin/env python3
"""Extract VPT inventory frames at native 640×360 resolution.

The old /data/vvm33/vpt_frames/ contains 224×224 square crops that
destroy inventory GUI geometry. This script re-extracts ONLY the
frames where isGuiInventory=true, preserving the original 640×360
aspect ratio so patch→slot mappings from synthetic training transfer.

Output: /data/vvm33/vpt_extracted_inventory_frames/
"""

import sys
import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm

VPT_CONTRACTOR_DIR = Path("/data/vvm33/vpt_contractor")
OUTPUT_DIR = Path("/data/vvm33/vpt_extracted_inventory_frames")


def collect_inventory_frames():
    """Return dict: session_name -> sorted list of frame indices."""
    sessions = {}
    jsonl_files = sorted(VPT_CONTRACTOR_DIR.glob("*.jsonl"))
    print(f"Scanning {len(jsonl_files)} sessions for inventory frames...")

    for jsonl_path in tqdm(jsonl_files, desc="Scanning jsonls"):
        sess = jsonl_path.stem
        indices = []
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tick.get("isGuiInventory") and tick.get("inventory"):
                    # Only extract if inventory is non-empty
                    if len(tick["inventory"]) > 0:
                        frame_idx = tick.get("tick", 0)
                        indices.append(frame_idx)
        if indices:
            sessions[sess] = sorted(set(indices))

    total = sum(len(v) for v in sessions.values())
    print(f"Found {total} inventory frames across {len(sessions)} sessions.")
    return sessions


def extract_frames_for_session(sess_name, frame_indices):
    """Open MP4 once, seek to each frame, save at native resolution."""
    mp4_path = VPT_CONTRACTOR_DIR / f"{sess_name}.mp4"
    if not mp4_path.exists():
        return 0

    out_dir = OUTPUT_DIR / sess_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  WARNING: could not open {mp4_path}")
        return 0

    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # frame is BGR; save as JPEG preserving native resolution
        out_path = out_dir / f"{idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = collect_inventory_frames()

    total_saved = 0
    for sess_name, indices in tqdm(sessions.items(), desc="Extracting"):
        total_saved += extract_frames_for_session(sess_name, indices)

    print(f"\nDone. Saved {total_saved} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
