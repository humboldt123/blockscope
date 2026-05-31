"""
synth/gen_dataset.py — Generate a large labelled synthetic dataset.

Usage:
    python synth/gen_dataset.py [--n 100000] [--seed 0] [--out /data/vvm33/inv_synth]

Output layout:
    <out>/images/   — JPEG frames
    <out>/meta.jsonl — one JSON record per line

JSONL schema:
  {
    "filename":    "00000042.jpg",
    "version":     "1.16.4",        // icon set
    "gui":         "survival",      // layout name, null if hotbar-only (future)
    "show_hotbar": true,
    "player": {                     // non-empty player slots, keyed by protocol id (str)
      "36": {"name": "dirt", "id": 3, "count": 12}, ...
    },
    "container": [                  // one entry per container slot (null = empty)
      {"name": "oak_log", "id": 17, "count": 5}, null, ...
    ]
  }
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
import synth.generate as G
import vocab as V

# ── Layout sampling weights ────────────────────────────────────────────────────
# survival 70%, chest 13%, large_chest 10%, furnace 3%, hopper 2%,
# remaining 10 layouts share the last 2% equally.
_FIXED = {
    "survival":    70.0,
    "chest":       13.0,
    "large_chest": 10.0,
    "furnace":      3.0,
    "hopper":       2.0,
}
_REST_SHARE = 2.0

def _build_layout_sampler():
    all_layouts = list(G.LAYOUTS.keys())
    rest = [l for l in all_layouts if l not in _FIXED]
    rest_w = _REST_SHARE / max(len(rest), 1)
    names   = list(_FIXED.keys()) + rest
    weights = [_FIXED[l] for l in _FIXED] + [rest_w] * len(rest)
    return names, weights

_LAYOUT_NAMES, _LAYOUT_WEIGHTS = _build_layout_sampler()

# ── Version sampling (equal weight) ───────────────────────────────────────────
VERSIONS = ["1.16.4", "1.9"]


def slot_entry(item_id: int, count: int):
    if item_id == 0:
        return None
    return {"name": V.item_id_to_name(item_id) or str(item_id), "id": item_id, "count": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,   default=100_000)
    parser.add_argument("--seed", type=int,   default=0)
    parser.add_argument("--out",  type=str,   default="/data/vvm33/inv_synth")
    parser.add_argument("--jpg-quality", type=int, default=93)
    args = parser.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.jsonl"
    # Append mode so reruns can resume; start fresh if --seed matches nothing
    start_idx = 0
    if meta_path.exists():
        with open(meta_path) as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            print(f"Resuming from sample {existing} (found {existing} existing records)")
            start_idx = existing

    rng       = random.Random(args.seed)
    # Advance rng to match already-generated samples
    for _ in range(start_idx * 4):   # 4 random draws per sample (version, layout, hotbar, seed)
        rng.random()

    current_version = None

    with open(meta_path, "a") as jsonl_f:
        for i in tqdm(range(start_idx, args.n), initial=start_idx, total=args.n):
            version     = rng.choice(VERSIONS)
            layout_name = rng.choices(_LAYOUT_NAMES, weights=_LAYOUT_WEIGHTS)[0]
            show_hotbar = rng.random() < 0.5
            sample_seed = rng.randint(0, 2**31)

            if version != current_version:
                G.set_icons_dir(version)
                current_version = version

            img, lname, p_ids, p_counts, c_ids, c_counts = G.generate_sample(
                layout_name=layout_name,
                rng=random.Random(sample_seed),
                show_hotbar=show_hotbar,
            )

            fname = f"{i:08d}.jpg"
            img.save(img_dir / fname, "JPEG", quality=args.jpg_quality)

            player = {
                str(pid): slot_entry(p_ids[pid], p_counts[pid])
                for pid in range(len(p_ids))
                if p_ids[pid] != 0
            }
            container = [slot_entry(c_ids[j], c_counts[j]) for j in range(len(c_ids))]

            jsonl_f.write(json.dumps({
                "filename":    fname,
                "version":     version,
                "gui":         lname,
                "show_hotbar": show_hotbar,
                "player":      player,
                "container":   container,
            }) + "\n")

            # Flush every 1000 to avoid losing progress
            if i % 1000 == 0:
                jsonl_f.flush()

    print(f"\nDone. {args.n} samples in {out_dir}/")
    # Print rough size
    total_bytes = sum(f.stat().st_size for f in img_dir.iterdir())
    print(f"Image dir size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
