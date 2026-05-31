"""
synth/gen_test_samples.py — Generate labelled test samples (images + JSONL).

Usage:
    python synth/gen_test_samples.py [--n N] [--seed S]

Each run clears synth/test_samples/ and regenerates.
JSONL schema (one object per line):
  {
    "filename": "...",          # relative to test_samples/
    "version":  "1.16.4",      # icon set used
    "gui":      "survival",     # layout name, or null for hotbar-only (future)
    "show_hotbar": true,
    "player": {                 # non-empty player slots keyed by protocol slot id (str)
      "36": {"name": "dirt", "id": 3, "count": 12},
      ...
    },
    "container": [              # one entry per container slot (null = empty)
      {"name": "oak_log", "id": 17, "count": 5},
      null,
      ...
    ]
  }
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import synth.generate as G
import vocab as V

OUT_DIR = Path(__file__).parent / "test_samples"


def slot_entry(item_id: int, count: int):
    if item_id == 0:
        return None
    return {"name": V.item_id_to_name(item_id) or str(item_id), "id": item_id, "count": count}


def generate_batch(version: str, layout: str, show_hotbar: bool,
                   n: int, seed: int, records: list):
    rng = random.Random(seed)
    for i in range(n):
        img, lname, p_ids, p_counts, c_ids, c_counts = G.generate_sample(
            layout_name=layout, rng=rng, show_hotbar=show_hotbar,
        )
        tag = "hotbar" if show_hotbar else "nohotbar"
        fname = f"{version}_{layout}_{tag}_{i:03d}.png"
        img.save(OUT_DIR / fname)

        player = {
            str(pid): slot_entry(p_ids[pid], p_counts[pid])
            for pid in range(len(p_ids))
            if p_ids[pid] != 0
        }
        container = [slot_entry(c_ids[j], c_counts[j]) for j in range(len(c_ids))]

        records.append({
            "filename":    fname,
            "version":     version,
            "gui":         lname,
            "show_hotbar": show_hotbar,
            "player":      player,
            "container":   container,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=3,  help="samples per config")
    parser.add_argument("--seed", type=int, default=42, help="base rng seed")
    args = parser.parse_args()

    # Clear and recreate output dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    configs = [
        # (version,   layout,             hotbar)
        ("1.16.4",   "survival",          True),
        ("1.16.4",   "survival",          False),
        ("1.16.4",   "survival_recipe",   True),
        ("1.16.4",   "chest",             True),
        ("1.16.4",   "large_chest",       True),
        ("1.16.4",   "furnace",           True),
        ("1.16.4",   "dispenser",         True),
        ("1.16.4",   "hopper",            True),
        ("1.16.4",   "brewing_stand",     True),
        ("1.16.4",   "anvil",             True),
        ("1.9",      "survival",          True),
        ("1.9",      "survival",          False),
        ("1.9",      "chest",             True),
        ("1.9",      "furnace",           True),
    ]

    records = []
    current_version = None
    for i, (version, layout, hotbar) in enumerate(configs):
        if version != current_version:
            G.set_icons_dir(version)
            current_version = version
        seed = args.seed + i * 1000
        generate_batch(version, layout, hotbar, args.n, seed, records)
        print(f"  {version} {layout} {'hotbar' if hotbar else 'nohotbar'}: {args.n} samples")

    jsonl_path = OUT_DIR / "samples.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{len(records)} samples → {OUT_DIR}/")
    print(f"Metadata → {jsonl_path.name}")


if __name__ == "__main__":
    main()
