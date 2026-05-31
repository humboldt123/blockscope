"""
build_vocab.py — Build block-type vocab from the canonical SCATTER_BLOCKS set.

Maps every block-state ID in blockstate_table_1.19.4.json to the scatter-block
class it belongs to (or -1 if not in SCATTER_BLOCKS). No data scanning needed;
the vocab is determined entirely by the voidworld block palette.

Multiple block state IDs for the same block name (e.g. grass_block[snowy=false]
and grass_block[snowy=true]) all collapse to the same class index.

Usage:
    python build_vocab.py [--config config.yaml] [--out /path/to/vocab.json]
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BLOCKSTATE_TABLE = Path(
    "/home/vvm33/blockscope/furnace/pipeline/data/blockstate_table_1.19.4.json"
)
BLOCK_VOCAB_MODULE = "/home/vvm33/blockscope/furnace/pipeline/src/python"


def build_vocab(top_k: int | None = None) -> dict:
    import sys
    sys.path.insert(0, BLOCK_VOCAB_MODULE)
    from block_vocab import SCATTER_BLOCKS

    # Sorted class names for determinism
    classes = sorted(SCATTER_BLOCKS)
    name_to_class = {name: idx for idx, name in enumerate(classes)}
    log.info("SCATTER_BLOCKS: %d unique block names → %d classes", len(classes), len(classes))

    with open(BLOCKSTATE_TABLE) as f:
        table = json.load(f)
    states = table["states"]   # list; index = block-state ID

    sid_to_class: dict[int, int] = {}
    counts = [0] * len(classes)

    for state_id, entry in enumerate(states):
        block_name = entry["name"].removeprefix("minecraft:")
        cls = name_to_class.get(block_name, -1)
        if cls >= 0:
            sid_to_class[state_id] = cls
            counts[cls] += 1

    n_mapped = len(sid_to_class)
    log.info("Mapped %d block-state IDs → %d scatter classes", n_mapped, len(classes))
    for cls, name in enumerate(classes):
        log.info("  class %3d  %-40s  %d state IDs", cls, name, counts[cls])

    return {
        "n_classes": len(classes),
        "classes":   classes,
        "sid_to_class": {str(k): v for k, v in sid_to_class.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_path = args.out or Path(cfg["data"]["vocab_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab()

    with open(out_path, "w") as f:
        json.dump(vocab, f, indent=2)
    log.info("Vocab saved → %s  (%d classes)", out_path, vocab["n_classes"])


if __name__ == "__main__":
    main()
