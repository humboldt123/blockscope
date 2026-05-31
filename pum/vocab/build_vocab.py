"""
build_vocab.py — build stage1_vocab.json from blockstate_table + KNOWN set.

Maps raw Minecraft block-state IDs (uint16 from world_states.bin) to compact
token IDs used by the Stage 1 tokenizer.

Token 0 = unknown / air (any block type not in KNOWN, plus all air variants).
Tokens 1..N = one token per distinct block-state ID whose block type is in KNOWN,
              assigned in blockstate table order.

Usage:
    python build_vocab.py [--furnace-root /path/to/furnace/pipeline] [--out stage1_vocab.json]
"""

import argparse
import json
import sys
from pathlib import Path

FURNACE_ROOT_DEFAULT = Path("/home/vvm33/blockscope/furnace/pipeline")


def build(furnace_root: Path, out_path: Path):
    sys.path.insert(0, str(furnace_root / "src" / "python"))
    from block_vocab import KNOWN, _AIR

    bst_path = furnace_root / "data" / "blockstate_table_1.19.4.json"
    print(f"Loading blockstate table: {bst_path}")
    with open(bst_path) as f:
        table = json.load(f)
    states = table["states"]
    print(f"  {len(states)} total block states in 1.19.4")

    state_to_token = {}
    next_token = 1

    for state_id, entry in enumerate(states):
        raw_name = entry.get("name", "")
        name = raw_name.removeprefix("minecraft:")

        if name in _AIR or name not in KNOWN:
            state_to_token[state_id] = 0
        else:
            state_to_token[state_id] = next_token
            next_token += 1

    num_tokens = next_token  # 0..num_tokens-1
    known_states = sum(1 for v in state_to_token.values() if v > 0)
    print(f"  {known_states} states map to KNOWN tokens (1..{num_tokens-1})")
    print(f"  {len(states) - known_states} states map to token 0 (unknown/air)")
    print(f"  Total vocab size: {num_tokens}")

    vocab = {
        "version": "1.19.4",
        "num_tokens": num_tokens,
        "state_to_token": {str(k): v for k, v in state_to_token.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(vocab, f)
    print(f"Wrote {out_path}")
    return vocab


def load_vocab(vocab_path: Path) -> dict:
    """Load vocab and return state_to_token as int→int dict."""
    with open(vocab_path) as f:
        raw = json.load(f)
    return {
        "num_tokens": raw["num_tokens"],
        "state_to_token": {int(k): v for k, v in raw["state_to_token"].items()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--furnace-root", type=Path, default=FURNACE_ROOT_DEFAULT)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "stage1_vocab.json")
    args = ap.parse_args()
    build(args.furnace_root, args.out)
