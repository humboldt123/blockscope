"""One-shot script: scan 10 training sessions and print observed token frequencies."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with open("/home/vvm33/blockscope/furnace/pipeline/data/blockstate_table_1.19.4.json") as f:
    table = json.load(f)
state_names = [e.get("name", "").replace("minecraft:", "") for e in table["states"]]

with open(Path(__file__).parent / "stage1_vocab.json") as f:
    vocab = json.load(f)
s2t = {int(k): v for k, v in vocab["state_to_token"].items()}

sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")
from io_helpers import load_world_states

smelted = sorted(Path("/data/vvm33/SMELTED_DATA").glob("session_*"))[:10]
token_counts: dict[int, int] = {}
for sdir in smelted:
    try:
        labels_dir = sdir / "labels"
        if not (labels_dir / "world_states.bin").exists():
            print("skip", sdir.name, "no world_states.bin")
            continue
        _, _, _, blocks = load_world_states(labels_dir)
        for sid in np.unique(blocks):
            tok = s2t.get(int(sid), 0)
            if tok > 0:
                token_counts[tok] = token_counts.get(tok, 0) + int((blocks == sid).sum())
    except Exception as e:
        print("skip", sdir.name, e)

tok_to_name: dict[int, str] = {}
for sid, name in enumerate(state_names):
    tok = s2t.get(sid, 0)
    if tok > 0 and tok not in tok_to_name:
        tok_to_name[tok] = name

rows = sorted(token_counts.items(), key=lambda x: -x[1])
print(f"Observed {len(rows)} distinct non-air tokens across {len(smelted)} sessions\n")
print(f"{'token':>6}  {'count':>12}  name")
for tok, cnt in rows:
    print(f"{tok:>6}  {cnt:>12,}  {tok_to_name.get(tok, '?')}")
