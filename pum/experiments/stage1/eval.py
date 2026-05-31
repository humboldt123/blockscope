"""Standalone eval for Stage 1 tokenizer checkpoint.

Computes against the validation (test) split:
  - overall accuracy, non-air accuracy
  - mIoU over all observed non-air classes  (ship criterion: >= 75%)
  - rare-class accuracy                     (ship criterion: >= 60%)
  - cycle_acc on N val samples              (ship criterion: >= 95%)
  - encode+decode latency, batch=1          (ship criterion: < 3ms)

Usage:
    python eval.py [--ckpt /path/to/best.pt] [--config config.yaml]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pum root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

VOCAB_JSON = Path(__file__).parents[2] / "vocab" / "stage1_vocab.json"


def load_vocab(path: Path):
    with open(path) as f:
        raw = json.load(f)
    s2t = {int(k): v for k, v in raw["state_to_token"].items()}
    return raw["num_tokens"], s2t


def build_model(cfg, num_tokens, device):
    from models.tokenizer import VoxelTokenizer
    m = cfg["model"]
    return VoxelTokenizer(
        vocab_size=num_tokens,
        d_embed=m["d_embed"],
        channels=m["channels"],
        fsq_d=m["fsq_d"],
        fsq_L=m["fsq_L"],
    ).to(device)


def load_checkpoint(model, ckpt_path: Path, device):
    state = torch.load(ckpt_path, map_location=device)
    # Support both checkpoint formats: {"model": sd} and {"model_state_dict": sd}
    sd = state.get("model", state.get("model_state_dict", state))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"  val_nonair_acc at save: {state.get('val_nonair_acc', '?')}")
    return state.get("epoch", "?")


@torch.no_grad()
def build_confusion(model, loader, num_tokens, device, max_batches=None):
    """Accumulate (num_tokens, num_tokens) confusion matrix over val set."""
    model.eval()
    conf = np.zeros((num_tokens, num_tokens), dtype=np.int64)
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        preds = logits.argmax(dim=1).cpu().numpy().ravel().astype(np.int32)
        gt = x.cpu().numpy().ravel().astype(np.int32)
        np.add.at(conf, (gt, preds), 1)
        if (i + 1) % 20 == 0:
            print(f"  batch {i+1} done", flush=True)
    return conf


def confusion_metrics(conf, num_tokens, rare_thresh):
    gt_counts = conf.sum(axis=1)
    total = conf.sum()

    overall_acc = np.diag(conf).sum() / total if total > 0 else 0.0

    nonair = np.arange(num_tokens) > 0
    na_total = conf[nonair].sum()
    nonair_acc = np.diag(conf)[nonair].sum() / na_total if na_total > 0 else 0.0

    per_iou, per_acc = {}, {}
    for c in range(num_tokens):
        if gt_counts[c] == 0:
            continue
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = gt_counts[c] - tp
        denom = tp + fp + fn
        per_iou[c] = float(tp / denom) if denom > 0 else 0.0
        per_acc[c] = float(tp / gt_counts[c])

    observed_nonair_iou = [v for c, v in per_iou.items() if c > 0]
    miou = float(np.mean(observed_nonair_iou)) if observed_nonair_iou else 0.0

    nonair_counts = {c: int(gt_counts[c]) for c in range(1, num_tokens) if gt_counts[c] > 0}
    rare_tokens: set[int] = set()
    if nonair_counts:
        threshold = rare_thresh * max(nonair_counts.values())
        rare_tokens = {c for c, cnt in nonair_counts.items() if cnt < threshold}
    rare_accs = [per_acc[c] for c in rare_tokens if c in per_acc]
    rare_acc = float(np.mean(rare_accs)) if rare_accs else 0.0

    return dict(
        overall_acc=overall_acc,
        nonair_acc=nonair_acc,
        miou=miou,
        rare_acc=rare_acc,
        per_iou=per_iou,
        per_acc=per_acc,
        rare_tokens=rare_tokens,
        nonair_counts=nonair_counts,
    )


@torch.no_grad()
def eval_cycle(model, loader, device, n_samples):
    model.eval()
    total = matched = seen = 0
    for batch in loader:
        if seen >= n_samples:
            break
        x = batch.to(device)
        tokens1 = model.encode(x)
        x_hat = model.decode(x)
        tokens2 = model.encode(x_hat)
        for t1, t2 in zip(tokens1, tokens2):
            total += t1.numel()
            matched += (t1 == t2).sum().item()
        seen += x.shape[0]
    return matched / total if total > 0 else 0.0


def measure_latency(model, device, n_warmup=20, n_trials=200):
    model.eval()
    dummy = torch.zeros(1, 32, 32, 32, dtype=torch.long, device=device)
    for _ in range(n_warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.decode(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.decode(dummy)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_trials * 1000  # ms


BST_PATH = Path("/home/vvm33/blockscope/furnace/pipeline/data/blockstate_table_1.19.4.json")


def load_tok_names(vocab_json: Path, bst_path: Path) -> dict[int, str]:
    """Return {token_id: block_name} using blockstate table + vocab."""
    if not bst_path.exists():
        return {}
    with open(bst_path) as f:
        table = json.load(f)
    state_names = [e.get("name", "").replace("minecraft:", "") for e in table["states"]]
    with open(vocab_json) as f:
        raw = json.load(f)
    s2t = {int(k): v for k, v in raw["state_to_token"].items()}
    tok_to_name: dict[int, str] = {}
    for sid, name in enumerate(state_names):
        tok = s2t.get(sid, 0)
        if tok > 0 and tok not in tok_to_name:
            tok_to_name[tok] = name
    return tok_to_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/data/vvm33/pum_checkpoints/stage1/best.pt"))
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--rare-thresh", type=float, default=0.01,
                    help="Tokens with count < this * max_nonair_count are 'rare'")
    ap.add_argument("--cycle-samples", type=int, default=200)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_tokens, s2t = load_vocab(VOCAB_JSON)
    tok_names = load_tok_names(VOCAB_JSON, BST_PATH)
    print(f"Vocab: {num_tokens} tokens")

    from data.tok_dataset import TokenizerDataset, get_session_dirs
    from torch.utils.data import DataLoader

    raw_root = Path(cfg["data"]["raw_data_dir"])
    n_train = cfg["data"]["n_train"]
    val_dirs = get_session_dirs(raw_root, "test", n_train=n_train)
    print(f"Val sessions: {len(val_dirs)}")

    val_ds = TokenizerDataset(val_dirs, s2t, num_tokens)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Val ticks: {len(val_ds)}")

    model = build_model(cfg, num_tokens, device)
    epoch = load_checkpoint(model, args.ckpt, device)
    model.eval()
    print(f"Checkpoint epoch {epoch}: {args.ckpt}\n")

    # --- Latency ---
    print("--- Latency ---")
    lat_ms = measure_latency(model, device)
    lat_pass = lat_ms < 3.0
    print(f"Encode+decode: {lat_ms:.2f} ms  ({'PASS' if lat_pass else 'FAIL'} <3ms)\n")

    # --- Reconstruction ---
    print("--- Building confusion matrix ---")
    conf = build_confusion(model, val_loader, num_tokens, device,
                           max_batches=args.max_val_batches)
    m = confusion_metrics(conf, num_tokens, args.rare_thresh)

    miou_pass  = m["miou"]     >= 0.75
    rare_pass  = m["rare_acc"] >= 0.60

    print(f"\n--- Reconstruction metrics ---")
    print(f"Overall accuracy:  {m['overall_acc']:.4f}")
    print(f"Non-air accuracy:  {m['nonair_acc']:.4f}")
    print(f"mIoU (non-air):    {m['miou']:.4f}  ({'PASS' if miou_pass else 'FAIL'} >=75%)")
    print(f"Rare-class acc:    {m['rare_acc']:.4f}  ({'PASS' if rare_pass else 'FAIL'} >=60%)")
    print(f"  {len(m['rare_tokens'])} rare tokens (count < {args.rare_thresh:.1%} of dominant)")

    print(f"\n--- Per-class breakdown (observed non-air, by frequency) ---")
    print(f"{'tok':>5}  {'count':>10}  {'acc':>6}  {'iou':>6}  {'':4}  name")
    sorted_cls = sorted(
        [(c, m['nonair_counts'].get(c, 0)) for c in m['per_iou'] if c > 0],
        key=lambda x: -x[1],
    )
    for c, cnt in sorted_cls:
        acc = m['per_acc'].get(c, 0.0)
        iou = m['per_iou'].get(c, 0.0)
        flag = "rare" if c in m['rare_tokens'] else ""
        name = tok_names.get(c, "")
        print(f"{c:>5}  {cnt:>10,}  {acc:>6.3f}  {iou:>6.3f}  {flag:<4}  {name}")

    # --- Cycle accuracy ---
    print(f"\n--- Cycle accuracy ({args.cycle_samples} samples) ---")
    cycle_loader = DataLoader(val_ds, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=True)
    cycle_acc = eval_cycle(model, cycle_loader, device, args.cycle_samples)
    cycle_pass = cycle_acc >= 0.95
    print(f"Cycle acc: {cycle_acc:.4f}  ({'PASS' if cycle_pass else 'FAIL'} >=95%)\n")

    # --- Summary ---
    print("=== Ship criteria ===")
    print(f"  mIoU >= 75%:           {'PASS' if miou_pass else 'FAIL'}  ({m['miou']:.1%})")
    print(f"  Rare-class acc >= 60%: {'PASS' if rare_pass else 'FAIL'}  ({m['rare_acc']:.1%})")
    print(f"  Cycle acc >= 95%:      {'PASS' if cycle_pass else 'FAIL'}  ({cycle_acc:.1%})")
    print(f"  Latency < 3ms:         {'PASS' if lat_pass else 'FAIL'}  ({lat_ms:.2f}ms)")
    all_pass = miou_pass and rare_pass and cycle_pass and lat_pass
    print(f"\n  Verdict: {'READY TO SHIP' if all_pass else 'NOT READY'}")


if __name__ == "__main__":
    main()
