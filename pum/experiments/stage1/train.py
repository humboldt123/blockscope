"""
train.py — Stage 1 tokenizer pretraining (Run 2).

Run 2 changes vs Run 1:
  - Focal loss (gamma=2, air_alpha=0.05) replaces weighted cross-entropy
  - 50/50 real + synthetic training data via ConcatDataset
  - Early stopping: patience=5 epochs on val_nonair_acc
  - weight_decay raised to 1e-4 (see config.yaml)
  - Token utilisation per FSQ scale logged to wandb every epoch

Usage (on Brev, in a screen):
    cd /home/vvm33/pum
    CUDA_VISIBLE_DEVICES=6,7 /data/conda_envs/dropper/bin/torchrun \
        --nproc_per_node=2 --master_port=29500 \
        -m experiments.stage1.train \
        [--config experiments/stage1/config.yaml]
"""

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("HF_HOME", "/data/hf_cache_root")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_FSQ_SCALES = [1, 2, 4, 8]


# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_dist():
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()


# ── Probe-tick visualisation ──────────────────────────────────────────────────

def make_slice_image(
    b_gt: np.ndarray,
    b_pred: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Top-down XZ slice at Y=16. Returns (H, 2*W+4, 3) RGB GT | pred."""
    SCALE = 8
    S = 32

    def colorise(grid_2d: np.ndarray) -> np.ndarray:
        img = np.zeros((S * SCALE, S * SCALE, 3), dtype=np.uint8)
        for i in range(S):
            for j in range(S):
                tok = int(grid_2d[i, j])
                if tok == 0:
                    color = (20, 20, 20)
                else:
                    hue = (tok * 137.508) % 360
                    h = hue / 60.0
                    c = int(0.6 * 255)
                    x = int(c * (1 - abs(h % 2 - 1)))
                    if   h < 1: r, g, b = c, x, 0
                    elif h < 2: r, g, b = x, c, 0
                    elif h < 3: r, g, b = 0, c, x
                    elif h < 4: r, g, b = 0, x, c
                    elif h < 5: r, g, b = x, 0, c
                    else:        r, g, b = c, 0, x
                    color = (r, g, b)
                y0, y1 = i * SCALE, (i + 1) * SCALE
                x0, x1 = j * SCALE, (j + 1) * SCALE
                img[y0:y1, x0:x1] = color
        return img

    y_slice = S // 2
    left    = colorise(b_gt[:, y_slice, :])
    right   = colorise(b_pred[:, y_slice, :])
    divider = np.full((S * SCALE, 4, 3), 200, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def select_probe_ticks(val_dataset, n: int, rng: np.random.Generator) -> list[dict]:
    candidates = rng.choice(len(val_dataset), size=min(200, len(val_dataset)), replace=False)
    scored = []
    for idx in candidates:
        grid = val_dataset[int(idx)]
        n_nonair = (grid > 0).sum().item()
        scored.append((n_nonair, int(idx), grid))
    scored.sort(key=lambda x: -x[0])
    probes = [{"idx": idx, "b_gt": grid.numpy()} for _, idx, grid in scored[:n]]
    log.info("Selected %d probe ticks (max non-air: %d)", len(probes), scored[0][0])
    return probes


@torch.no_grad()
def log_probe_renders(probes, model, device, epoch, vocab_size):
    model.eval()
    images = {}
    for i, p in enumerate(probes):
        x = torch.from_numpy(p["b_gt"]).long().unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            b_pred = model.decode(x).squeeze(0).cpu().numpy()
        img = make_slice_image(p["b_gt"], b_pred, vocab_size)
        images[f"probe_{i:02d}"] = wandb.Image(
            img, caption=f"probe {i} | epoch {epoch} | left=GT right=pred"
        )
    wandb.log(images, step=epoch)


# ── Loss ──────────────────────────────────────────────────────────────────────

def build_alpha(num_tokens: int, air_weight: float, device: torch.device) -> torch.Tensor:
    """Per-class alpha for focal loss: air/unknown=air_weight, rest=1.0."""
    alpha = torch.ones(num_tokens, device=device)
    alpha[0] = air_weight
    return alpha


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    logits:  (B, V, *spatial)
    targets: (B, *spatial) long
    alpha:   (V,) per-class weight tensor
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # (B, *spatial)
    pt = torch.exp(-ce)
    loss = (1.0 - pt).pow(gamma) * ce
    if alpha is not None:
        loss = alpha[targets] * loss
    return loss.mean()


# ── Token utilisation ─────────────────────────────────────────────────────────

@torch.no_grad()
def log_token_utilization(
    model, val_ds, device: torch.device, batch_size: int, n_batches: int = 20
) -> dict:
    """Return fraction of FSQ codebook used at each scale over n_batches val batches."""
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    stage0   = model.bottleneck.stages[0]
    max_code = stage0.L ** stage0.d  # 5^6 = 15 625
    seen     = [set() for _ in _FSQ_SCALES]

    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, token_ids = model(batch.to(device))
        for s, tids in enumerate(token_ids):
            seen[s].update(tids.detach().cpu().numpy().flatten().tolist())

    result = {}
    for scale, s_seen in zip(_FSQ_SCALES, seen):
        frac = len(s_seen) / max_code
        result[f"val/codebook_util_{scale}x{scale}x{scale}"] = frac
        log.info(
            "  Codebook util %d^3: %d / %d (%.1f%%)",
            scale, len(s_seen), max_code, 100 * frac,
        )
    return result


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict):
    local_rank, rank, world_size = setup_dist()
    is_main = rank == 0
    device  = torch.device(f"cuda:{local_rank}")

    if is_main:
        log.info("World size: %d GPUs", world_size)

    # ── env setup ─────────────────────────────────────────────────────────────
    os.environ["FURNACE_ROOT"] = cfg["data"]["furnace_root"]
    os.environ["SMELTED_ROOT"] = cfg["data"]["smelted_dir"]
    os.environ["RAW_DATA_ROOT"] = cfg["data"]["raw_data_dir"]

    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── vocab ─────────────────────────────────────────────────────────────────
    vocab_path = Path(__file__).parent.parent.parent / "vocab" / "stage1_vocab.json"
    if is_main and not vocab_path.exists():
        log.info("stage1_vocab.json not found — building now ...")
        from vocab.build_vocab import build
        build(Path(cfg["data"]["furnace_root"]), vocab_path)
    dist.barrier()

    from vocab.build_vocab import load_vocab
    vocab          = load_vocab(vocab_path)
    num_tokens     = vocab["num_tokens"]
    state_to_token = vocab["state_to_token"]
    if is_main:
        log.info("Vocab: %d tokens", num_tokens)

    # ── datasets ──────────────────────────────────────────────────────────────
    from data.tok_dataset import get_session_dirs, TokenizerDataset
    from data.synth_dataset import SynthDataset

    raw_root   = Path(cfg["data"]["raw_data_dir"])
    train_dirs = get_session_dirs(raw_root, "train", cfg["data"]["n_train"])
    test_dirs  = get_session_dirs(raw_root, "test",  cfg["data"]["n_train"])

    if is_main:
        log.info(
            "Train sessions: %d  |  Test sessions: %d",
            len(train_dirs), len(test_dirs),
        )
        log.info("Validating first session loads correctly ...")
        _check_ds = TokenizerDataset(train_dirs[:1], state_to_token, num_tokens)
        sample = _check_ds[0]
        assert sample.shape == (32, 32, 32), f"Unexpected shape: {sample.shape}"
        log.info("  OK — sample shape %s, max token %d", sample.shape, sample.max().item())
    dist.barrier()

    train_ds = TokenizerDataset(train_dirs, state_to_token, num_tokens)
    val_ds   = TokenizerDataset(test_dirs,  state_to_token, num_tokens)

    # 50/50 real + synthetic mix — synth length matches real for equal weighting
    synth_ds          = SynthDataset(num_tokens=num_tokens, length=len(train_ds))
    combined_train_ds = ConcatDataset([train_ds, synth_ds])

    if is_main:
        log.info(
            "Real train: %d  |  Synth: %d  |  Combined: %d",
            len(train_ds), len(synth_ds), len(combined_train_ds),
        )

    train_sampler = DistributedSampler(
        combined_train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        combined_train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["data"]["n_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=cfg["data"]["n_workers"],
        pin_memory=True,
    )

    if is_main:
        log.info(
            "Train batches/rank: %d  |  Val batches/rank: %d",
            len(train_loader), len(val_loader),
        )

    # ── model ─────────────────────────────────────────────────────────────────
    from models.tokenizer import VoxelTokenizer
    model = VoxelTokenizer(
        vocab_size=num_tokens,
        d_embed=cfg["model"]["d_embed"],
        channels=cfg["model"]["channels"],
        fsq_d=cfg["model"]["fsq_d"],
        fsq_L=cfg["model"]["fsq_L"],
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Model params: %s", f"{n_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"]["lr_warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    focal_gamma = float(cfg["training"]["focal_gamma"])
    alpha       = build_alpha(num_tokens, float(cfg["training"]["air_loss_weight"]), device)
    patience    = int(cfg["training"]["early_stopping_patience"])

    # ── wandb + probes (rank 0 only) ──────────────────────────────────────────
    if is_main:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=cfg["wandb"]["run_name"],
            config=cfg,
        )
        wandb.watch(model.module, log="gradients", log_freq=200)
        rng           = np.random.default_rng(42)
        probes        = select_probe_ticks(val_ds, cfg["vis"]["n_probe_ticks"], rng)
        best_val_loss = float("inf")
        best_val_nonair_acc = 0.0
        patience_counter    = 0

    # ── training ──────────────────────────────────────────────────────────────
    stop = False
    for epoch in range(epochs):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = total_correct = total_cells = 0

        pbar = tqdm(
            train_loader,
            desc=f"Ep {epoch+1}/{epochs} [train]",
            disable=not is_main,
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(batch)
                loss = focal_loss(logits, batch, gamma=focal_gamma, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds          = logits.argmax(dim=1)
                total_correct += (preds == batch).sum().item()
                total_cells   += batch.numel()
                total_loss    += loss.item() * batch.size(0)

            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        train_metrics = torch.tensor(
            [total_loss, total_correct, total_cells],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_cells = train_metrics.tolist()

        scheduler.step()
        train_loss = total_loss / len(combined_train_ds)
        train_acc  = total_correct / total_cells

        # ── validation (all ranks, then all-reduce) ────────────────────────────
        model.eval()
        val_loss_sum = val_correct = val_cells = 0
        val_nonair_correct = val_nonair_total = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Ep {epoch+1}/{epochs} [val]",
                disable=not is_main,
                leave=False,
                dynamic_ncols=True,
            ):
                batch = batch.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(batch)
                    loss = focal_loss(logits, batch, gamma=focal_gamma, alpha=alpha)
                val_loss_sum += loss.item() * batch.size(0)
                preds          = logits.argmax(dim=1)
                val_correct   += (preds == batch).sum().item()
                val_cells     += batch.numel()
                mask = batch > 0
                val_nonair_correct += (preds[mask] == batch[mask]).sum().item()
                val_nonair_total   += mask.sum().item()

        val_metrics = torch.tensor(
            [val_loss_sum, val_correct, val_cells, val_nonair_correct, val_nonair_total],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        val_loss_sum, val_correct, val_cells, val_nonair_correct, val_nonair_total = (
            val_metrics.tolist()
        )

        val_loss       = val_loss_sum / len(val_ds)
        val_acc        = val_correct / val_cells
        val_nonair_acc = val_nonair_correct / val_nonair_total if val_nonair_total > 0 else 0.0

        # ── logging + checkpointing + early stopping (rank 0 only) ────────────
        if is_main:
            # Cycle accuracy — every epoch (early stopping means short runs)
            n_cycle = min(8, len(val_ds))
            cycle_sample = torch.stack([val_ds[i] for i in range(n_cycle)]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cycle_acc = model.module.cycle_accuracy(cycle_sample)

            log.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.4f  val_nonair_acc=%.4f  cycle_acc=%.4f",
                epoch + 1, epochs,
                train_loss, val_loss, val_acc, val_nonair_acc, cycle_acc,
            )

            # Token utilisation per FSQ scale
            util_dict = log_token_utilization(
                model.module, val_ds, device,
                batch_size=cfg["training"]["batch_size"],
            )

            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/nonair_acc": val_nonair_acc,
                "val/cycle_acc": cycle_acc,
                "lr": scheduler.get_last_lr()[0],
                **util_dict,
            }
            wandb.log(log_dict, step=epoch + 1)

            if (epoch + 1) % cfg["vis"]["probe_render_every_epochs"] == 0:
                log_probe_renders(probes, model.module, device, epoch + 1, num_tokens)

            # Best-checkpoint tracking (by val_loss for the checkpoint, nonair_acc for stopping)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.module.state_dict(),
                        "val_loss": val_loss,
                        "val_nonair_acc": val_nonair_acc,
                        "num_tokens": num_tokens,
                    },
                    ckpt_dir / "best.pt",
                )
                wandb.run.summary["best_val_loss"]       = val_loss
                wandb.run.summary["best_val_nonair_acc"] = val_nonair_acc
                wandb.run.summary["best_epoch"]          = epoch + 1

            if (epoch + 1) % cfg["checkpointing"]["save_every_epochs"] == 0:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_loss": val_loss,
                        "num_tokens": num_tokens,
                    },
                    ckpt_dir / f"epoch_{epoch+1:04d}.pt",
                )

            # Early stopping on val_nonair_acc
            if val_nonair_acc > best_val_nonair_acc:
                best_val_nonair_acc = val_nonair_acc
                patience_counter    = 0
            else:
                patience_counter += 1
                log.info(
                    "  No improvement in nonair_acc (patience %d/%d)",
                    patience_counter, patience,
                )

            stop = patience_counter >= patience
            if stop:
                log.info(
                    "Early stopping at epoch %d (patience=%d, best nonair_acc=%.4f)",
                    epoch + 1, patience, best_val_nonair_acc,
                )

        dist.barrier()

        # Broadcast early-stop decision from rank 0 to all ranks
        stop_flag = torch.tensor(int(stop), device=device, dtype=torch.int32)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item():
            break

    if is_main:
        wandb.finish()
        log.info("Done. Best val loss: %.4f", best_val_loss)

    cleanup_dist()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "config.yaml",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
