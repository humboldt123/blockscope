"""
train.py — SigLIP-B/16 NaFlex (frozen) + per-patch visible-block head.

Architecture (living.tex Section 2.3):
  Backbone : SigLIP-B/16 NaFlex — all patch tokens kept, no mean pooling
             Input 640×352 center crop (22×40 = 880 patches @ 16px)
             Output shape: (B, 880, 768) → view to (B, 22, 40, 768)
  Head     : nn.Linear(768, 92) applied per patch
             class 0 = sky/none,  classes 1..91 = SCATTER_BLOCKS

Prediction direction (patch → block):
  For each patch (r,c), the model predicts WHAT block the frustum ray through
  that patch hits.  Ground truth comes from precomputed patch_labels_s20.npz
  (built by precompute_patch_labels.py).  No block array is ever a model input.

Loss  : CrossEntropyLoss per patch, ignore_index=-1 (OOV non-scatter blocks)
Metric: top-1 accuracy on scatter patches only (target class ∈ 1..91)

Run precompute_patch_labels.py on all sessions before training.
Usage:
    python train.py [--config config.yaml] [--device cuda:0]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.patch_label_dataset import (
    PatchLabelDataset, patch_label_collate, build_viblock_lut,
    SMELTED_ROOT, SIGLIP_H, SIGLIP_W,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_session_names(split: str, n_train: int = 73) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


def load_siglip(backbone_name: str, device: torch.device):
    from transformers import AutoModel, AutoProcessor
    log.info("Loading SigLIP backbone: %s", backbone_name)
    model  = AutoModel.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    vision = model.vision_model
    vision.eval().to(device)
    for p in vision.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    log.info("SigLIP loaded (%d params frozen)", sum(p.numel() for p in vision.parameters()))
    return vision, proc


@torch.no_grad()
def siglip_patch_grid(vision: nn.Module, proc, frames_np: list, device: torch.device) -> torch.Tensor:
    """
    List of (360, 640, 3) uint8 numpy frames → (B, 22, 40, 768) patch feature grid.

    Uses AutoProcessor with max_num_patches=880 to get the full 22×40 patch grid.
    NaFlex requires pixel_values (B, N, 768), attention_mask, spatial_shapes.
    """
    from PIL import Image
    pil_images = [Image.fromarray(f) for f in frames_np]
    inputs = proc(images=pil_images, return_tensors="pt", max_num_patches=880)
    pixel_values  = inputs["pixel_values"].to(device)        # (B, 880, 768)
    attention_mask = inputs["pixel_attention_mask"].to(device) # (B, 880)
    spatial_shapes = inputs["spatial_shapes"].to(device)      # (B, 2)

    outputs = vision(
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        spatial_shapes=spatial_shapes,
    )
    patch_h = outputs.last_hidden_state   # (B, 880, 768)
    B = patch_h.shape[0]
    return patch_h.view(B, SIGLIP_H, SIGLIP_W, 768)


class ViBlockHead(nn.Module):
    """Per-patch linear classifier: (*, 768) → (*, n_classes)."""
    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, n_classes)

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        return self.proj(patch_grid)   # (B, 22, 40, n_classes)


@torch.no_grad()
def evaluate(vision: nn.Module, proc, head: ViBlockHead,
             val_loader: DataLoader, device: torch.device,
             n_classes: int) -> dict:
    head.eval()
    correct = n_scatter = 0

    for frames_np, labels in val_loader:
        labels = labels.to(device)   # (B, 22, 40)

        grid   = siglip_patch_grid(vision, proc, frames_np, device)  # (B, 22, 40, 768)
        logits = head(grid)                             # (B, 22, 40, n_classes)

        preds        = logits.argmax(dim=-1)            # (B, 22, 40)
        scatter_mask = labels >= 0                      # exclude sky+OOV (label=-1)

        if scatter_mask.any():
            correct   += (preds[scatter_mask] == labels[scatter_mask]).sum().item()
            n_scatter += scatter_mask.sum().item()

    return {
        "scatter_top1": correct / max(n_scatter, 1),
        "n_scatter":    n_scatter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_path = Path(cfg["data"]["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. "
            "Run pum/experiments/blocktype/build_vocab.py first.")

    lut, n_classes = build_viblock_lut(vocab_path)
    log.info("ViBlock LUT: %d scatter classes (sky+OOV → ignore)", n_classes)

    n_train = cfg["data"]["n_train"]
    train_sessions = get_session_names("train", n_train)
    val_sessions   = get_session_names("val",   n_train)
    log.info("Train: %d sessions  Val: %d sessions", len(train_sessions), len(val_sessions))

    train_ds = PatchLabelDataset(train_sessions, lut, shuffle=True)
    val_ds   = PatchLabelDataset(val_sessions,   lut, shuffle=False)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs,
                              collate_fn=patch_label_collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs,
                              collate_fn=patch_label_collate, num_workers=2)

    vision, proc = load_siglip(cfg["model"]["backbone"], device)
    head   = ViBlockHead(cfg["model"]["feat_dim"], n_classes).to(device)
    log.info("ViBlockHead params: %d  n_classes: %d",
             sum(p.numel() for p in head.parameters()), n_classes)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer  = torch.optim.AdamW(
        head.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    epochs          = cfg["training"]["epochs"]
    steps_per_epoch = max(1, (len(train_sessions) * 30) // bs)  # ~30 labelled ticks/session avg
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["training"]["lr"],
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=cfg["training"]["lr_warmup_epochs"] / epochs,
    )

    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    run = wandb.init(
        project = cfg["wandb"]["project"],
        entity  = cfg["wandb"]["entity"],
        name    = cfg["wandb"]["run_name"],
        config  = cfg,
    )

    best_scatter_top1 = 0.0

    for epoch in range(1, epochs + 1):
        head.train()
        train_loss = 0.0
        n_steps = 0

        for frames_np, labels in train_loader:
            labels = labels.to(device)     # (B, 22, 40)

            with torch.no_grad():
                grid = siglip_patch_grid(vision, proc, frames_np, device)  # (B, 22, 40, 768)

            logits = head(grid)                            # (B, 22, 40, n_classes)

            # Flatten to (B*22*40, n_classes) and (B*22*40,) for CE
            logits_flat = logits.view(-1, n_classes)
            labels_flat = labels.view(-1)

            loss = criterion(logits_flat, labels_flat)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if n_steps < steps_per_epoch:
                scheduler.step()

            train_loss += loss.item()
            n_steps    += 1

        train_loss /= max(n_steps, 1)
        metrics = evaluate(vision, proc, head, val_loader, device, n_classes)

        log.info(
            "Epoch %d/%d  loss=%.4f  scatter_top1=%.4f  n_scatter=%d",
            epoch, epochs, train_loss,
            metrics["scatter_top1"], metrics["n_scatter"],
        )

        run.log({
            "epoch":              epoch,
            "train/loss":         train_loss,
            "val/scatter_top1":   metrics["scatter_top1"],
            "val/n_scatter":      metrics["n_scatter"],
            "lr":                 scheduler.get_last_lr()[0],
        })

        if metrics["scatter_top1"] > best_scatter_top1:
            best_scatter_top1 = metrics["scatter_top1"]
            torch.save({
                "epoch":        epoch,
                "model":        head.state_dict(),
                "scatter_top1": best_scatter_top1,
                "n_classes":    n_classes,
                "vocab":        str(vocab_path),
                "backbone":     cfg["model"]["backbone"],
            }, ckpt_dir / "best.pt")
            log.info("  → new best scatter_top1=%.4f saved", best_scatter_top1)

        if epoch % cfg["checkpointing"]["save_every_epochs"] == 0:
            torch.save({"epoch": epoch, "model": head.state_dict()},
                       ckpt_dir / f"epoch_{epoch:03d}.pt")

    log.info("Done. Best scatter_top1: %.4f", best_scatter_top1)
    run.finish()


if __name__ == "__main__":
    main()
