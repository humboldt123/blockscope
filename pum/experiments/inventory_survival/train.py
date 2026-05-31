#!/usr/bin/env python3
"""EXP1 — Survival Inventory with Slot Queries.

Trains SigLIP-B/16-224 + learned slot queries on survival-only synthetic data.
46 queries cross-attend to all patch tokens, producing exactly 46 slot predictions.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ.setdefault("HF_HOME", "/data/hf_cache_root")

import yaml
import torch
import torch.nn as nn
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data.synth_survival import SurvivalInventoryDataset, survival_collate
from models.survival_inventory import SurvivalInventoryModel
from models.geometry import get_slot_type
from vocab import num_classes, id_to_name
from vis_inventory import (
    log_inventory_grid, log_attention_overlays,
    log_confusion_matrix, log_slot_type_bar_chart,
)


def compute_per_slot_accuracy(preds, targets, slot_type_lists):
    """Compute accuracy broken down by slot type."""
    type_correct = {}
    type_total = {}

    for b in range(preds.shape[0]):
        for q in range(preds.shape[1]):
            slot_type = slot_type_lists[b][q]
            pred = preds[b, q].item()
            gt = targets[b, q].item()
            type_total[slot_type] = type_total.get(slot_type, 0) + 1
            if pred == gt:
                type_correct[slot_type] = type_correct.get(slot_type, 0) + 1

    return {st: type_correct.get(st, 0) / type_total[st] for st in type_total}


# Visualization functions replaced by vis_inventory.py module


def train(config_path: str = None, gpu: int = None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "../../configs/exp1_survival.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device_id = gpu if gpu is not None else cfg.get("gpu", 0)
    device = torch.device(f"cuda:{device_id}")
    torch.manual_seed(cfg.get("seed", 42))

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["name"],
        tags=cfg["wandb"].get("tags", []),
        config=cfg,
    )

    train_ds = SurvivalInventoryDataset(split="train", val_frac=cfg["data"]["val_frac"])
    val_ds = SurvivalInventoryDataset(split="val", val_frac=cfg["data"]["val_frac"])

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=survival_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=survival_collate,
        pin_memory=True,
    )

    n_classes = num_classes()
    model = SurvivalInventoryModel(
        freeze_encoder=cfg["model"]["freeze_encoder"],
        n_classes=n_classes,
        num_queries=cfg["model"]["num_queries"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"].get("dropout", 0.1),
    )

    # Optional resume
    resume_path = cfg["checkpoint"].get("resume_from")
    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["max_epochs"]
    )

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(cfg["training"]["max_epochs"]):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
            pixel_values = batch["pixel_values"].to(device)
            targets = batch["target"].to(device)

            logits, _ = model(pixel_values)  # (B, 46, n_classes)
            loss = criterion(logits.reshape(-1, n_classes), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            optimizer.step()

            train_loss += loss.item() * pixel_values.shape[0]
            preds = logits.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.numel()
            global_step += 1

            if global_step % 100 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": train_correct / max(train_total, 1),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

        train_loss /= len(train_ds)
        train_acc = train_correct / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        all_slot_types = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} val"):
                pixel_values = batch["pixel_values"].to(device)
                targets = batch["target"].to(device)

                logits, _ = model(pixel_values)
                loss = criterion(logits.reshape(-1, n_classes), targets.reshape(-1))
                val_loss += loss.item() * pixel_values.shape[0]

                preds = logits.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.numel()

                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())
                all_slot_types.extend(batch["slot_types"])

        val_loss /= len(val_ds)
        val_acc = val_correct / max(val_total, 1)

        all_preds_cat = torch.cat(all_preds)
        all_labels_cat = torch.cat(all_labels)
        slot_accs = compute_per_slot_accuracy(all_preds_cat, all_labels_cat, all_slot_types)

        metrics = {
            "epoch": epoch + 1,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_acc,
        }
        for st, acc in slot_accs.items():
            metrics[f"val/accuracy_{st}"] = acc
        wandb.log(metrics, step=global_step)

        # Rich visualizations every epoch
        sample_batch = next(iter(val_loader))
        log_inventory_grid(model, sample_batch, device, step=global_step, prefix="val", n_show=3)

        # Attention overlays every epoch
        log_attention_overlays(model, sample_batch, device, step=global_step, prefix="val")

        # Confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            flat_preds = torch.cat(all_preds).reshape(-1)
            flat_labels = torch.cat(all_labels).reshape(-1)
            log_confusion_matrix(flat_preds, flat_labels, step=global_step, prefix="val", k=20)

        # Slot-type bar chart every epoch
        log_slot_type_bar_chart(slot_accs, step=global_step, prefix="val")

        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        for st, acc in sorted(slot_accs.items()):
            print(f"  {st}: {acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "slot_accuracies": slot_accs,
            }, ckpt_dir / "best.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"  Early stopping at epoch {epoch+1} (patience={cfg['training']['patience']})")
                break

    wandb.finish()
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.gpu)
