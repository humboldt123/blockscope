"""Rich inventory visualizations for wandb.

Creates Minecraft-style inventory grids, attention overlays, and confusion matrices.
"""

import io
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from vocab import id_to_name

# ---------------------------------------------------------------------------
# 1. Minecraft-style inventory grid
# ---------------------------------------------------------------------------

# Slot layout in normalized GUI coordinates (0-1 within the 176x166 panel)
# These match the geometry.py normalized positions for 224x224 display
SLOT_LAYOUT = {
    0:  (0.928, 0.199),   # crafting output
    1:  (0.625, 0.139),   # crafting
    2:  (0.727, 0.139),
    3:  (0.625, 0.247),
    4:  (0.727, 0.247),
    5:  (0.074, 0.084),   # armor
    6:  (0.074, 0.193),
    7:  (0.074, 0.301),
    8:  (0.074, 0.410),
    9:  (0.074, 0.536),   # inventory row 1
    10: (0.176, 0.536),
    11: (0.278, 0.536),
    12: (0.381, 0.536),
    13: (0.483, 0.536),
    14: (0.585, 0.536),
    15: (0.688, 0.536),
    16: (0.790, 0.536),
    17: (0.892, 0.536),
    18: (0.074, 0.645),   # inventory row 2
    19: (0.176, 0.645),
    20: (0.278, 0.645),
    21: (0.381, 0.645),
    22: (0.483, 0.645),
    23: (0.585, 0.645),
    24: (0.688, 0.645),
    25: (0.790, 0.645),
    26: (0.892, 0.645),
    27: (0.074, 0.753),   # inventory row 3
    28: (0.176, 0.753),
    29: (0.278, 0.753),
    30: (0.381, 0.753),
    31: (0.483, 0.753),
    32: (0.585, 0.753),
    33: (0.688, 0.753),
    34: (0.790, 0.753),
    35: (0.892, 0.753),
    36: (0.074, 0.892),   # hotbar
    37: (0.176, 0.892),
    38: (0.278, 0.892),
    39: (0.381, 0.892),
    40: (0.483, 0.892),
    41: (0.585, 0.892),
    42: (0.688, 0.892),
    43: (0.790, 0.892),
    44: (0.892, 0.892),
    45: (0.926, 0.476),   # offhand
}

SLOT_SIZE_NORM = 0.102  # ~18/176


def draw_inventory_grid(pixel_values, preds, targets, slot_types, max_slots=46):
    """Draw a Minecraft-style inventory grid with predictions vs GT.

    Args:
        pixel_values: (3, 224, 224) tensor
        preds: (num_slots,) predicted class IDs
        targets: (num_slots,) ground truth class IDs
        slot_types: list of slot type strings
    Returns:
        PIL Image
    """
    # Ensure everything is on CPU
    pixel_values = pixel_values.cpu()
    preds = preds.cpu()
    targets = targets.cpu()

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Denormalize image
    img = pixel_values.cpu()
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.axis("off")

    for q in range(min(len(preds), max_slots)):
        if q not in SLOT_LAYOUT:
            continue
        nx, ny = SLOT_LAYOUT[q]
        px = nx * 224
        py = ny * 224
        sz = SLOT_SIZE_NORM * 224

        pred_name = id_to_name(preds[q].item())
        gt_name = id_to_name(targets[q].item())
        correct = preds[q] == targets[q]

        # Draw slot box
        color = "#2ecc71" if correct else "#e74c3c"  # green / red
        rect = Rectangle((px - sz/2, py - sz/2), sz, sz,
                         linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        # Label: short name (first 10 chars)
        label = pred_name[:10]
        ax.text(px, py - sz/2 - 4, label, ha="center", va="top",
                fontsize=5, color=color, fontweight="bold")

        # If wrong, show GT below
        if not correct:
            ax.text(px, py + sz/2 + 2, f"GT:{gt_name[:8]}", ha="center", va="bottom",
                    fontsize=4, color="yellow", fontweight="bold")

    # Title with summary
    n_correct = (preds == targets).sum().item()
    acc = n_correct / len(preds)
    ax.set_title(f"Inventory Predictions — {n_correct}/{len(preds)} correct ({acc:.1%})",
                 fontsize=12, fontweight="bold", color="white")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.5)
    fig.savefig(buf, format="png", dpi=120, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# 2. Attention heatmap overlay
# ---------------------------------------------------------------------------

def draw_attention_overlay(pixel_values, attn_weights, slot_idx, slot_type):
    """Overlay attention heatmap for a single slot query on the image.

    Args:
        pixel_values: (3, 224, 224) tensor
        attn_weights: (196,) attention weights for one query
        slot_idx: slot index
        slot_type: slot type string
    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    img = pixel_values.cpu()
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    ax.imshow(img)

    # Reshape attention to 14x14 and upsample
    heatmap = attn_weights.cpu().numpy().reshape(14, 14)
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap_up = F.interpolate(heatmap_t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()

    ax.imshow(heatmap_up, cmap="hot", alpha=0.5)
    ax.set_title(f"Query {slot_idx} ({slot_type}) attention", fontsize=10, color="white")
    ax.axis("off")
    fig.patch.set_facecolor("black")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", dpi=100, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# 3. Confusion matrix (top-k items)
# ---------------------------------------------------------------------------

def draw_confusion_matrix(preds, targets, k=20):
    """Draw confusion matrix for the k most frequent items.

    Args:
        preds: (N,) predicted class IDs
        targets: (N,) ground truth class IDs
        k: number of items to show
    Returns:
        PIL Image
    """
    from collections import Counter

    # Find top-k most frequent GT items
    gt_counts = Counter(targets.tolist())
    top_items = [item for item, _ in gt_counts.most_common(k)]
    item_to_idx = {item: i for i, item in enumerate(top_items)}

    # Build confusion matrix
    cm = np.zeros((k, k), dtype=int)
    for p, t in zip(preds.tolist(), targets.tolist()):
        if t in item_to_idx and p in item_to_idx:
            cm[item_to_idx[t], item_to_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, cmap="Blues")

    names = [id_to_name(item)[:12] for item in top_items]
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)
    ax.set_title(f"Confusion Matrix (top-{k} items)", fontsize=12)

    # Add counts
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=5, color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# 4. Per-slot-type bar chart
# ---------------------------------------------------------------------------

def draw_slot_type_accuracy(slot_accs):
    """Draw bar chart of per-slot-type accuracy.

    Args:
        slot_accs: dict {slot_type: accuracy}
    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    types = list(slot_accs.keys())
    accs = [slot_accs[t] for t in types]
    colors = ["#2ecc71" if a > 0.5 else "#f39c12" if a > 0.2 else "#e74c3c" for a in accs]

    bars = ax.barh(types, accs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title("Per-Slot-Type Accuracy", fontsize=12, fontweight="bold")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{acc:.2%}", va="center", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# 5. Wandb logging helpers
# ---------------------------------------------------------------------------

def log_inventory_grid(model, batch, device, step, prefix="val", n_show=3):
    """Log inventory grid visualizations to wandb."""
    model.eval()
    pixel_values = batch["pixel_values"][:n_show].to(device)
    targets = batch["target"][:n_show].to(device)

    with torch.no_grad():
        logits, _ = model(pixel_values)
    preds = logits.argmax(dim=-1).cpu()

    images = []
    for b in range(min(n_show, pixel_values.shape[0])):
        img = draw_inventory_grid(
            pixel_values[b], preds[b], targets[b], batch["slot_types"][b]
        )
        images.append(wandb.Image(img))

    wandb.log({f"{prefix}/inventory_grid": images}, step=step)


def log_attention_overlays(model, batch, device, step, prefix="val"):
    """Log attention overlay visualizations for representative slots."""
    model.eval()
    pixel_values = batch["pixel_values"][0:1].to(device)

    with torch.no_grad():
        logits, attn_weights = model(pixel_values)
    attn = attn_weights[0].cpu()  # (46, 196)

    show_types = ["crafting_output", "crafting", "armor", "inventory", "hotbar", "offhand"]
    images = []
    for st in show_types:
        idx = None
        for i, t in enumerate(batch["slot_types"][0]):
            if t == st:
                idx = i
                break
        if idx is None:
            continue
        img = draw_attention_overlay(
            batch["pixel_values"][0], attn[idx], idx, st
        )
        images.append(wandb.Image(img, caption=f"{st} (slot {idx})"))

    if images:
        wandb.log({f"{prefix}/attention_overlays": images}, step=step)


def log_confusion_matrix(preds, targets, step, prefix="val", k=20):
    """Log confusion matrix to wandb."""
    img = draw_confusion_matrix(preds, targets, k=k)
    wandb.log({f"{prefix}/confusion_matrix": wandb.Image(img)}, step=step)


def log_slot_type_bar_chart(slot_accs, step, prefix="val"):
    """Log per-slot-type accuracy bar chart."""
    img = draw_slot_type_accuracy(slot_accs)
    wandb.log({f"{prefix}/slot_type_accuracy": wandb.Image(img)}, step=step)
