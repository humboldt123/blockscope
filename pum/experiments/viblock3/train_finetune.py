"""
train_finetune.py — viblock3 DDP + AMP training.

Task: given a video frame (no camera pose, no block coordinates), predict a
32×32×32 camera-relative voxel grid of block types.

Launch:
  torchrun --nproc_per_node=6 train_finetune.py
  or:  EXPERIMENT=viblock3 bash /home/vvm33/train.sh

Requires:
  - viblock3_data.npz per session  (run precompute_data.py --force first)
  - world_states.bin per session   (produced by Furnace)
  - frames/ JPEG cache per session (run viblock2/precompute_frames.py, or falls back to MP4)
"""

import argparse
import logging
import os
import subprocess
import sys
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT
from pum.experiments.viblock3.dataset import ViBlock3Dataset, collate_fn, STEM
from pum.experiments.viblock3.model import ViBlock3

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SIGLIP_H = 22
SIGLIP_W = 40


# ── Session split ─────────────────────────────────────────────────────────────

def get_session_names(split: str, n_train: int) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / STEM).exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


# ── SigLIP ────────────────────────────────────────────────────────────────────

def load_siglip_partial(backbone_name: str, unfreeze_n: int, device: torch.device):
    from transformers import AutoModel, AutoProcessor
    log.info("Loading SigLIP %s, unfreezing last %d encoder blocks", backbone_name, unfreeze_n)
    model  = AutoModel.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    vision = model.vision_model.to(device)
    proc   = AutoProcessor.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")

    for p in vision.parameters():
        p.requires_grad_(False)

    layers   = vision.encoder.layers
    n_layers = len(layers)
    for layer in layers[n_layers - unfreeze_n:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in vision.post_layernorm.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in vision.parameters() if not p.requires_grad)
    log.info("SigLIP: %d trainable, %d frozen", trainable, frozen)
    return vision, proc


def siglip_patch_grid(vision, proc, frames_np, device, training: bool = False):
    """Run SigLIP on a list of raw RGB frames. Returns (B, H_p, W_p, 768)."""
    pil = [Image.fromarray(f) for f in frames_np]
    inp = proc(images=pil, return_tensors="pt", max_num_patches=880)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        out = vision(
            pixel_values   = inp["pixel_values"].to(device),
            attention_mask = inp["pixel_attention_mask"].to(device),
            spatial_shapes = inp["spatial_shapes"].to(device),
        )
    return out.last_hidden_state.view(len(frames_np), SIGLIP_H, SIGLIP_W, 768)


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(vision, proc, model, val_loader, n_classes, device):
    vision.eval(); model.eval()

    total_loss  = 0.0
    n_steps     = 0
    occ_tp = occ_fp = occ_fn = 0
    type_correct = type_total = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for frames_np, cam_blocks in val_loader:
        cam_blocks = cam_blocks.to(device)                      # (B, 32, 32, 32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            grid   = siglip_patch_grid(vision, proc, frames_np, device, training=False)
            logits = model(grid)                                 # (B, 32, 32, 32, C+1)

        B = cam_blocks.shape[0]
        loss = criterion(logits.reshape(B * 32768, n_classes + 1),
                         cam_blocks.reshape(B * 32768))
        total_loss += loss.item()
        n_steps    += 1

        pred = logits.argmax(dim=-1)                            # (B, 32, 32, 32)

        # Occupancy IoU: predicted non-air vs GT non-air
        gt_occ   = cam_blocks > 0                               # (B, 32, 32, 32) bool
        pred_occ = pred > 0
        occ_tp  += int((gt_occ  &  pred_occ).sum())
        occ_fp  += int((~gt_occ &  pred_occ).sum())
        occ_fn  += int((gt_occ  & ~pred_occ).sum())

        # Type accuracy on non-air GT in-vocab cells
        in_vocab = cam_blocks > 0                               # excludes air (-1 already ignore)
        type_correct += int((pred[in_vocab] == cam_blocks[in_vocab]).sum())
        type_total   += int(in_vocab.sum())

    occ_iou  = occ_tp / max(occ_tp + occ_fp + occ_fn, 1)
    type_acc = type_correct / max(type_total, 1)

    return {
        "val_loss": total_loss / max(n_steps, 1),
        "occ_iou":  occ_iou,
        "type_acc": type_acc,
        "type_total": type_total,
    }


# ── WandB slice visualisation (rank 0 only) ───────────────────────────────────

_SLICE_COLORS = None


def _get_palette(n_classes: int):
    global _SLICE_COLORS
    if _SLICE_COLORS is None:
        rng = np.random.RandomState(42)
        _SLICE_COLORS = np.vstack([
            [[40, 40, 40]],                            # 0 = air (dark grey)
            rng.randint(60, 220, size=(n_classes, 3)),  # 1..n_classes
        ]).astype(np.uint8)
    return _SLICE_COLORS


@torch.no_grad()
def render_slice_wandb(vision, proc, model, val_sessions, vocab_path, n_classes, device,
                       n_samples: int = 4):
    """Render top-down (y=16) slices: [frame | GT slice | pred slice] per tick."""
    import wandb
    from pum.experiments.viblock3.dataset import (
        build_class_lut, rotate_blocks_yaw, STEM, FRAMES_DIR
    )
    from pum.data.vis_dataset import RAW_ROOT
    from io_helpers import load_world_states  # type: ignore

    vision.eval(); model.eval()
    palette = _get_palette(n_classes)
    lut     = build_class_lut(vocab_path)

    session    = val_sessions[0]
    labels_dir = SMELTED_ROOT / session / "labels"
    frames_dir = labels_dir / FRAMES_DIR

    if not (labels_dir / STEM).exists() or not (labels_dir / "world_states.bin").exists():
        return []

    _, _, _, blocks = load_world_states(labels_dir)
    d            = np.load(labels_dir / STEM)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]

    K      = len(tick_indices)
    step   = max(1, K // n_samples)
    sample = list(range(0, K, step))[:n_samples]

    wb_images = []
    for ki in sample:
        tick_idx = int(tick_indices[ki])
        yaw      = float(tick_yaws[ki])
        if tick_idx >= len(blocks):
            continue

        frame_rgb = None
        if frames_dir.exists():
            jpg = frames_dir / f"{tick_idx:06d}.jpg"
            if jpg.exists():
                bgr = cv2.imread(str(jpg))
                if bgr is not None:
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb is None:
            video_path = RAW_ROOT / session / "video.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, tick_idx)
                ok, bgr = cap.read()
                cap.release()
                if ok:
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb is None:
            continue

        # Build camera-relative GT class grid
        cam_u16   = rotate_blocks_yaw(blocks[tick_idx], yaw)
        flat      = cam_u16.reshape(-1)
        cam_cls   = lut[np.clip(flat, 0, len(lut) - 1)].reshape(32, 32, 32)  # int64

        # Run model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grid   = siglip_patch_grid(vision, proc, [frame_rgb], device, training=False)
            logits = model(grid)                                  # (1, 32, 32, 32, C+1)
        pred_cls = logits[0].argmax(dim=-1).cpu().numpy()        # (32, 32, 32)

        # Top-down slice at y=16 (player eye level)
        Y = 16
        gt_slice   = cam_cls[:, Y, :]    # (32, 32)  int64, -1/0..n_classes
        pred_slice = pred_cls[:, Y, :]   # (32, 32)  int (0..n_classes)

        def to_rgb(cls_grid):
            clipped = np.clip(cls_grid, 0, n_classes).astype(np.int32)
            return palette[clipped]   # (32, 32, 3)

        gt_img   = to_rgb(gt_slice)
        pred_img = to_rgb(pred_slice)

        # Scale to 128×128 for visibility
        def scale(arr, sz=128):
            return np.array(Image.fromarray(arr).resize((sz, sz), Image.NEAREST))

        frame_small = np.array(Image.fromarray(frame_rgb).resize((256, 144)))
        # Pad frame height to 128
        pad = np.zeros((128 - 128, 256, 3), dtype=np.uint8)
        frame_panel = np.array(Image.fromarray(frame_rgb).resize((256, 128)))

        panel = np.concatenate([frame_panel, scale(gt_img), scale(pred_img)], axis=1)

        # Occupancy stats for caption
        n_gt   = int((gt_slice > 0).sum())
        n_pred = int((pred_slice > 0).sum())
        n_correct = int((pred_slice[gt_slice > 0] == gt_slice[gt_slice > 0]).sum()) if n_gt else 0

        wb_images.append(wandb.Image(
            panel,
            caption=(f"tick={tick_idx} yaw={yaw:.0f}° | "
                     f"GT non-air={n_gt} | pred non-air={n_pred} | "
                     f"type_correct={n_correct}/{n_gt}")
        ))

    return wb_images


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--checkpoint",   type=Path, default=None)
    ap.add_argument("--device",       default=None)
    ap.add_argument("--unfreeze_n",   type=int,   default=2)
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--head_lr",      type=float, default=3e-4)
    ap.add_argument("--backbone_lr",  type=float, default=1e-5)
    ap.add_argument("--run_name",     default="viblock3")
    args = ap.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp     = world_size > 1
    is_main    = (local_rank == 0)

    if is_ddp:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or "cuda:0")
    torch.cuda.set_device(device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_path = Path(cfg["data"]["vocab_path"])
    n_train    = cfg["data"]["n_train"]
    all_train  = get_session_names("train", n_train)
    val_sessions = get_session_names("val", n_train)

    # Shard training sessions by rank
    rank           = dist.get_rank() if is_ddp else 0
    train_sessions = all_train[rank::world_size]

    if is_main:
        log.info("DDP world_size=%d | rank0 train sessions=%d | val sessions=%d",
                 world_size, len(all_train), len(val_sessions))

    bs = cfg["training"]["batch_size"]
    m_cfg = cfg["model"]
    n_classes = m_cfg.get("n_classes", 91)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds     = ViBlock3Dataset(train_sessions, vocab_path, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=bs, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    # Sync step count across ranks (all ranks must call backward the same times)
    n_ticks = sum(
        len(np.load(SMELTED_ROOT / s / "labels" / STEM)["tick_indices"])
        for s in train_sessions
        if (SMELTED_ROOT / s / "labels" / STEM).exists()
    )
    if is_ddp:
        n_t = torch.tensor([n_ticks // bs], device=device, dtype=torch.long)
        dist.all_reduce(n_t, op=dist.ReduceOp.MIN)
        max_steps = int(n_t.item())
    else:
        max_steps = None
    if is_main:
        log.info("Steps per epoch: max_steps=%s", max_steps)

    if is_main:
        val_ds     = ViBlock3Dataset(val_sessions, vocab_path, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate_fn,
                                num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    vision, proc = load_siglip_partial(m_cfg["backbone"], args.unfreeze_n, device)
    model = ViBlock3(
        n_classes = n_classes,
        d_model   = m_cfg["d_model"],
        n_ref     = m_cfg["n_ref"],
        n_heads   = m_cfg["n_heads"],
        n_layers  = m_cfg["n_layers"],
        feat_dim  = m_cfg["feat_dim"],
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        log.info("ViBlock3: %d parameters", n_params)

    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if is_main:
            log.info("Loaded checkpoint: epoch=%d occ_iou=%.4f",
                     ckpt.get("epoch", 0), ckpt.get("occ_iou", 0))
    elif is_main:
        log.info("Starting from scratch")

    if is_ddp:
        vision = DDP(vision, device_ids=[local_rank], find_unused_parameters=False)
        model  = DDP(model,  device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer + AMP ───────────────────────────────────────────────────────
    backbone_params = [p for p in vision.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params,    "lr": args.backbone_lr},
        {"params": model.parameters(), "lr": args.head_lr},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler    = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    ckpt_dir  = Path(cfg["checkpointing"]["dir"])
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB (rank 0 only) ───────────────────────────────────────────────────
    if is_main:
        import wandb
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        wandb.init(
            project = cfg["wandb"]["project"],
            entity  = cfg["wandb"]["entity"],
            name    = args.run_name,
            tags    = ["viblock3", f"unfrz{args.unfreeze_n}", "ddp", "amp", "image-only"],
            config  = {
                "backbone":        m_cfg["backbone"],
                "d_model":         m_cfg["d_model"],
                "n_ref":           m_cfg["n_ref"],
                "n_heads":         m_cfg["n_heads"],
                "n_layers":        m_cfg["n_layers"],
                "n_classes":       n_classes,
                "unfreeze_n":      args.unfreeze_n,
                "head_lr":         args.head_lr,
                "backbone_lr":     args.backbone_lr,
                "epochs":          args.epochs,
                "batch_size":      bs,
                "effective_batch": bs * world_size,
                "world_size":      world_size,
                "weight_decay":    cfg["training"]["weight_decay"],
                "git_sha":         git_sha,
            },
        )

    best_occ_iou = 0.0
    save_every   = cfg["checkpointing"]["save_every_epochs"]

    for epoch in range(1, args.epochs + 1):
        vision.train(); model.train()
        train_loss = 0.0
        n_steps    = 0

        def _infinite(loader):
            while True:
                yield from loader

        src = _infinite(train_loader) if max_steps is not None else train_loader
        for frames_np, cam_blocks in islice(src, max_steps):
            cam_blocks = cam_blocks.to(device)                   # (B, 32, 32, 32)
            B = cam_blocks.shape[0]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                grid   = siglip_patch_grid(vision, proc, frames_np, device, training=True)
                logits = model(grid)                             # (B, 32, 32, 32, n_classes+1)
                loss   = criterion(logits.reshape(B * 32768, n_classes + 1),
                                   cam_blocks.reshape(B * 32768))

            if torch.isnan(loss):
                # Dummy loss to keep DDP gradient sync alive
                loss = logits.sum() * 0.0

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                backbone_params + list(model.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_steps    += 1

        scheduler.step()
        train_loss /= max(n_steps, 1)

        # All-reduce train loss across ranks
        if is_ddp:
            loss_t = torch.tensor([train_loss * n_steps, float(n_steps)], device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            train_loss = (loss_t[0] / loss_t[1]).item()
            dist.barrier()

        if is_main:
            _vision = vision.module if is_ddp else vision
            _model  = model.module  if is_ddp else model

            metrics = evaluate(_vision, proc, _model, val_loader, n_classes, device)

            log.info(
                "Epoch %d/%d  loss=%.4f  val_loss=%.4f  occ_iou=%.4f  type_acc=%.4f  n=%d",
                epoch, args.epochs, train_loss,
                metrics["val_loss"], metrics["occ_iou"], metrics["type_acc"],
                metrics["type_total"],
            )

            wb_log = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   metrics["val_loss"],
                "occ_iou":    metrics["occ_iou"],
                "type_acc":   metrics["type_acc"],
                "lr_head":    optimizer.param_groups[1]["lr"],
                "lr_backbone":optimizer.param_groups[0]["lr"],
            }

            # Slice renders every 5 epochs
            if epoch % save_every == 0 or epoch == 1:
                renders = render_slice_wandb(
                    _vision, proc, _model, val_sessions, vocab_path, n_classes, device)
                if renders:
                    wb_log["renders/slice"] = renders

            import wandb as _wb
            _wb.log(wb_log)

            # Checkpoint
            if epoch % save_every == 0 or metrics["occ_iou"] > best_occ_iou:
                save_path = ckpt_dir / "best.pt"
                torch.save({
                    "model":    _model.state_dict(),
                    "epoch":    epoch,
                    "occ_iou":  metrics["occ_iou"],
                    "type_acc": metrics["type_acc"],
                }, save_path)
                if metrics["occ_iou"] > best_occ_iou:
                    best_occ_iou = metrics["occ_iou"]
                    log.info("New best occ_iou=%.4f → saved %s", best_occ_iou, save_path)

        if is_ddp:
            dist.barrier()

    if is_main:
        import wandb as _wb
        _wb.finish()
        log.info("Training complete. Best occ_iou=%.4f", best_occ_iou)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
