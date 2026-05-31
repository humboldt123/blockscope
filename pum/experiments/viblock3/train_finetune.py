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
    if unfreeze_n >= n_layers:
        # Full unfreeze: all SigLIP weights including embeddings
        for p in vision.parameters():
            p.requires_grad_(True)
    else:
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
def evaluate(vision, proc, model, val_loader, n_classes, device, precomputed: bool = False):
    vision.eval(); model.eval()

    total_loss  = 0.0
    n_steps     = 0
    occ_tp = occ_fp = occ_fn = 0
    type_correct = type_total = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for batch_input, cam_blocks in val_loader:
        cam_blocks = cam_blocks.to(device)                      # (B, 32, 32, 32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if precomputed:
                grid = batch_input.to(device)
            else:
                grid = siglip_patch_grid(vision, proc, batch_input, device, training=False)
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


# ── WandB first-person renders (rank 0 only) ──────────────────────────────────

def _build_class_to_sid(vocab_path: Path) -> dict:
    """Map viblock3 class index → representative block state ID."""
    import json
    with open(vocab_path) as f:
        v = json.load(f)
    out = {0: 0}  # class 0 = air (sid 0)
    for sid_str, cls in v["sid_to_class"].items():
        sid = int(sid_str)
        vb3_cls = int(cls) + 1   # viblock3 shifts by 1 to reserve 0 for air
        if vb3_cls not in out or sid < out[vb3_cls]:
            out[vb3_cls] = sid
    return out


def inverse_rotate_blocks_yaw(cam_blocks: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Inverse of rotate_blocks_yaw: camera-relative → world-relative (nearest-neighbor)."""
    yaw_rad = np.radians(yaw_deg)
    cos_y   = float(np.cos(yaw_rad))
    sin_y   = float(np.sin(yaw_rad))

    xi, yi, zi = np.where(cam_blocks > 0)
    if len(xi) == 0:
        return np.zeros((32, 32, 32), dtype=np.uint16)

    sids  = cam_blocks[xi, yi, zi]
    x_cam = xi.astype(np.float32) - 15.0
    z_cam = zi.astype(np.float32) - 15.0

    # Transpose (= inverse) of the yaw rotation matrix
    x_world = (np.round(x_cam * cos_y - z_cam * sin_y).astype(np.int32) + 15)
    z_world = (np.round(x_cam * sin_y + z_cam * cos_y).astype(np.int32) + 15)

    valid = (
        (x_world >= 0) & (x_world < 32) &
        (yi >= 0) & (yi < 32) &
        (z_world >= 0) & (z_world < 32)
    )

    world_blocks = np.zeros((32, 32, 32), dtype=np.uint16)
    world_blocks[x_world[valid], yi[valid], z_world[valid]] = sids[valid]
    return world_blocks


@torch.no_grad()
def render_perspective_wandb(vision, proc, model, val_sessions, vocab_path, n_classes, device,
                             n_samples: int = 4):
    """
    Render [video | GT first-person | predicted first-person] using Furnace visualizer.

    Predictions are in camera-relative space; we inverse-rotate back to world-relative
    before calling render_perspective_view, so the Furnace renderer sees the same
    coordinate system it always expects.
    """
    import json as _json
    import wandb
    from visualize import render_perspective_view, IMG_W, IMG_H  # type: ignore
    from io_helpers import load_world_states                      # type: ignore
    from pum.experiments.viblock3.dataset import rotate_blocks_yaw, STEM, FRAMES_DIR
    from pum.data.vis_dataset import RAW_ROOT

    vision.eval(); model.eval()

    class_to_sid = _build_class_to_sid(vocab_path)
    # Build sid LUT: class → sid for converting pred_cls → block grid
    max_cls = max(class_to_sid.keys()) if class_to_sid else 0
    sid_lut  = np.zeros(max_cls + 1, dtype=np.uint16)
    for cls, sid in class_to_sid.items():
        if cls <= max_cls:
            sid_lut[cls] = sid

    session    = val_sessions[0]
    labels_dir = SMELTED_ROOT / session / "labels"
    frames_dir = labels_dir / FRAMES_DIR
    raw_dir    = RAW_ROOT / session

    if not (labels_dir / STEM).exists() or not (labels_dir / "world_states.bin").exists():
        return []

    _, _, _, blocks = load_world_states(labels_dir)

    d            = np.load(labels_dir / STEM)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]

    # Load visibility masks and ticks for player pose
    vis_path = labels_dir / "visibility_s20.npz"
    vis_lookup = {}
    if vis_path.exists():
        vd = np.load(vis_path)
        vis_lookup = {int(vd["tick_indices"][i]): vd["visibility"][i]
                      for i in range(len(vd["tick_indices"]))}

    ticks_lookup = {}
    ticks_path = raw_dir / "ticks.jsonl"
    if ticks_path.exists():
        for line in ticks_path.read_text().splitlines():
            if line:
                t = _json.loads(line)
                ticks_lookup[int(t["tick"])] = t

    K      = len(tick_indices)
    step   = max(1, K // n_samples)
    sample = list(range(0, K, step))[:n_samples]

    wb_images = []
    for ki in sample:
        tick_idx = int(tick_indices[ki])
        yaw      = float(tick_yaws[ki])
        if tick_idx >= len(blocks) or tick_idx not in ticks_lookup:
            continue

        # Load frame
        frame_rgb = None
        if frames_dir.exists():
            jpg = frames_dir / f"{tick_idx:06d}.jpg"
            if jpg.exists():
                bgr = cv2.imread(str(jpg))
                if bgr is not None:
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb is None:
            video_path = raw_dir / "video.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, tick_idx)
                ok, bgr = cap.read()
                cap.release()
                if ok:
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb is None:
            continue

        # Run model → camera-relative class predictions
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grid   = siglip_patch_grid(vision, proc, [frame_rgb], device, training=False)
            logits = model(grid)
        pred_cam_cls = logits[0].argmax(dim=-1).cpu().numpy()   # (32,32,32) class indices

        # Convert class indices → SIDs
        pred_cam_sid = sid_lut[np.clip(pred_cam_cls, 0, len(sid_lut) - 1)]  # (32,32,32) uint16

        # Inverse-rotate predictions back to world-relative
        b_pred = inverse_rotate_blocks_yaw(pred_cam_sid, yaw).astype(np.int32)

        # GT world-relative blocks
        b_gt = blocks[tick_idx].astype(np.int32)

        # Visibility mask (use closest available tick if exact not found)
        if vis_lookup:
            closest = min(vis_lookup, key=lambda t: abs(t - tick_idx))
            m_vis   = vis_lookup[closest].astype(bool)
        else:
            m_vis = (b_gt > 0)

        # Player pose for renderer
        p    = ticks_lookup[tick_idx].get("player", ticks_lookup[tick_idx])
        pose = {
            "x": p.get("x", 0), "y": p.get("y", 0), "z": p.get("z", 0),
            "yaw": p.get("yaw", 0), "pitch": p.get("pitch", 0), "fov": p.get("fov", 70),
        }

        m_vis_pred  = (b_pred > 0).astype(bool)
        render_gt   = render_perspective_view(m_vis,      b_gt,   pose, fullbright=True)
        render_pred = render_perspective_view(m_vis_pred, b_pred, pose, fullbright=True)

        frame_panel = np.array(Image.fromarray(frame_rgb).resize((IMG_W, IMG_H)))
        panel = np.concatenate([frame_panel, render_gt, render_pred], axis=1)

        n_occ_gt   = int((b_gt   > 0).sum())
        n_occ_pred = int((b_pred > 0).sum())

        wb_images.append(wandb.Image(
            panel,
            caption=(f"tick={tick_idx}  yaw={yaw:.0f}°  "
                     f"GT blocks={n_occ_gt}  pred blocks={n_occ_pred}")
        ))

    return wb_images


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",         type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--checkpoint",     type=Path, default=None)
    ap.add_argument("--device",         default=None)
    ap.add_argument("--unfreeze_n",     type=int,   default=2)
    ap.add_argument("--epochs",         type=int,   default=50)
    ap.add_argument("--head_lr",        type=float, default=3e-4)
    ap.add_argument("--backbone_lr",    type=float, default=1e-5)
    ap.add_argument("--run_name",       default="viblock3")
    ap.add_argument("--no_precomputed", action="store_true",
                    help="Force online SigLIP mode even if precomputed feats exist")
    ap.add_argument("--eval_every",     type=int, default=5,
                    help="Run eval every N epochs (default 5)")
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

    # Auto-detect precomputed SigLIP features (siglip_feats/ dirs)
    from pum.experiments.viblock3.dataset import FEATS_DIR as _FEATS_DIR
    n_precomputed = sum(
        1 for s in (all_train + val_sessions)
        if (SMELTED_ROOT / s / "labels" / _FEATS_DIR).exists()
    )
    precomputed = (not args.no_precomputed) and (n_precomputed >= len(all_train + val_sessions) // 2)
    if is_main:
        log.info("Precomputed SigLIP features: %d/%d sessions → mode=%s",
                 n_precomputed, len(all_train + val_sessions),
                 "PRECOMPUTED" if precomputed else "ONLINE")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds     = ViBlock3Dataset(train_sessions, vocab_path, shuffle=True, precomputed=precomputed)
    train_loader = DataLoader(train_ds, batch_size=bs, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)

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
        val_ds     = ViBlock3Dataset(val_sessions, vocab_path, shuffle=False, precomputed=precomputed)
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

    model = torch.compile(model)

    if is_ddp:
        if not precomputed:
            vision = DDP(vision, device_ids=[local_rank], find_unused_parameters=True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer + AMP ───────────────────────────────────────────────────────
    # In precomputed mode SigLIP is bypassed in the forward pass — no gradients flow
    # through it, so exclude it from DDP all-reduce and optimizer.
    backbone_params = ([] if precomputed
                       else [p for p in vision.parameters() if p.requires_grad])
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

    # ── Epoch 0: baseline render before any training ──────────────────────────
    if is_main:
        _vision0 = vision.module if (is_ddp and not precomputed) else vision
        _model0  = model.module  if is_ddp else model
        _vision0.eval(); _model0.eval()
        renders0 = render_perspective_wandb(
            _vision0, proc, _model0, val_sessions, vocab_path, n_classes, device)
        import wandb as _wb
        if renders0:
            _wb.log({"epoch": 0, "renders/perspective": renders0})
        log.info("Epoch 0 renders uploaded (pre-training baseline)")

    if is_ddp:
        dist.barrier()

    for epoch in range(1, args.epochs + 1):
        vision.train(); model.train()
        train_loss = 0.0
        n_steps    = 0

        def _infinite(loader):
            while True:
                yield from loader

        src = _infinite(train_loader) if max_steps is not None else train_loader
        for batch_input, cam_blocks in islice(src, max_steps):
            cam_blocks = cam_blocks.to(device)                   # (B, 32, 32, 32)
            B = cam_blocks.shape[0]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                if precomputed:
                    # batch_input is (B, 22, 40, 768) float16 tensor — no SigLIP needed
                    grid = batch_input.to(device)
                else:
                    # batch_input is list of raw RGB frames
                    grid = siglip_patch_grid(vision, proc, batch_input, device, training=True)
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
            _vision = vision.module if (is_ddp and not precomputed) else vision
            _model  = model.module  if is_ddp else model

            do_eval = (epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs)

            wb_log = {
                "epoch":       epoch,
                "train_loss":  train_loss,
                "lr_head":     optimizer.param_groups[1]["lr"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
            }

            if do_eval:
                metrics = evaluate(_vision, proc, _model, val_loader, n_classes, device, precomputed)
                log.info(
                    "Epoch %d/%d  loss=%.4f  val_loss=%.4f  occ_iou=%.4f  type_acc=%.4f  n=%d",
                    epoch, args.epochs, train_loss,
                    metrics["val_loss"], metrics["occ_iou"], metrics["type_acc"],
                    metrics["type_total"],
                )
                wb_log.update({
                    "val_loss": metrics["val_loss"],
                    "occ_iou":  metrics["occ_iou"],
                    "type_acc": metrics["type_acc"],
                })

                renders = render_perspective_wandb(
                    _vision, proc, _model, val_sessions, vocab_path, n_classes, device)
                if renders:
                    wb_log["renders/perspective"] = renders

                if metrics["occ_iou"] > best_occ_iou:
                    best_occ_iou = metrics["occ_iou"]
                    save_path = ckpt_dir / "best.pt"
                    # Unwrap torch.compile's OptimizedModule before saving
                    raw_model = getattr(_model, '_orig_mod', _model)
                    torch.save({
                        "model":    raw_model.state_dict(),
                        "epoch":    epoch,
                        "occ_iou":  metrics["occ_iou"],
                        "type_acc": metrics["type_acc"],
                    }, save_path)
                    log.info("New best occ_iou=%.4f → saved %s", best_occ_iou, save_path)
            else:
                log.info("Epoch %d/%d  loss=%.4f  (eval every 5 epochs)",
                         epoch, args.epochs, train_loss)

            import wandb as _wb
            _wb.log(wb_log)

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
