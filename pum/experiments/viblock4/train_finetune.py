"""
train_finetune.py — viblock4 DDP + AMP training.

Architecture change vs viblock3:
  viblock3: SigLIP → patch memory → separate cross-attn decoder → 32^3 logits
  viblock4: SigLIP with 8^3 voxel tokens injected at layer `inject_at` →
            joint self-attn for remaining blocks → 3-D conv upsample → 32^3

The vision model lives inside ViBlock4, wrapped together as one DDP module.
Always runs online SigLIP (no precomputed features; voxel tokens are inside
the encoder so the backbone must be in the forward pass).

Launch:
  EXPERIMENT=viblock4 UNFREEZE_N=2 EPOCHS=100 HEAD_LR=3e-4 BACKBONE_LR=1e-5 \\
    INJECT_AT=10 bash /home/vvm33/blockscope/train.sh
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
# New furnace pipeline (renderer) lives under furnace/pipeline/
_FURNACE_DIR = Path(__file__).resolve().parents[3] / "furnace"
if str(_FURNACE_DIR) not in sys.path:
    sys.path.insert(0, str(_FURNACE_DIR))

from pum.data.vis_dataset import SMELTED_ROOT
from pum.experiments.viblock3.dataset import ViBlock3Dataset, collate_fn, STEM
from pum.experiments.viblock4.model import ViBlock4

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Session split ─────────────────────────────────────────────────────────────

def get_session_names(split: str, n_train: int) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / STEM).exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


# ── SigLIP loader (unfreezes last unfreeze_n encoder blocks) ──────────────────

def load_siglip(backbone_name: str, unfreeze_n: int, device: torch.device):
    from transformers import AutoModel, AutoProcessor
    log.info("Loading SigLIP %s, unfreezing last %d blocks", backbone_name, unfreeze_n)
    siglip = AutoModel.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    vision = siglip.vision_model.to(device)
    proc   = AutoProcessor.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")

    for p in vision.parameters():
        p.requires_grad_(False)

    layers   = vision.encoder.layers
    n_layers = len(layers)
    if unfreeze_n >= n_layers:
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


def proc_frames(proc, frames_np: list, device: torch.device) -> dict:
    """Run processor on a list of RGB ndarrays; move tensors to device."""
    pil = [Image.fromarray(f) for f in frames_np]
    inp = proc(images=pil, return_tensors="pt", max_num_patches=880)
    return {k: v.to(device) for k, v in inp.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, proc, val_loader, n_classes, device):
    model.eval()
    total_loss   = 0.0
    n_steps      = 0
    occ_tp = occ_fp = occ_fn = 0
    type_correct = type_total = 0
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WARNING: CLASS-WEIGHTED LOSS — air (0) and stone (83) are downweighted.
    # Air dominates ~90% of voxels; stone is ubiquitous underground filler.
    # Without this, the model collapses to predicting air/stone everywhere.
    # To ablate: set all weights to 1.0.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    _w = torch.ones(n_classes + 1, device=device)
    _w[0]  = 0.1   # air
    _w[83] = 0.3   # stone (vocab idx 82, model class 83)
    criterion    = nn.CrossEntropyLoss(weight=_w, ignore_index=-1)

    for frames, cam_blocks in val_loader:
        cam_blocks = cam_blocks.to(device)   # (B, 32, 32, 32)
        B = cam_blocks.shape[0]
        inp = proc_frames(proc, frames, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                pixel_values         = inp["pixel_values"],
                pixel_attention_mask = inp["pixel_attention_mask"],
                spatial_shapes       = inp["spatial_shapes"],
            )  # (B, 32, 32, 32, n_classes+1)
            loss = criterion(logits.reshape(B * 32768, n_classes + 1),
                             cam_blocks.reshape(B * 32768))

        total_loss += loss.item()
        n_steps    += 1

        pred = logits.argmax(dim=-1)   # (B, 32, 32, 32)
        gt_occ   = cam_blocks > 0
        pred_occ = pred > 0
        occ_tp  += int((gt_occ  &  pred_occ).sum())
        occ_fp  += int((~gt_occ &  pred_occ).sum())
        occ_fn  += int((gt_occ  & ~pred_occ).sum())

        in_vocab = cam_blocks > 0
        type_correct += int((pred[in_vocab] == cam_blocks[in_vocab]).sum())
        type_total   += int(in_vocab.sum())

    return {
        "val_loss":   total_loss / max(n_steps, 1),
        "occ_iou":    occ_tp / max(occ_tp + occ_fp + occ_fn, 1),
        "type_acc":   type_correct / max(type_total, 1),
        "type_total": type_total,
    }


# ── WandB renders (rank 0 only) ───────────────────────────────────────────────

def _build_class_to_sid(vocab_path: Path) -> dict:
    import json
    with open(vocab_path) as f:
        v = json.load(f)
    out = {0: 0}
    for sid_str, cls in v["sid_to_class"].items():
        sid = int(sid_str)
        vb_cls = int(cls) + 1
        if vb_cls not in out or sid < out[vb_cls]:
            out[vb_cls] = sid
    return out


_gpu_renderer = None   # lazy-initialized singleton

def _get_renderer():
    global _gpu_renderer
    if _gpu_renderer is None:
        for cache_candidate in [
            Path("/home/vvm33/blockscope/furnace/pipeline/cache"),
            Path(__file__).resolve().parents[3] / "furnace" / "pipeline" / "cache",
        ]:
            if cache_candidate.exists():
                from pipeline.renderer import Renderer
                _gpu_renderer = Renderer(cache_candidate, "1.19.4")
                log.info("GPU renderer initialised from %s", cache_candidate)
                break
        if _gpu_renderer is None:
            log.warning("GPU renderer: no baker cache found — renders will be skipped")
    return _gpu_renderer


def render_wandb_samples(model, proc, train_sessions: list, val_sessions: list,
                         cls_to_sid: dict, n_each: int, device: torch.device, epoch: int):
    """Render [video | GT reconstruction | model prediction] strips and log to WandB."""
    import wandb
    IMG_W, IMG_H = 640, 360

    max_cls = max(cls_to_sid.keys(), default=0)
    sid_lut = np.zeros(max_cls + 1, dtype=np.int32)
    for cls, sid in cls_to_sid.items():
        if cls <= max_cls:
            sid_lut[cls] = sid

    _m = getattr(model, "module", model)
    _m.eval()

    def _collect(sessions, n, split):
        rng    = np.random.default_rng(epoch * 10007 + (0 if split == "train" else 1))
        order  = rng.permutation(len(sessions)).tolist()
        images = []
        for idx in order:
            if len(images) >= n:
                break
            session    = sessions[idx]
            labels_dir = SMELTED_ROOT / session / "labels"
            cc_dir     = labels_dir / "cam_classes"
            frames_dir = labels_dir / "frames"
            data_path  = labels_dir / STEM
            if not (data_path.exists() and cc_dir.exists() and frames_dir.exists()):
                continue

            d            = np.load(data_path)
            tick_indices = d["tick_indices"]
            tick_yaws    = d["tick_yaws"]
            try:
                tick_pitches = d["tick_pitches"]
            except KeyError:
                tick_pitches = np.zeros_like(tick_yaws)

            candidates = [k for k in range(len(tick_indices))
                          if (frames_dir / f"{int(tick_indices[k]):06d}.jpg").exists()
                          and (cc_dir / f"{int(tick_indices[k]):06d}.npy").exists()]
            if not candidates:
                continue

            rng.shuffle(candidates)
            k = None
            for ck in candidates[:20]:
                t = int(tick_indices[ck])
                if np.load(cc_dir / f"{t:06d}.npy").astype(np.int32).clip(0, None).sum() > 0:
                    k = ck
                    break
            if k is None:
                continue

            tick_idx = int(tick_indices[k])
            yaw      = float(tick_yaws[k])
            pitch    = float(tick_pitches[k])
            snap_y   = round(yaw / 90) * 90
            local_y  = yaw - snap_y

            gt_cam_cls = np.load(cc_dir / f"{tick_idx:06d}.npy").astype(np.int32)
            gt_sid     = sid_lut[np.clip(gt_cam_cls, 0, max_cls)]
            gt_sid[gt_cam_cls < 0] = 0

            frame_rgb = np.array(Image.open(frames_dir / f"{tick_idx:06d}.jpg").convert("RGB"))
            pose = {"x": 15.5, "y": 15.62, "z": 15.5, "yaw": local_y, "pitch": pitch, "fov": 70.0}

            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    inp    = proc_frames(proc, [frame_rgb], device)
                    logits = _m(
                        pixel_values         = inp["pixel_values"],
                        pixel_attention_mask = inp["pixel_attention_mask"],
                        spatial_shapes       = inp["spatial_shapes"],
                    )
                pred_cls = logits[0].argmax(dim=-1).cpu().numpy().astype(np.int32)
                pred_sid = sid_lut[np.clip(pred_cls, 0, max_cls)]
                pred_sid[pred_cls <= 0] = 0

                renderer = _get_renderer()
                if renderer is None:
                    continue
                pose_r = {"x": 15.5, "y": 15.62, "z": 15.5,
                          "yaw": local_y, "pitch": pitch, "fov": 70.0}
                r_gt   = renderer.render_rgb(gt_sid.astype(np.uint16),   pose_r, (15, 15, 15), mask=None)
                r_pred = renderer.render_rgb(pred_sid.astype(np.uint16), pose_r, (15, 15, 15), mask=None)
                f_vid  = np.array(Image.fromarray(frame_rgb).resize((IMG_W, IMG_H)))
                strip  = np.concatenate([f_vid, r_gt, r_pred], axis=1)

                caption = (f"[{split}] {session} tick={tick_idx} "
                           f"yaw={yaw:.1f}° snap={snap_y:.0f}° local={local_y:+.1f}° pitch={pitch:.1f}° | "
                           f"video | GT | pred")
                images.append((wandb.Image(Image.fromarray(strip.astype(np.uint8)), caption=caption), caption))
            except Exception as e:
                log.warning("Render failed for %s tick=%d: %s", session, tick_idx, e)

        return images

    train_imgs = _collect(train_sessions, n_each, "train")
    val_imgs   = _collect(val_sessions,   n_each, "val")
    _m.train()

    all_items = train_imgs + val_imgs
    if all_items:
        import wandb as _wb
        log_dict = {}
        for i, (img, _) in enumerate(train_imgs):
            log_dict[f"renders/train_{i}"] = img
        for i, (img, _) in enumerate(val_imgs):
            log_dict[f"renders/val_{i}"] = img
        _wb.log(log_dict, step=epoch)
        log.info("Logged %d render samples to WandB (epoch %d)", len(all_items), epoch)


def inverse_rotate_blocks_yaw(cam_blocks: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw_rad = np.radians(yaw_deg)
    cos_y = float(np.cos(yaw_rad))
    sin_y = float(np.sin(yaw_rad))
    xi, yi, zi = np.where(cam_blocks > 0)
    if len(xi) == 0:
        return np.zeros((32, 32, 32), dtype=np.uint16)
    sids  = cam_blocks[xi, yi, zi]
    x_cam = xi.astype(np.float32) - 15.0
    z_cam = zi.astype(np.float32) - 15.0
    x_world = (np.round(x_cam * cos_y - z_cam * sin_y).astype(np.int32) + 15)
    z_world = (np.round(x_cam * sin_y + z_cam * cos_y).astype(np.int32) + 15)
    valid = ((x_world >= 0) & (x_world < 32) & (yi >= 0) & (yi < 32) &
             (z_world >= 0) & (z_world < 32))
    world_blocks = np.zeros((32, 32, 32), dtype=np.uint16)
    world_blocks[x_world[valid], yi[valid], z_world[valid]] = sids[valid]
    return world_blocks



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      type=Path,  default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--checkpoint",  type=Path,  default=None)
    ap.add_argument("--device",      default=None)
    ap.add_argument("--unfreeze_n",  type=int,   default=2)
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--head_lr",     type=float, default=3e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--inject_at",   type=int,   default=10)
    ap.add_argument("--eval_every",  type=int,   default=5)
    ap.add_argument("--run_name",    default="viblock4")
    args = ap.parse_args()

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

    vocab_path   = Path(cfg["data"]["vocab_path"])
    n_train      = cfg["data"]["n_train"]
    all_train    = get_session_names("train", n_train)
    val_sessions = get_session_names("val", n_train)
    cls_to_sid   = _build_class_to_sid(vocab_path)

    rank           = dist.get_rank() if is_ddp else 0
    train_sessions = all_train[rank::world_size]

    if is_main:
        log.info("DDP world_size=%d | train sessions=%d | val sessions=%d",
                 world_size, len(all_train), len(val_sessions))

    bs    = cfg["training"]["batch_size"]
    m_cfg = cfg["model"]
    with open(vocab_path) as _vf:
        n_classes = json.load(_vf)["n_classes"]   # read from vocab (currently 93)

    # Dataset always online (no precomputed mode — voxel tokens live inside SigLIP)
    train_ds     = ViBlock3Dataset(train_sessions, vocab_path, shuffle=True, precomputed=False)
    train_loader = DataLoader(train_ds, batch_size=bs, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)

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
        val_ds     = ViBlock3Dataset(val_sessions, vocab_path, shuffle=False, precomputed=False)
        val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate_fn,
                                num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    vision, proc = load_siglip(m_cfg["backbone"], args.unfreeze_n, device)
    model = ViBlock4(
        vision    = vision,
        n_classes = n_classes,
        inject_at = args.inject_at,
        vox_res   = m_cfg.get("vox_res", 8),
        feat_dim  = m_cfg["feat_dim"],
        mid_dim   = m_cfg.get("mid_dim", 256),
    ).to(device)

    if is_main:
        head_params = sum(p.numel() for p in model.vox_tokens.requires_grad_(True).__class__
                          .__mro__[0:1]) if False else (
            sum(p.numel() for p in model.upsample.parameters()) +
            model.vox_tokens.numel()
        )
        total_params = sum(p.numel() for p in model.parameters())
        log.info("ViBlock4: %d total params (%d head / vox_tokens)", total_params, head_params)

    if args.checkpoint is not None and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if is_main:
            log.info("Loaded checkpoint: epoch=%d occ_iou=%.4f",
                     ckpt.get("epoch", 0), ckpt.get("occ_iou", 0))

    if is_ddp:
        # Whole model (vision + vox_tokens + upsample) as one DDP module.
        # find_unused_parameters=True needed for NaFlex dynamic patching.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Two lr groups: backbone (vision params) and head (vox_tokens + upsample)
    _m = model.module if is_ddp else model
    backbone_params = [p for p in _m.vision.parameters() if p.requires_grad]
    head_params     = list(_m.vox_tokens.unsqueeze(0).__class__.__mro__) if False else (
        [_m.vox_tokens] + list(_m.upsample.parameters())
    )
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,     "lr": args.head_lr},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WARNING: CLASS-WEIGHTED LOSS — air (0) and stone (83) are downweighted.
    # Air dominates ~90% of voxels; stone is ubiquitous underground filler.
    # Without this, the model collapses to predicting air/stone everywhere.
    # To ablate: set all weights to 1.0.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    _w = torch.ones(n_classes + 1, device=device)
    _w[0]  = 0.1   # air
    _w[83] = 0.3   # stone (vocab idx 82, model class 83)
    criterion = nn.CrossEntropyLoss(weight=_w, ignore_index=-1)
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
            tags    = ["viblock4", f"unfrz{args.unfreeze_n}", f"inject{args.inject_at}",
                       "ddp", "amp", "image-only", "3d-upsample"],
            config  = {
                "backbone":    m_cfg["backbone"],
                "inject_at":   args.inject_at,
                "vox_res":     m_cfg.get("vox_res", 8),
                "mid_dim":     m_cfg.get("mid_dim", 256),
                "n_classes":   n_classes,
                "unfreeze_n":  args.unfreeze_n,
                "head_lr":     args.head_lr,
                "backbone_lr": args.backbone_lr,
                "epochs":      args.epochs,
                "batch_size":  bs,
                "world_size":  world_size,
                "git_sha":     git_sha,
            },
        )

    best_occ_iou = 0.0
    save_every   = cfg["checkpointing"]["save_every_epochs"]
    all_params   = backbone_params + head_params

    if is_ddp:
        dist.barrier()

    if is_main:
        _m = model.module if is_ddp else model
        render_wandb_samples(_m, proc, all_train, val_sessions, cls_to_sid,
                             n_each=3, device=device, epoch=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_steps    = 0

        def _infinite(loader):
            while True:
                yield from loader

        src = _infinite(train_loader) if max_steps is not None else train_loader
        for frames, cam_blocks in islice(src, max_steps):
            cam_blocks = cam_blocks.to(device)   # (B, 32, 32, 32)
            B = cam_blocks.shape[0]
            inp = proc_frames(proc, frames, device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    pixel_values         = inp["pixel_values"],
                    pixel_attention_mask = inp["pixel_attention_mask"],
                    spatial_shapes       = inp["spatial_shapes"],
                )  # (B, 32, 32, 32, n_classes+1)
                loss = criterion(logits.reshape(B * 32768, n_classes + 1),
                                 cam_blocks.reshape(B * 32768))

            if torch.isnan(loss):
                loss = logits.sum() * 0.0

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_steps    += 1

        scheduler.step()
        train_loss /= max(n_steps, 1)

        if is_ddp:
            loss_t = torch.tensor([train_loss * n_steps, float(n_steps)], device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            train_loss = (loss_t[0] / loss_t[1]).item()
            dist.barrier()

        if is_main:
            _m = model.module if is_ddp else model
            do_eval = (epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs)

            wb_log = {
                "train_loss":  train_loss,
                "lr_head":     optimizer.param_groups[1]["lr"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
            }

            if do_eval:
                metrics = evaluate(_m, proc, val_loader, n_classes, device)
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
                render_wandb_samples(_m, proc, all_train, val_sessions, cls_to_sid,
                                     n_each=3, device=device, epoch=epoch)
                if metrics["occ_iou"] > best_occ_iou:
                    best_occ_iou = metrics["occ_iou"]
                    raw = getattr(_m, "_orig_mod", _m)
                    torch.save({"model": raw.state_dict(), "epoch": epoch,
                                "occ_iou": metrics["occ_iou"], "type_acc": metrics["type_acc"]},
                               ckpt_dir / "best_viblock4.pt")
                    log.info("New best occ_iou=%.4f → saved best_viblock4.pt", best_occ_iou)
            else:
                log.info("Epoch %d/%d  loss=%.4f  (eval every %d epochs)",
                         epoch, args.epochs, train_loss, args.eval_every)

            if epoch % save_every == 0:
                raw = getattr(_m, "_orig_mod", _m)
                torch.save({"model": raw.state_dict(), "epoch": epoch},
                           ckpt_dir / f"viblock4_epoch{epoch:04d}.pt")

            import wandb as _wb
            _wb.log(wb_log, step=epoch)

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
