"""
train_finetune.py — Partial SigLIP fine-tuning with DDP + AMP + precomputed frames.

Speedups vs original:
  1. bf16 AMP  — forward in bf16, master weights fp32; ~2x faster SigLIP on A100
  2. DDP       — torchrun shards sessions across N GPUs; near-linear throughput scaling
  3. Frames    — JPEG frame cache eliminates H.264 seek bottleneck (run precompute_frames.py first)

With 6× A100s: ~32 min/epoch → ~3-4 min/epoch.
WandB renders (video | GT | predicted) uploaded every epoch from rank 0.

Launch:
  torchrun --nproc_per_node=6 train_finetune.py [--unfreeze_n 2 --epochs 30 ...]
  or via:  bash /home/vvm33/train.sh   (sets torchrun automatically)

Single-GPU fallback:
  WORLD_SIZE=1 python train_finetune.py --device cuda:0 ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from itertools import islice
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT
from pum.experiments.viblock2.train import ViBlock2Head, load_vocab, get_session_names

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SIGLIP_W  = 40
SIGLIP_H  = 22
DATA_STEM  = "viblock2_data.npz"
FRAMES_DIR = "frames"   # relative to session labels dir


# ── Frame dataset (JPEG cache, no H.264 seeking) ─────────────────────────────

class ViBlock2FrameDataset(IterableDataset):
    """
    Yields (frame_rgb, block_uvs, block_sids, block_xyz) per tick.
    Loads frames from precomputed JPEGs (no video seeking).
    Falls back to cv2 video seek if JPEG not found.
    """

    def __init__(self, session_names: list[str], shuffle: bool = True):
        self.session_names = session_names
        self.shuffle = shuffle

    def __iter__(self):
        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            per   = int(np.ceil(len(sessions) / wi.num_workers))
            sessions = sessions[wi.id * per: wi.id * per + per]

        for session in sessions:
            labels_dir = SMELTED_ROOT / session / "labels"
            data_path  = labels_dir / DATA_STEM
            frames_dir = labels_dir / FRAMES_DIR
            video_path = RAW_ROOT / session / "video.mp4"
            if not data_path.exists():
                continue

            d             = np.load(data_path)
            tick_indices  = d["tick_indices"]
            offsets       = d["block_offsets"]
            block_uvs     = d["block_uvs"]
            block_sids    = d["block_sids"]
            block_xyz     = d["block_xyz"]

            use_jpegs = frames_dir.exists()
            cap = None
            if not use_jpegs and video_path.exists():
                cap = cv2.VideoCapture(str(video_path))

            order = list(range(len(tick_indices)))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                s, e = int(offsets[k]), int(offsets[k + 1])
                if s >= e:
                    continue

                tick_idx = int(tick_indices[k])
                frame_rgb = None

                if use_jpegs:
                    jpg = frames_dir / f"{tick_idx:06d}.jpg"
                    if jpg.exists():
                        bgr = cv2.imread(str(jpg))
                        if bgr is not None:
                            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if frame_rgb is None and cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, tick_idx)
                    ok, bgr = cap.read()
                    if ok:
                        frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if frame_rgb is None:
                    continue

                # Skip loading screens / black frames (no visual signal)
                if frame_rgb.std() < 8.0:
                    continue

                yield (frame_rgb,
                       block_uvs[s:e],
                       block_sids[s:e].astype(np.int32),
                       block_xyz[s:e])

            if cap is not None:
                cap.release()


def collate_fn(batch):
    return (
        [b[0] for b in batch],
        [b[1] for b in batch],
        [b[2] for b in batch],
        [torch.from_numpy(b[3]) for b in batch],
    )


# ── SigLIP (partial unfreeze) ─────────────────────────────────────────────────

def load_siglip_partial(backbone_name: str, unfreeze_n: int, device: torch.device):
    from transformers import AutoModel, AutoProcessor
    log.info("Loading SigLIP %s, unfreezing last %d encoder blocks", backbone_name, unfreeze_n)
    model  = AutoModel.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    vision = model.vision_model.to(device)
    proc   = AutoProcessor.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")

    for p in vision.parameters():
        p.requires_grad_(False)

    layers  = vision.encoder.layers
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
    """Run SigLIP on a list of frames. Returns (B, H, W, 768) patch grid."""
    from PIL import Image
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


def uv_to_patch(uvs: np.ndarray):
    col = np.floor(uvs[:, 0] * SIGLIP_W).astype(np.int32).clip(0, SIGLIP_W - 1)
    row = np.floor(uvs[:, 1] * SIGLIP_H).astype(np.int32).clip(0, SIGLIP_H - 1)
    return row, col


def extract_block_feats(grid: torch.Tensor, uvs_list: list) -> torch.Tensor:
    parts = []
    for b, uvs in enumerate(uvs_list):
        rows, cols = uv_to_patch(uvs)
        parts.append(grid[b, rows, cols, :])
    return torch.cat(parts, dim=0)


# ── Eval (rank 0 only) ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(vision, proc, head, val_loader, class_lut, device):
    vision.eval(); head.eval()
    correct1 = correct5 = n_valid = n_total = 0
    for frames_np, uvs_list, sids_list, xyz_list in val_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grid  = siglip_patch_grid(vision, proc, frames_np, device, training=False)
            feats = extract_block_feats(grid, uvs_list)
        xyz          = torch.cat(xyz_list, dim=0).to(device)
        all_sids     = np.concatenate(sids_list)
        all_classes  = class_lut[np.clip(all_sids, 0, len(class_lut) - 1)]
        valid        = all_classes >= 0
        n_total     += len(all_classes)
        if valid.sum() == 0:
            continue
        targets = torch.from_numpy(all_classes[valid]).to(device)
        logits  = head(feats[valid].float(), xyz[valid])
        _, top5 = logits.topk(min(5, logits.shape[1]), dim=1)
        correct1 += (top5[:, :1] == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct5 += (top5        == targets.unsqueeze(1)).any(dim=1).sum().item()
        n_valid  += int(valid.sum())
    return {"top1": correct1 / max(n_valid, 1),
            "top5": correct5 / max(n_valid, 1),
            "n_valid": n_valid,
            "oov_rate": 1 - n_valid / max(n_total, 1)}


# ── WandB renders (rank 0 only) ───────────────────────────────────────────────

def _build_class_to_sid(vocab_path: Path) -> dict:
    with open(vocab_path) as f:
        v = json.load(f)
    out = {}
    for sid_str, cls in v["sid_to_class"].items():
        sid, cls = int(sid_str), int(cls)
        if cls not in out or sid < out[cls]:
            out[cls] = sid
    return out


@torch.no_grad()
def render_epoch_samples(vision, proc, head, val_sessions, class_lut,
                          vocab_path, device, n_samples=4):
    """Render n_samples ticks: [video | GT | predicted] → wandb.Image list."""
    import wandb
    from visualize import render_perspective_view, IMG_W, IMG_H  # type: ignore
    from io_helpers import load_world_states                      # type: ignore

    vision.eval(); head.eval()
    class_to_sid = _build_class_to_sid(vocab_path)

    session    = val_sessions[0]
    labels_dir = SMELTED_ROOT / session / "labels"
    frames_dir = labels_dir / FRAMES_DIR

    d             = np.load(labels_dir / DATA_STEM)
    tick_indices  = d["tick_indices"]
    block_offsets = d["block_offsets"]
    block_xyz_all = d["block_xyz"]
    block_sids_all= d["block_sids"]
    block_uvs_all = d["block_uvs"]

    vis_data   = np.load(labels_dir / "visibility_s20.npz", allow_pickle=True)
    vis_lookup = {int(vis_data["tick_indices"][i]): vis_data["visibility"][i]
                  for i in range(len(vis_data["tick_indices"]))}

    _, _, _, ws_blocks = load_world_states(labels_dir)

    ticks_jsonl = {}
    for line in (RAW_ROOT / session / "ticks.jsonl").read_text().splitlines():
        if line:
            t = json.loads(line)
            ticks_jsonl[int(t["tick"])] = t

    K      = len(tick_indices)
    step   = max(1, K // n_samples)
    sample = list(range(0, K, step))[:n_samples]

    wb_images = []
    for ki in sample:
        tick_num = int(tick_indices[ki])
        f_s, f_e = int(block_offsets[ki]), int(block_offsets[ki + 1])
        if f_e == f_s or tick_num >= len(ws_blocks):
            continue

        uvs_np  = block_uvs_all[f_s:f_e]
        xyz_np  = block_xyz_all[f_s:f_e]
        sids_gt = block_sids_all[f_s:f_e].astype(np.int32)

        # Load frame (JPEG or video fallback)
        frame_np = None
        jpg = frames_dir / f"{tick_num:06d}.jpg"
        if jpg.exists():
            bgr = cv2.imread(str(jpg))
            if bgr is not None:
                frame_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame_np is None:
            try:
                import av
                video_path = str(RAW_ROOT / session / "video.mp4")
                container  = av.open(video_path)
                for idx, frm in enumerate(container.decode(video=0)):
                    if idx == tick_num:
                        frame_np = frm.to_ndarray(format="rgb24").astype(np.uint8)
                        break
                container.close()
            except Exception:
                continue
        if frame_np is None:
            continue

        # Run SigLIP + head
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grid = siglip_patch_grid(vision, proc, [frame_np], device, training=False)
        rows, cols = uv_to_patch(uvs_np)
        feats_t  = grid[0, rows, cols, :].float().to(device)
        xyz_t    = torch.from_numpy(xyz_np).to(device)
        pred_cls = head(feats_t, xyz_t).argmax(dim=1).cpu().numpy()

        # Build predicted world state
        b_gt   = ws_blocks[tick_num].astype(np.int32)
        b_pred = np.zeros_like(b_gt)  # start from air; only fill model predictions
        for bi in range(len(pred_cls)):
            gi = int(round(float(xyz_np[bi, 0]) * 16 + 16))
            gj = int(round(float(xyz_np[bi, 1]) * 16 + 16))
            gk = int(round(float(xyz_np[bi, 2]) * 16 + 16))
            if 0 <= gi < 32 and 0 <= gj < 32 and 0 <= gk < 32:
                b_pred[gi, gj, gk] = class_to_sid.get(int(pred_cls[bi]), 0)

        m_vis = vis_lookup.get(tick_num,
            vis_lookup[min(vis_lookup, key=lambda t: abs(t - tick_num))]).astype(bool)

        p     = ticks_jsonl.get(tick_num, {}).get("player", {})
        pose  = {"x": p.get("x",0), "y": p.get("y",0), "z": p.get("z",0),
                 "yaw": p.get("yaw",0), "pitch": p.get("pitch",0), "fov": p.get("fov",70)}

        render_gt   = render_perspective_view(m_vis, b_gt,   pose, fullbright=True)
        render_pred = render_perspective_view(m_vis, b_pred, pose, fullbright=True)

        from PIL import Image as _PIL
        vid_resized = np.array(_PIL.fromarray(frame_np).resize((IMG_W, IMG_H)))
        panel = np.concatenate([vid_resized, render_gt, render_pred], axis=1)

        gt_cls = class_lut[np.clip(sids_gt, 0, len(class_lut) - 1)]
        valid  = gt_cls >= 0
        acc    = (pred_cls[valid] == gt_cls[valid]).mean() if valid.sum() else 0.0

        wb_images.append(wandb.Image(
            panel,
            caption=f"tick={tick_num}  top1={acc:.1%}  ({valid.sum()}/{len(sids_gt)} in-vocab)"
        ))

    return wb_images


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--checkpoint",   type=Path, default=None,
                    help="Path to checkpoint to resume from. Omit for scratch training.")
    ap.add_argument("--device",       default=None,
                    help="Override device (single-GPU mode only; ignored with torchrun)")
    ap.add_argument("--unfreeze_n",   type=int, default=2)
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--head_lr",      type=float, default=1e-4)
    ap.add_argument("--backbone_lr",  type=float, default=1e-5)
    ap.add_argument("--run_name",     default="viblock2_ft")
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
    class_lut, n_classes, _ = load_vocab(vocab_path)
    if is_main:
        log.info("Vocab: %d classes", n_classes)

    n_train       = cfg["data"]["n_train"]
    all_train     = get_session_names("train", n_train)
    val_sessions  = get_session_names("val",   n_train)

    # Shard training sessions by rank
    rank          = dist.get_rank() if is_ddp else 0
    train_sessions = all_train[rank::world_size]

    if is_main:
        log.info("DDP world_size=%d | this rank train sessions: %d | val sessions: %d",
                 world_size, len(train_sessions), len(val_sessions))

    # ── Datasets ──────────────────────────────────────────────────────────────
    bs = cfg["training"]["batch_size"]
    train_ds  = ViBlock2FrameDataset(train_sessions, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=bs,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # DDP step-count sync: all ranks must call backward() the same number of times.
    # Pre-count ticks, all-reduce the minimum, each rank stops there.
    n_ticks = sum(
        len(np.load(SMELTED_ROOT / s / "labels" / DATA_STEM)["tick_indices"])
        for s in train_sessions
        if (SMELTED_ROOT / s / "labels" / DATA_STEM).exists()
    )
    if is_ddp:
        n_t = torch.tensor([n_ticks // bs], device=device, dtype=torch.long)
        dist.all_reduce(n_t, op=dist.ReduceOp.MIN)
        max_steps = int(n_t.item())
    else:
        max_steps = None
    if is_main:
        log.info("Steps per epoch: max_steps=%s (sync'd across ranks)", max_steps)

    # Val loader only used on rank 0
    if is_main:
        val_ds   = ViBlock2FrameDataset(val_sessions, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs,
                                collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    vision, proc = load_siglip_partial(cfg["model"]["backbone"], args.unfreeze_n, device)
    m_cfg  = cfg["model"]
    head   = ViBlock2Head(m_cfg["feat_dim"], m_cfg["xyz_dim"],
                          m_cfg["hidden_dim"], n_classes, m_cfg["dropout"]).to(device)

    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        head.load_state_dict(ckpt["model"])
        if is_main:
            log.info("Loaded checkpoint: top1=%.4f epoch=%d",
                     ckpt.get("top1", 0), ckpt.get("epoch", 0))
    elif args.checkpoint is not None and is_main:
        log.warning("No checkpoint at %s — starting from scratch", args.checkpoint)
    elif is_main:
        log.info("No checkpoint specified — starting from scratch")

    if is_ddp:
        vision = DDP(vision, device_ids=[local_rank], find_unused_parameters=False)
        head   = DDP(head,   device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer + AMP ───────────────────────────────────────────────────────
    backbone_params = [p for p in vision.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params,    "lr": args.backbone_lr},
        {"params": head.parameters(),  "lr": args.head_lr},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler()

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    ckpt_dir  = Path(cfg["checkpointing"]["dir"])
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB (rank 0 only) ───────────────────────────────────────────────────
    if is_main:
        import wandb, subprocess as _sp
        _git_sha = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=args.run_name,
            tags=["finetune", f"unfrz{args.unfreeze_n}", "ddp", "amp"],
            config={
                "backbone":      cfg["model"]["backbone"],
                "feat_dim":      m_cfg["feat_dim"],
                "hidden_dim":    m_cfg["hidden_dim"],
                "n_classes":     n_classes,
                "unfreeze_n":    args.unfreeze_n,
                "head_lr":       args.head_lr,
                "backbone_lr":   args.backbone_lr,
                "epochs":        args.epochs,
                "batch_size":    bs,
                "effective_batch": bs * world_size,
                "world_size":    world_size,
                "weight_decay":  cfg["training"]["weight_decay"],
                "grad_clip":     1.0,
                "scheduler":     "CosineAnnealingLR",
                "amp":           "bf16",
                "n_train":       cfg["data"]["n_train"],
                "vocab_path":    str(vocab_path),
                "checkpoint":    str(args.checkpoint),
                "git_sha":       _git_sha,
            },
        )

    best_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        vision.train(); head.train()
        train_loss = 0.0
        n_steps    = 0

        def _infinite(loader):
            while True:
                yield from loader

        _src = _infinite(train_loader) if max_steps is not None else train_loader
        for step_idx, (frames_np, uvs_list, sids_list, xyz_list) in enumerate(islice(_src, max_steps)):

            with torch.autocast("cuda", dtype=torch.bfloat16):
                grid  = siglip_patch_grid(vision, proc, frames_np, device, training=True)
                feats = extract_block_feats(grid, uvs_list)
                xyz   = torch.cat(xyz_list, dim=0).to(device)

                all_sids    = np.concatenate(sids_list)
                all_classes = class_lut[np.clip(all_sids, 0, len(class_lut) - 1)]
                valid       = all_classes >= 0

                if valid.sum() > 0 and not torch.isnan(feats).any():
                    targets = torch.from_numpy(all_classes[valid]).to(device)
                    logits  = head(feats[valid], xyz[valid])
                    loss    = criterion(logits, targets)
                    if torch.isnan(loss):
                        # head still ran — gradients flow through both DDP modules
                        logits_zero = head(feats[:1], xyz[:1])
                        loss = feats.sum() * 0.0 + logits_zero.sum() * 0.0
                else:
                    # No valid targets — dummy forward through head keeps its DDP
                    # gradient hooks firing (find_unused_parameters=False requires this)
                    logits_zero = head(feats[:1], xyz[:1])
                    loss = feats.sum() * 0.0 + logits_zero.sum() * 0.0

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                backbone_params + list(head.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_steps    += 1

        scheduler.step()
        train_loss /= max(n_steps, 1)

        # All-reduce loss across ranks so rank 0 logs the global average
        if is_ddp:
            loss_t = torch.tensor([train_loss * n_steps, float(n_steps)], device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            train_loss = (loss_t[0] / loss_t[1]).item()

        if is_ddp:
            dist.barrier()   # wait for all ranks before rank 0 evaluates

        if is_main:
            _vision = vision.module if is_ddp else vision
            _head   = head.module   if is_ddp else head

            metrics = evaluate(_vision, proc, _head, val_loader, class_lut, device)

            log.info("Epoch %d/%d  loss=%.4f  top1=%.4f  top5=%.4f  n=%d",
                     epoch, args.epochs, train_loss,
                     metrics["top1"], metrics["top5"], metrics["n_valid"])

            # Render every epoch
            log_dict = {
                "epoch":        epoch,
                "train/loss":   train_loss,
                "val/top1":     metrics["top1"],
                "val/top5":     metrics["top5"],
                "val/oov_rate": metrics["oov_rate"],
                "lr_head":      scheduler.get_last_lr()[-1],
                "lr_backbone":  scheduler.get_last_lr()[0],
            }
            try:
                imgs = render_epoch_samples(
                    _vision, proc, _head, val_sessions, class_lut,
                    vocab_path, device, n_samples=4)
                if imgs:
                    log_dict["renders"] = imgs
            except Exception as e:
                log.warning("render_epoch_samples failed: %s", e)

            run.log(log_dict)

            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                torch.save({
                    "epoch":      epoch,
                    "model":      _head.state_dict(),
                    "vision":     {k: v for k, v in _vision.state_dict().items()
                                   if any(f"layers.{i}" in k
                                          for i in range(12 - args.unfreeze_n, 12))
                                   or "post_layernorm" in k},
                    "top1":       best_top1,
                    "n_classes":  n_classes,
                    "vocab":      str(vocab_path),
                    "backbone":   cfg["model"]["backbone"],
                    "unfreeze_n": args.unfreeze_n,
                }, ckpt_dir / "best_finetune.pt")
                log.info("  → best top1=%.4f saved", best_top1)

            if epoch % cfg["checkpointing"]["save_every_epochs"] == 0:
                torch.save({"epoch": epoch, "model": _head.state_dict()},
                           ckpt_dir / f"finetune_epoch_{epoch:03d}.pt")

        if is_ddp:
            dist.barrier()   # all ranks wait for rank 0 to finish eval/logging

    if is_main:
        log.info("Done. Best top1: %.4f", best_top1)
        run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
