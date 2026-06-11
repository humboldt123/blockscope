"""
train_finetune.py — viblock5 DDP + AMP training (clean re-baseline of viblock4).

The MODEL is identical to viblock4 (SigLIP + 8^3 injected voxel slot tokens +
3-D conv upsample).  Everything different is a correctness fix, so the resulting
number is directly comparable to viblock4's logged occ_iou and tells us how much
of 0.7635 was real perception vs a terrain prior.

Fixes baked in here (data-side fixes live in dataset.py / precompute_cam_vis.py):

  E10 (headline) — VISIBILITY-MASKED LOSS.  viblock4 ran CE over all 32,768
      voxels, ~half of them behind the camera, so the loss rewarded a learned
      terrain prior.  viblock5 weights each voxel by Furnace's visibility mask:
      visible voxels (m=1) get full weight, occluded / behind-camera voxels get
      `occluded_weight` (default 0.0 — single-frame perception stage).

  T3 — the stone class weight is derived from the vocab (`classes.index("stone")
      + 1`) instead of the hardcoded index 83, which silently re-points if the
      block set / vocab ordering changes.

  T4 — metrics.  occ_iou / type_acc are now reported VISIBLE-ONLY (the honest
      perception number, used for checkpoint selection) AND OOV voxels (-1) are
      excluded from the confusion entirely.  A `full_occ_iou` is also logged as a
      bridge to viblock4's old (full-grid) metric.

Launch (same harness as viblock4):
  EXPERIMENT=viblock5 UNFREEZE_N=2 EPOCHS=100 HEAD_LR=3e-4 BACKBONE_LR=1e-5 \\
    CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 N_GPUS=6 bash /home/vvm33/pum/train.sh
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from itertools import islice
from pathlib import Path

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
from pum.experiments.viblock5.dataset import ViBlock5Dataset, collate_fn, STEM
from pum.experiments.viblock5.model import ViBlock5

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VOXELS = 32 * 32 * 32   # 32,768


# ── Session split ─────────────────────────────────────────────────────────────

def get_session_names(split: str, n_train: int) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / STEM).exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


# ── Class weights (T3: stone derived from vocab, not hardcoded index) ──────────

def build_class_weights(vocab_path: Path, n_classes: int, device: torch.device) -> torch.Tensor:
    """
    Air and stone are downweighted so the model doesn't collapse to predicting
    them everywhere.  Stone's class index is looked up from the vocab's `classes`
    list (+1 for the air shift) instead of the old hardcoded 83.
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    classes = vocab.get("classes", [])
    w = torch.ones(n_classes + 1, device=device)
    w[0] = 0.1   # air
    if "stone" in classes:
        stone_cls = classes.index("stone") + 1   # +1: class 0 reserved for air
        w[stone_cls] = 0.3
        log.info("Class weights: air=0.1, stone(class %d)=0.3", stone_cls)
    else:
        log.warning("'stone' not found in vocab classes — stone weight left at 1.0")
    return w


# ── Visibility-masked loss (E10) ───────────────────────────────────────────────

def masked_ce(criterion_none: nn.CrossEntropyLoss,
              logits: torch.Tensor, cam_blocks: torch.Tensor, cam_vis: torch.Tensor,
              occluded_weight: float) -> torch.Tensor:
    """
    Per-voxel CE weighted by visibility.
      criterion_none : CrossEntropyLoss(weight=class_w, ignore_index=-1, reduction="none")
      logits         : (B, 32, 32, 32, C)
      cam_blocks     : (B, 32, 32, 32) int64   (-1 = OOV, ignored)
      cam_vis        : (B, 32, 32, 32) uint8    (1 = visible)
    Visible voxels weight 1.0; occluded / behind-camera weight `occluded_weight`.
    OOV voxels (cam_blocks == -1) are excluded entirely.
    """
    B = cam_blocks.shape[0]
    C = logits.shape[-1]
    ce  = criterion_none(logits.reshape(B * VOXELS, C),
                         cam_blocks.reshape(B * VOXELS))            # (B*V,)
    vis = cam_vis.reshape(B * VOXELS) > 0
    w   = torch.where(vis, torch.ones_like(ce), torch.full_like(ce, occluded_weight))
    w   = w * (cam_blocks.reshape(B * VOXELS) != -1).to(w.dtype)    # drop OOV
    denom = w.sum().clamp_min(1.0)
    return (ce * w).sum() / denom


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


# ── Metrics (T4: visible-only selection metric + OOV exclusion) ────────────────

@torch.no_grad()
def evaluate(model, proc, val_loader, criterion_none, occluded_weight, n_classes, device):
    model.eval()
    total_loss = 0.0
    n_steps    = 0
    # visible-only confusion (the honest perception metric)
    v_tp = v_fp = v_fn = 0
    v_type_correct = v_type_total = 0
    # full-grid confusion excluding OOV (bridge to viblock4's old number)
    f_tp = f_fp = f_fn = 0

    for frames, cam_blocks, cam_vis in val_loader:
        cam_blocks = cam_blocks.to(device)   # (B, 32, 32, 32)
        cam_vis    = cam_vis.to(device)
        inp = proc_frames(proc, frames, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                pixel_values         = inp["pixel_values"],
                pixel_attention_mask = inp["pixel_attention_mask"],
                spatial_shapes       = inp["spatial_shapes"],
            )  # (B, 32, 32, 32, n_classes+1)
            loss = masked_ce(criterion_none, logits, cam_blocks, cam_vis, occluded_weight)

        total_loss += loss.item()
        n_steps    += 1

        pred     = logits.argmax(dim=-1)        # (B, 32, 32, 32)
        valid    = cam_blocks >= 0              # exclude OOV (-1)   [T4]
        visible  = (cam_vis > 0) & valid
        gt_occ   = cam_blocks > 0
        pred_occ = pred > 0

        # visible-only occ_iou (checkpoint selection)
        v_tp += int((gt_occ  & pred_occ & visible).sum())
        v_fp += int((~gt_occ & pred_occ & visible).sum())
        v_fn += int((gt_occ  & ~pred_occ & visible).sum())

        # visible-only type accuracy over occupied voxels
        vmask = visible & gt_occ
        v_type_correct += int((pred[vmask] == cam_blocks[vmask]).sum())
        v_type_total   += int(vmask.sum())

        # full-grid occ_iou, OOV excluded (bridge metric)
        f_tp += int((gt_occ  & pred_occ & valid).sum())
        f_fp += int((~gt_occ & pred_occ & valid).sum())
        f_fn += int((gt_occ  & ~pred_occ & valid).sum())

    return {
        "val_loss":      total_loss / max(n_steps, 1),
        "occ_iou":       v_tp / max(v_tp + v_fp + v_fn, 1),   # VISIBLE — selection metric
        "type_acc":      v_type_correct / max(v_type_total, 1),
        "type_total":    v_type_total,
        "full_occ_iou":  f_tp / max(f_tp + f_fp + f_fn, 1),   # bridge to viblock4
    }


# ── WandB renders (rank 0 only) ───────────────────────────────────────────────

def _build_class_to_sid(vocab_path: Path) -> dict:
    """Map class index → a canonical block state ID for rendering.
    Prefers states with snowy=false, axis=y, facing=south, waterlogged=false, half=lower.
    """
    _PREF = {"snowy": {"false": 0, "true": 10},
             "axis":  {"y": 0, "x": 5, "z": 5},
             "facing": {"south": 0, "north": 1, "east": 1, "west": 1, "up": 2, "down": 2},
             "half":  {"lower": 0, "bottom": 0, "upper": 5, "top": 5},
             "waterlogged": {"false": 0, "true": 3},
             "type": {"single": 0, "double": 3, "left": 2, "right": 2}}

    try:
        from pathlib import Path as _Path
        _repo = _Path(__file__).resolve().parents[3]   # .../pum/
        state_props = {}
        for candidate in [
            _repo / "furnace/pipeline/cache/1.19.4/state_properties_1.19.4.json",
            _Path("/data/vvm33/pum_checkpoints/blocktype/state_properties_1.19.4.json"),
        ]:
            if candidate.exists():
                state_props = json.loads(candidate.read_text())
                break
    except Exception:
        state_props = {}

    with open(vocab_path) as f:
        v = json.load(f)

    def _score(sid: int) -> int:
        sp = state_props.get(str(sid), {})
        props = sp.get("properties", {})
        return sum(_PREF.get(k, {}).get(v2, 0) for k, v2 in props.items())

    out = {0: 0}
    from collections import defaultdict
    by_class: dict = defaultdict(list)
    for sid_str, cls in v["sid_to_class"].items():
        by_class[int(cls) + 1].append(int(sid_str))
    for vb_cls, sids in by_class.items():
        out[vb_cls] = min(sids, key=_score)
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
    ap.add_argument("--occluded_weight", type=float, default=None,
                    help="Loss weight for occluded / behind-camera voxels "
                         "(default: config training.occluded_weight, else 0.0)")
    ap.add_argument("--run_name",    default="viblock5")
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

    occluded_weight = (args.occluded_weight if args.occluded_weight is not None
                       else float(cfg["training"].get("occluded_weight", 0.0)))

    vocab_path   = Path(cfg["data"]["vocab_path"])
    n_train      = cfg["data"]["n_train"]
    all_train    = get_session_names("train", n_train)
    val_sessions = get_session_names("val", n_train)
    cls_to_sid   = _build_class_to_sid(vocab_path)

    rank           = dist.get_rank() if is_ddp else 0
    train_sessions = all_train[rank::world_size]

    if is_main:
        log.info("DDP world_size=%d | train sessions=%d | val sessions=%d | occluded_weight=%.3f",
                 world_size, len(all_train), len(val_sessions), occluded_weight)

    bs    = cfg["training"]["batch_size"]
    m_cfg = cfg["model"]
    with open(vocab_path) as _vf:
        n_classes = json.load(_vf)["n_classes"]   # read from vocab

    # Online SigLIP mode (voxel tokens live inside the encoder).
    train_ds     = ViBlock5Dataset(train_sessions, vocab_path, shuffle=True)
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
        val_ds     = ViBlock5Dataset(val_sessions, vocab_path, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, collate_fn=collate_fn,
                                num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    vision, proc = load_siglip(m_cfg["backbone"], args.unfreeze_n, device)
    model = ViBlock5(
        vision    = vision,
        n_classes = n_classes,
        inject_at = args.inject_at,
        vox_res   = m_cfg.get("vox_res", 8),
        feat_dim  = m_cfg["feat_dim"],
        mid_dim   = m_cfg.get("mid_dim", 256),
    ).to(device)

    if is_main:
        head_params_n = sum(p.numel() for p in model.upsample.parameters()) + model.vox_tokens.numel()
        total_params  = sum(p.numel() for p in model.parameters())
        log.info("ViBlock5: %d total params (%d head / vox_tokens)", total_params, head_params_n)

    if args.checkpoint is not None and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if is_main:
            log.info("Loaded checkpoint: epoch=%d occ_iou=%.4f",
                     ckpt.get("epoch", 0), ckpt.get("occ_iou", 0))

    if is_ddp:
        # Whole model as one DDP module. find_unused_parameters=True needed for
        # NaFlex dynamic patching (variable patch counts per batch).
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    _m = model.module if is_ddp else model
    backbone_params = [p for p in _m.vision.parameters() if p.requires_grad]
    head_params     = [_m.vox_tokens] + list(_m.upsample.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,     "lr": args.head_lr},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    # Class-weighted CE, reduction='none' so masked_ce can apply per-voxel
    # visibility weights on top (E10).  ignore_index=-1 zeroes OOV voxels.
    class_w        = build_class_weights(vocab_path, n_classes, device)
    criterion_none = nn.CrossEntropyLoss(weight=class_w, ignore_index=-1, reduction="none")

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
            tags    = ["viblock5", f"unfrz{args.unfreeze_n}", f"inject{args.inject_at}",
                       "ddp", "amp", "image-only", "3d-upsample", "vis-masked", "rebaseline"],
            config  = {
                "backbone":        m_cfg["backbone"],
                "inject_at":       args.inject_at,
                "vox_res":         m_cfg.get("vox_res", 8),
                "mid_dim":         m_cfg.get("mid_dim", 256),
                "n_classes":       n_classes,
                "unfreeze_n":      args.unfreeze_n,
                "head_lr":         args.head_lr,
                "backbone_lr":     args.backbone_lr,
                "occluded_weight": occluded_weight,
                "epochs":          args.epochs,
                "batch_size":      bs,
                "world_size":      world_size,
                "git_sha":         git_sha,
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
        for frames, cam_blocks, cam_vis in islice(src, max_steps):
            cam_blocks = cam_blocks.to(device)   # (B, 32, 32, 32)
            cam_vis    = cam_vis.to(device)
            inp = proc_frames(proc, frames, device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    pixel_values         = inp["pixel_values"],
                    pixel_attention_mask = inp["pixel_attention_mask"],
                    spatial_shapes       = inp["spatial_shapes"],
                )  # (B, 32, 32, 32, n_classes+1)
                loss = masked_ce(criterion_none, logits, cam_blocks, cam_vis, occluded_weight)

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
                metrics = evaluate(_m, proc, val_loader, criterion_none,
                                   occluded_weight, n_classes, device)
                log.info(
                    "Epoch %d/%d  loss=%.4f  val_loss=%.4f  vis_occ_iou=%.4f  "
                    "type_acc=%.4f  full_occ_iou=%.4f  n=%d",
                    epoch, args.epochs, train_loss,
                    metrics["val_loss"], metrics["occ_iou"], metrics["type_acc"],
                    metrics["full_occ_iou"], metrics["type_total"],
                )
                wb_log.update({
                    "val_loss":     metrics["val_loss"],
                    "occ_iou":      metrics["occ_iou"],       # visible — selection metric
                    "type_acc":     metrics["type_acc"],
                    "full_occ_iou": metrics["full_occ_iou"],  # bridge to viblock4
                })
                render_wandb_samples(_m, proc, all_train, val_sessions, cls_to_sid,
                                     n_each=3, device=device, epoch=epoch)
                if metrics["occ_iou"] > best_occ_iou:
                    best_occ_iou = metrics["occ_iou"]
                    raw = getattr(_m, "_orig_mod", _m)
                    torch.save({"model": raw.state_dict(), "epoch": epoch,
                                "occ_iou": metrics["occ_iou"], "type_acc": metrics["type_acc"],
                                "full_occ_iou": metrics["full_occ_iou"]},
                               ckpt_dir / "best_viblock5.pt")
                    log.info("New best vis_occ_iou=%.4f → saved best_viblock5.pt", best_occ_iou)
            else:
                log.info("Epoch %d/%d  loss=%.4f  (eval every %d epochs)",
                         epoch, args.epochs, train_loss, args.eval_every)

            if epoch % save_every == 0:
                raw = getattr(_m, "_orig_mod", _m)
                torch.save({"model": raw.state_dict(), "epoch": epoch},
                           ckpt_dir / f"viblock5_epoch{epoch:04d}.pt")

            import wandb as _wb
            _wb.log(wb_log, step=epoch)

        if is_ddp:
            dist.barrier()

    if is_main:
        import wandb as _wb
        _wb.finish()
        log.info("Training complete. Best vis_occ_iou=%.4f", best_occ_iou)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
