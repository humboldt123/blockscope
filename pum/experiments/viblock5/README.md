# viblock5 — clean re-baseline of viblock4

viblock5 is **viblock4's architecture, unchanged**, with the data/label/loss bugs
from the code audit fixed. The point is a number that is directly comparable to
viblock4's logged `occ_iou=0.7635` and tells us how much of it was real perception
versus a learned terrain prior. The geometry-aware lifting work (ray PE / anchored
queries / FLoSP, living.tex **E2**) is intentionally a *separate* future
experiment so this re-baseline stays uncontaminated.

## What changed vs viblock4

| ID | Fix | Where |
|----|-----|-------|
| **E10** | **Visibility-masked loss** (headline). viblock4 ran CE over all 32,768 voxels — ~half behind the camera — so the loss rewarded a terrain prior. viblock5 weights each voxel by Furnace's visibility mask: visible (`m=1`) full weight, occluded/behind-camera weight `occluded_weight` (default `0.0`). | `precompute_cam_vis.py`, `dataset.py`, `train_finetune.py:masked_ce` |
| **T1** | Rotate-to-camera "fake air" front plane no longer pollutes the loss — those cells have `m=0`, so masking drops them for free. `cam_classes` is left byte-for-byte identical to viblock4's, keeping the comparison clean. | (subsumed by E10) |
| **T2** | Worker sharding. Old code shuffled the session list with each worker's own RNG, then sliced by worker id → sessions duplicated/dropped per epoch. Now: deterministic disjoint shard first, shuffle within the shard. | `dataset.py:__iter__` |
| **T3** | Stone class weight derived from the vocab (`classes.index("stone") + 1`) instead of the hardcoded index `83`. | `train_finetune.py:build_class_weights` |
| **T4** | Metrics are **visible-only** (the honest perception number; used for checkpoint selection) and exclude OOV voxels (`-1`) from the confusion. A `full_occ_iou` is also logged as a bridge to viblock4's old full-grid number. | `train_finetune.py:evaluate` |
| **C1** | Deleted the `cap.set(CAP_PROP_POS_FRAMES, tick_idx)` video fallback — tick indices are not video frame indices. Frames come only from the precomputed JPEGs (extracted correctly via `frame_mapping.jsonl`). | `dataset.py` |

`cam_classes/`, `frames/`, and `viblock3_data.npz` are **reused as-is** from the
viblock3/4 pipeline — viblock5 only *adds* the `cam_vis/` masks. This is what makes
it a true apples-to-apples re-baseline: identical inputs and labels, only the loss
masking and metrics differ.

## Run order (on Brev)

```bash
# 0. Prereq: cam_classes/ and frames/ already exist from the viblock3/4 smelt.
#    If not, run pum/experiments/viblock3/precompute_cam_classes.py and the
#    frame extractor first.

# 1. Generate the visibility masks (the E10 enabler).
#    Prefers Furnace's per-tick tick_<idx>.npz "m"; falls back to the GPU
#    renderer from world_states.bin if those npz were not retained.
PYTHONPATH=/home/vvm33/pum /home/nvidia/miniconda3/bin/python \
  -m pum.experiments.viblock5.precompute_cam_vis        # add --force to rebuild

# 2. Train (same launcher as viblock4; inject_at defaults to 10).
EXPERIMENT=viblock5 UNFREEZE_N=2 EPOCHS=100 HEAD_LR=3e-4 BACKBONE_LR=1e-5 \
  CUDA_VISIBLE_DEVICES=0,1,3,4,6,7 N_GPUS=6 bash /home/vvm33/pum/train.sh
```

To change `inject_at` or `occluded_weight` via the launcher, use
`EXTRA_ARGS="--inject_at 8 --occluded_weight 0.1"`.

## Reading the WandB metrics

- **`occ_iou`** — visible-only occupancy IoU. **This is the real perception number**
  and drives checkpoint selection (`best_viblock5.pt`). Expect it to differ from
  viblock4's 0.7635; that gap is the point of the experiment.
- **`full_occ_iou`** — full-grid IoU (OOV excluded). Roughly viblock4's old metric;
  logged only as a bridge for continuity, not for selection.
- **`type_acc`** — block-type accuracy over visible occupied voxels.

## Known limitations / deliberately out of scope

- `precompute_cam_vis.py` depends on either the per-tick `tick_<idx>.npz` masks or
  a re-render via the baker cache. If neither is available for a session, those
  ticks are **skipped in training** (never supervised with a fabricated all-visible
  mask). Re-smelt with `labeler --localize` to regenerate masks.
- The residual half of **T1** — a real world slab rotated out of the asymmetric
  32³ capture window at 90°/180° — is an inherent capture-window limitation,
  fixable only in the recording mod, and is not addressed here.
- No recipe changes (warmup / LLRD / EMA / augmentation) and no architecture
  changes — those are separate experiments by design.
