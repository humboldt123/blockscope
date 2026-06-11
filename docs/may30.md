Remember, not every tick has a valid frame because of how frames are recorded:
tick 1  → frame 0,1     (two video frames for the same tick — client ran faster than 20fps)
tick 11 → frame 2       (ticks 2-10 = 9 ticks with NO video frames = loading screen)
tick 21 → frame 3       (ticks 12-20 = same — still loading)
tick 28 → frame 9
tick 29 → frame 10
...
tick 43 → frame 24
tick 45 → frame 25      (tick 44 skipped — one frame dropped during gameplay)
For tick 43 specifically: raw tick 44 is simply absent from the file. The game briefly didn't render a frame at that tick (common during world state updates).

Implication for the ML model:

This is expected behavior. The fix is simple — filter at training time:


# Only use ticks that have a corresponding video frame
valid_ticks = [t for t in range(tick_count) if frame_mapping.get(t) is not None]


---

## TODO: viblock6 — geometry-aware lifting (E2)

viblock5 is the clean re-baseline (same slot-token arch as viblock4, all audit
bugs fixed, visibility-masked loss). viblock6 is the next experiment: replace
the anonymous slot tokens with geometry-anchored queries and see if that buys
the sample efficiency predicted by SSC literature (+9-12 mIoU vs anonymous).

Architecture changes relative to viblock5 (keep as one PR so the ablation is
clean; commit order = build order below):

1. **Per-patch ray PE** — for each patch, compute its central ray direction in
   the *label frame* (true residual yaw, pitch, per-tick FOV, vertical-FOV
   convention from renderer.py — NOT the horizontal-FOV/mirrored convention in
   vis_dataset.py which is wrong). MLP-encode the direction and add to patch
   embeddings before the first encoder layer. Needs `local_yaw` and per-tick
   `pitch`/`fov` piped through the dataset.

2. **Anchored voxel queries** — each of the 512 slot tokens gets a 3D position
   embedding of its cell centre in camera space (label frame), MLP-encoded and
   added to the learned base token. Additionally, FLoSP-style initialisation:
   project the cell centre to the image plane (using the same per-tick intrinsics
   from step 1), bilinearly sample from patch features, and use that as the token
   initial value. Behind-camera cells get a learned "unobservable" embedding
   instead of an image sample.

3. **Readout depth 2 → 4** — inject voxel tokens earlier (inject_at=8 or add 2
   extra decoder cross-attn layers after the backbone). All published SSC methods
   use ≥3 joint layers; viblock4's 2 is a known weak point.

4. **Aux depth head** — GPU renderer's ID-buffer pass already gives per-pixel
   visible cell indices (renderer.py:551-559); depth = ‖cell − cam‖. Small head
   on patch features, weight ~0.1-0.2.  +2-4 mIoU consistently in BEVDet/FB-OCC.

Key files to touch:
  pum/experiments/viblock6/model.py      (new arch)
  pum/experiments/viblock6/dataset.py    (pipe local_yaw + pitch + fov per tick)
  pum/experiments/viblock6/train_finetune.py
  furnace/pipeline/labeler.py            (write depth_gt if not already in npz)

Ablation metric: vis_occ_iou vs viblock5 baseline on identical data.
Expected: +5-12 mIoU based on VoxFormer/MonoScene geometry-PE ablations.

