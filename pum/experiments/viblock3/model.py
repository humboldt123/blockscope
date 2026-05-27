"""
ViBlock3 — image-only dense voxel semantic prediction.

Input:  SigLIP patch grid  (B, H_p, W_p, feat_dim)
Output: voxel logits       (B, 32, 32, 32, n_classes+1)
        class 0 = air/empty, 1..n_classes = block types

Architecture:
  1. Patch projector: Linear(feat_dim → d_model)
  2. Voxel queries initialised with factorised positional encodings:
       q[i,j,k] = E_x[i] + E_y[j] + E_z[k]
     3 × 32 learned vectors instead of 32³ = 32 768 independent embeddings.
     Nearby voxels share embedding components → smooth interpolation.
  3. n_layers × (VoxelCrossAttn + FFNBlock)
  4. Linear head → n_classes+1 logits per voxel

VoxelCrossAttn — learned reference points, no camera pose:
  Each voxel query predicts K UV reference points per head via sigmoid(MLP(q)).
  Features are bilinearly sampled from the projected image feature map at those
  points and combined with per-head softmax weights.  The model must learn through
  training where each camera-space voxel tends to appear in the image — no
  geometric prior is injected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelCrossAttn(nn.Module):
    """
    Multi-head deformable cross-attention from voxel queries to image features.

    Each of n_heads heads independently predicts n_ref UV reference points and
    a softmax attention weight over them.  Features are bilinearly sampled from
    the image feature map (no fixed camera projection).

    All weights initialised to zero for the ref predictor so training begins
    from uniformly distributed reference points (sigmoid(0) = 0.5, i.e. centre
    of the image) and learns from there.
    """

    def __init__(self, d_model: int = 256, n_ref: int = 4, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_ref   = n_ref
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.ref_head   = nn.Linear(d_model, n_heads * n_ref * 2)   # → UV coords
        self.attn_wt    = nn.Linear(d_model, n_heads * n_ref)        # → weights
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj   = nn.Linear(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)

        nn.init.zeros_(self.ref_head.weight)
        nn.init.zeros_(self.ref_head.bias)   # sigmoid(0)=0.5 → start at image centre

    def forward(self, queries: torch.Tensor, img_map: torch.Tensor) -> torch.Tensor:
        """
        queries : (B, Q, d_model)
        img_map : (B, d_model, H_p, W_p)  — projected image features
        returns : (B, Q, d_model)          — residual added internally
        """
        B, Q, d = queries.shape
        _, _, Hp, Wp = img_map.shape

        # Reference UV coords in [0,1] — per head, per ref point
        ref = torch.sigmoid(self.ref_head(queries))          # (B, Q, nh*nr*2)
        ref = ref.view(B, Q, self.n_heads, self.n_ref, 2)   # (B, Q, nh, nr, 2)

        # Per-head attention weights over reference points
        wt = F.softmax(
            self.attn_wt(queries).view(B, Q, self.n_heads, self.n_ref),
            dim=-1,
        )  # (B, Q, nh, nr)

        # Project image features, split into heads
        val = self.value_proj(img_map.permute(0, 2, 3, 1).reshape(B, Hp * Wp, d))
        val = val.reshape(B, Hp, Wp, self.n_heads, self.d_head)
        val = val.permute(0, 3, 4, 1, 2).reshape(B * self.n_heads, self.d_head, Hp, Wp)

        # Flatten (B, Q, nh, nr, 2) → (B*nh, Q*nr, 1, 2) for grid_sample
        ref_flat = ref.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, Q * self.n_ref, 1, 2)
        ref_grid = ref_flat * 2.0 - 1.0   # [0,1] → [-1,1] for grid_sample

        # Sample image features at reference points
        sampled = F.grid_sample(val, ref_grid, align_corners=False, mode="bilinear")
        # sampled: (B*nh, d_head, Q*nr, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)      # (B*nh, Q*nr, d_head)
        sampled = sampled.reshape(B, self.n_heads, Q, self.n_ref, self.d_head)
        sampled = sampled.permute(0, 2, 1, 3, 4)            # (B, Q, nh, nr, d_head)

        # Weighted sum over reference points, flatten heads
        out = (wt.unsqueeze(-1) * sampled).sum(dim=3)       # (B, Q, nh, d_head)
        out = self.out_proj(out.reshape(B, Q, d))

        return self.norm(queries + out)


class FFNBlock(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ViBlock3(nn.Module):
    """
    Image-only 32³ camera-relative block-type prediction.

    No camera pose, no block coordinates, no UV lookup — purely image-driven.
    The model must infer 3-D geometry from visual evidence alone.
    """

    def __init__(
        self,
        n_classes: int,
        d_model:   int = 256,
        n_ref:     int = 4,
        n_heads:   int = 8,
        n_layers:  int = 2,
        feat_dim:  int = 768,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes   # excludes air; head outputs n_classes+1

        self.patch_proj = nn.Linear(feat_dim, d_model)

        # Factorised positional encodings: one embedding table per axis
        self.pos_x = nn.Embedding(32, d_model)
        self.pos_y = nn.Embedding(32, d_model)
        self.pos_z = nn.Embedding(32, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                VoxelCrossAttn(d_model, n_ref, n_heads),
                FFNBlock(d_model),
            ])
            for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, n_classes + 1)

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        """
        patch_grid : (B, H_p, W_p, feat_dim)  — SigLIP last_hidden_state grid
        returns    : (B, 32, 32, 32, n_classes+1)
        """
        B, Hp, Wp, _ = patch_grid.shape

        img_map = self.patch_proj(patch_grid).permute(0, 3, 1, 2)  # (B, d_model, Hp, Wp)

        ix = torch.arange(32, device=patch_grid.device)
        iy = torch.arange(32, device=patch_grid.device)
        iz = torch.arange(32, device=patch_grid.device)

        # Broadcast-sum the three embedding tables → (32, 32, 32, d_model)
        queries = (
            self.pos_x(ix)[:, None, None, :]
            + self.pos_y(iy)[None, :, None, :]
            + self.pos_z(iz)[None, None, :, :]
        ).reshape(1, 32 * 32 * 32, self.d_model)
        queries = queries.expand(B, -1, -1).contiguous()   # (B, 32768, d_model)

        for cross_attn, ffn in self.layers:
            queries = cross_attn(queries, img_map)
            queries = ffn(queries)

        logits = self.head(queries)                         # (B, 32768, n_classes+1)
        return logits.view(B, 32, 32, 32, self.n_classes + 1)
