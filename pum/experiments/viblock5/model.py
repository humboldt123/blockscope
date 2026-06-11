"""
ViBlock5 — SigLIP with injected voxel tokens + 3-D conv upsample.

ARCHITECTURE IS IDENTICAL TO viblock4 — viblock5 is a clean *re-baseline*, not a
new architecture.  The only changes vs viblock4 live in the data/label/loss path:

  * visibility-masked loss (E10)   — the headline fix; viblock4 supervised all
                                      32,768 voxels including the ~half behind the
                                      camera, so its occ_iou partly measured a
                                      terrain prior, not perception.
  * worker sharding (T2)           — see dataset.py
  * stone class weight from vocab (T3), visible-only metrics + OOV exclusion (T4)
  * deleted the broken tick-index video fallback (C1)  — see dataset.py

Keeping the architecture byte-for-byte identical is the whole point: it lets the
fixed run be compared apples-to-apples against viblock4's logged numbers, and
isolates "how much of 0.7635 was real perception" from any architecture change.
The geometry-aware lifting work (ray PE / anchored queries / FLoSP, living.tex
E2) is deliberately a *separate* future experiment so the re-baseline stays clean.

Model recap (see viblock4 for the full derivation):
  inject 8^3 = 512 learned voxel slot tokens into the SigLIP encoder at layer
  `inject_at`; they aggregate patch info via self-attention for the remaining
  blocks; extract them, reshape to (8,8,8,768), upsample 8^3 → 16^3 → 32^3 via
  3-D transposed convs → dense (32,32,32,n_classes+1) logits.
"""

import torch
import torch.nn as nn


def _build_4d_attn_mask(pixel_attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    (B, N) bool mask (True=valid) → (B, 1, 1, N) additive float mask
    (0.0 = attend, -inf = masked).  Shape broadcast-compatible with SigLIP attention.
    """
    B, N = pixel_attention_mask.shape
    mask = torch.zeros(B, 1, 1, N, dtype=dtype, device=pixel_attention_mask.device)
    mask = mask.masked_fill(~pixel_attention_mask.bool().view(B, 1, 1, N),
                            torch.finfo(dtype).min)
    return mask


class ViBlock5(nn.Module):
    """
    Full model: SigLIP backbone + injected voxel tokens + 3-D conv upsample.

    The SigLIP vision model is owned by this module so the whole thing can be
    wrapped as a single DDP module.  Identical to ViBlock4.
    """

    def __init__(
        self,
        vision,               # SiglipVisionModel (from HuggingFace)
        n_classes: int,       # excluding air; head outputs n_classes+1
        inject_at: int = 10,  # inject voxel tokens after this many encoder blocks
        vox_res:   int = 8,   # coarse voxel resolution; n_vox = vox_res^3 tokens
        feat_dim:  int = 768, # SigLIP hidden dim
        mid_dim:   int = 256, # conv upsample intermediate channels
    ):
        super().__init__()
        self.vision     = vision
        self.n_classes  = n_classes
        self.inject_at  = inject_at
        self.vox_res    = vox_res
        self.n_vox      = vox_res ** 3   # 512

        # Learned voxel slot embeddings — specialise through training signal alone.
        # Initialised small so early steps stay close to the SigLIP feature scale.
        self.vox_tokens = nn.Parameter(torch.randn(self.n_vox, feat_dim) * 0.02)

        # 3-D upsample: vox_res^3 → (2*vox_res)^3 → (4*vox_res)^3 = 32^3
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(feat_dim,    mid_dim * 2, kernel_size=2, stride=2),
            nn.GroupNorm(8, mid_dim * 2),
            nn.GELU(),
            nn.ConvTranspose3d(mid_dim * 2, mid_dim,     kernel_size=2, stride=2),
            nn.GroupNorm(8, mid_dim),
            nn.GELU(),
            nn.Conv3d(mid_dim, n_classes + 1, kernel_size=1),
        )

    def forward(
        self,
        pixel_values:         torch.Tensor,   # (B, N_patches, C*P*P) NaFlex padded
        pixel_attention_mask: torch.Tensor,   # (B, N_patches) bool, True=valid
        spatial_shapes:       torch.Tensor,   # (B, 2)
    ) -> torch.Tensor:
        """Returns (B, 32, 32, 32, n_classes+1)."""
        vision = self.vision
        B = pixel_attention_mask.shape[0]

        # ── Patch embeddings ─────────────────────────────────────────────────────
        hidden = vision.embeddings(pixel_values, spatial_shapes)

        # ── 4-D additive attention mask ──────────────────────────────────────────
        attn_mask = _build_4d_attn_mask(pixel_attention_mask, hidden.dtype)

        # ── First inject_at encoder blocks (patches only) ────────────────────────
        for layer in vision.encoder.layers[:self.inject_at]:
            hidden = layer(hidden, attn_mask)

        # ── Inject voxel slot tokens ─────────────────────────────────────────────
        vox = self.vox_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()
        hidden = torch.cat([hidden, vox], dim=1)  # (B, N_patches + N_vox, feat_dim)

        # Extend mask: voxel key-positions are all valid (0.0)
        attn_mask = torch.cat([
            attn_mask,
            attn_mask.new_zeros(B, 1, 1, self.n_vox),
        ], dim=-1)  # (B, 1, 1, N_patches + N_vox)

        # ── Remaining encoder blocks (patches + voxel tokens together) ───────────
        for layer in vision.encoder.layers[self.inject_at:]:
            hidden = layer(hidden, attn_mask)

        # ── Extract voxel tokens + post-layernorm ────────────────────────────────
        vox_out = hidden[:, -self.n_vox:, :]        # (B, N_vox, feat_dim)
        vox_out = vision.post_layernorm(vox_out)

        # ── 3-D conv upsample: 8^3 → 32^3 ───────────────────────────────────────
        r = self.vox_res
        vox_3d  = vox_out.reshape(B, r, r, r, -1)   # (B, 8, 8, 8, feat_dim)
        vox_3d  = vox_3d.permute(0, 4, 1, 2, 3)     # (B, feat_dim, 8, 8, 8)
        logits  = self.upsample(vox_3d)              # (B, n_classes+1, 32, 32, 32)
        logits  = logits.permute(0, 2, 3, 4, 1)     # (B, 32, 32, 32, n_classes+1)

        return logits
