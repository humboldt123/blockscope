"""
ViBlock4 — SigLIP with injected voxel tokens + 3-D conv upsample.

Instead of a separate cross-attention decoder that reads the final SigLIP patch
features, we inject 8^3 = 512 learned voxel slot tokens directly into the SigLIP
encoder at layer `inject_at`.  The voxel tokens participate in the SigLIP
self-attention for the remaining blocks, aggregating patch information through the
backbone's own attention mechanism — no separate decoder needed.

After the last block + post_layernorm, the 512 voxel tokens are extracted,
reshaped to (8, 8, 8, 768), and upsampled via 3-D transposed convolutions:
  (8, 8, 8) → (16, 16, 16) → (32, 32, 32)
producing dense (32, 32, 32, n_classes+1) logits.

No explicit 3-D position embeddings for voxel tokens — they are learned slot
embeddings that specialise during training.  The 3-D spatial structure is
recovered at the end by the conv upsample, which also lets nearby voxels share
context (important for predicting spatially coherent walls / floors).

Compare to viblock3:
  viblock3: SigLIP → patch memory (22×40) → separate cross-attn decoder → 32^3
  viblock4: SigLIP (with voxel tokens injected inside) → 8^3 tokens → conv → 32^3

The depth supervision signal (3-D voxel grid GT) teaches the model which voxel
slot to activate for which depth — no explicit depth estimator needed.
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


class ViBlock4(nn.Module):
    """
    Full model: SigLIP backbone + injected voxel tokens + 3-D conv upsample.

    The SigLIP vision model is owned by this module so the whole thing can be
    wrapped as a single DDP module.
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
        # Siglip2VisionEmbeddings.forward() expects padded (B, N_max, C*P*P) and
        # returns (B, N_patches, feat_dim) — always 3D with NaFlex padded input.
        hidden = vision.embeddings(pixel_values, spatial_shapes)

        # ── 4-D additive attention mask ──────────────────────────────────────────
        # (B, 1, 1, N_patches): 0.0 = attend, -inf = masked.
        # Broadcasts over (B, n_heads, seq_len, seq_len) attention weights.
        attn_mask = _build_4d_attn_mask(pixel_attention_mask, hidden.dtype)

        # ── First inject_at encoder blocks (patches only) ────────────────────────
        # Siglip2EncoderLayer.forward() returns a plain tensor, not a tuple.
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
