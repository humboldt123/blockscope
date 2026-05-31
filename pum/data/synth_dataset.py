"""
synth_dataset.py — Synthetic voxel data for Stage 1 tokenizer training.

Generates random (32, 32, 32) long token grids matching TokenizerDataset format.
No Furnace or smelting required.

Three procedural layers per sample:
  1. Terrain  — fractal-noise heightmap → stone / dirt / surface layers
  2. Ore blobs — ellipsoidal blobs of random block types scattered through the grid
  3. Structures — hollow rectangular rooms / tunnels lined with random blocks

Block types are drawn uniformly from the full non-air vocabulary, so rare tokens
are over-represented relative to real data, compensating for class imbalance.

Each item is generated deterministically from its index so every epoch and every
DataLoader worker sees identical grids (reproducible, no random seed management).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# FSQ scale sizes — kept here to avoid circular imports with models/
_FSQ_SCALES = [1, 2, 4, 8]


def _fractal_heightmap(rng: np.random.Generator, size: int = 32) -> np.ndarray:
    """Return (size, size) int array of surface y-heights via sum-of-sines noise."""
    coords = np.arange(size, dtype=float)
    XX, ZZ = np.meshgrid(coords, coords, indexing="ij")
    h = np.zeros((size, size))
    for freq in [1, 2, 4, 8]:
        amp = 1.0 / freq
        px, pz = rng.uniform(0, 2 * np.pi, size=2)
        qx, qz = rng.uniform(0, 2 * np.pi, size=2)
        h += amp * (
            np.sin(2 * np.pi * freq * XX / size + px)
            * np.cos(2 * np.pi * freq * ZZ / size + pz)
            + 0.5
            * np.cos(2 * np.pi * freq * XX / size + qx)
            * np.sin(2 * np.pi * freq * ZZ / size + qz)
        )
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    return (h * 16 + 8).astype(int).clip(2, 29)  # surface y in [2, 29]


def _rand_tok(rng: np.random.Generator, num_tokens: int) -> int:
    return int(rng.integers(1, num_tokens))


def _add_terrain(
    grid: np.ndarray, rng: np.random.Generator, num_tokens: int
) -> np.ndarray:
    H = _fractal_heightmap(rng)           # (32, 32) surface heights
    stone_tok = _rand_tok(rng, num_tokens)
    dirt_tok  = _rand_tok(rng, num_tokens)
    surf_tok  = _rand_tok(rng, num_tokens)

    Y    = np.arange(32)[np.newaxis, :, np.newaxis]  # (1, 32, 1)
    Hexp = H[:, np.newaxis, :]                        # (32, 1, 32)

    grid = np.where(Y < (Hexp - 3), stone_tok, grid)
    grid = np.where((Y >= (Hexp - 3)) & (Y < Hexp), dirt_tok, grid)
    grid = np.where(Y == Hexp, surf_tok, grid)
    return grid


def _add_blobs(
    grid: np.ndarray, rng: np.random.Generator, num_tokens: int
) -> np.ndarray:
    n = int(rng.integers(3, 10))
    X, Y, Z = np.mgrid[0:32, 0:32, 0:32].astype(float)
    for _ in range(n):
        tok = _rand_tok(rng, num_tokens)
        cx, cz = rng.uniform(4, 28, size=2)
        cy     = rng.uniform(2, 20)
        rx, rz = rng.uniform(1.0, 3.5, size=2)
        ry     = rng.uniform(0.8, 2.5)
        mask = (
            ((X - cx) / rx) ** 2
            + ((Y - cy) / ry) ** 2
            + ((Z - cz) / rz) ** 2
        ) <= 1.0
        grid[mask] = tok
    return grid


def _add_structures(
    grid: np.ndarray, rng: np.random.Generator, num_tokens: int
) -> np.ndarray:
    n = int(rng.integers(0, 4))
    X, Y, Z = np.mgrid[0:32, 0:32, 0:32]
    for _ in range(n):
        wall_tok = _rand_tok(rng, num_tokens)
        w  = int(rng.integers(4, 13))
        hr = int(rng.integers(3, 9))
        d  = int(rng.integers(4, 13))
        ox = int(rng.integers(0, max(1, 33 - w)))
        oy = int(rng.integers(0, max(1, 33 - hr)))
        oz = int(rng.integers(0, max(1, 33 - d)))
        inside = (
            (X >= ox) & (X < ox + w)
            & (Y >= oy) & (Y < oy + hr)
            & (Z >= oz) & (Z < oz + d)
        )
        on_wall = inside & (
            (X == ox) | (X == ox + w - 1)
            | (Y == oy) | (Y == oy + hr - 1)
            | (Z == oz) | (Z == oz + d - 1)
        )
        grid[on_wall]          = wall_tok
        grid[inside & ~on_wall] = 0  # hollow interior
    return grid


class SynthDataset(Dataset):
    """
    Synthetic 32^3 voxel grids for tokenizer training augmentation.

    Each item is a (32, 32, 32) long tensor matching TokenizerDataset format.
    Generated deterministically from the item index for reproducibility.
    """

    def __init__(self, num_tokens: int, length: int = 50_000):
        self.num_tokens = num_tokens
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng  = np.random.default_rng(idx)
        grid = np.zeros((32, 32, 32), dtype=np.int32)
        grid = _add_terrain(grid, rng, self.num_tokens)
        grid = _add_blobs(grid, rng, self.num_tokens)
        grid = _add_structures(grid, rng, self.num_tokens)
        return torch.from_numpy(grid).long()
