"""
tok_dataset.py — Stage 1 tokenizer dataset.

Yields (32, 32, 32) tensors of compact token IDs from smelted b arrays.
On first access to an unsmelted session, triggers b-only smelt via smelt.py.

Token mapping: raw block-state IDs (uint16) → compact token IDs (0..V-1)
via stage1_vocab.json. Unknown/air → 0.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", "/data/vvm33/BLOCKSCOPE_DATA"))


def get_session_dirs(raw_root: Path, split: str, n_train: int = 73) -> list[Path]:
    """
    Return sorted session dirs for the requested split.
    Sessions sorted by name (timestamp-encoded). First n_train = train, rest = test.
    """
    dirs = sorted(Path(raw_root).glob("session_*"))
    if not dirs:
        raise FileNotFoundError(f"No sessions found in {raw_root}")
    if split == "train":
        return dirs[:n_train]
    elif split == "test":
        return dirs[n_train:]
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


class TokenizerDataset(Dataset):
    """
    Each item is a (32, 32, 32) long tensor of token IDs.

    Loads b arrays lazily per session (smelt on first access).
    Caches loaded arrays in memory.
    """

    def __init__(
        self,
        session_dirs: list[Path],
        state_to_token: dict[int, int],
        num_tokens: int,
    ):
        self.session_dirs = [Path(d) for d in session_dirs]
        self.state_to_token = state_to_token
        self.num_tokens = num_tokens

        # Build index: list of (session_idx, tick_idx)
        self._index: list[tuple[int, int]] = []
        self._tick_counts: list[int] = []

        log.info("Counting ticks across %d sessions ...", len(self.session_dirs))
        from utils.smelt import session_tick_count
        for i, d in enumerate(self.session_dirs):
            n = session_tick_count(d)
            self._tick_counts.append(n)
            for t in range(n):
                self._index.append((i, t))
        log.info("Dataset: %d total ticks across %d sessions",
                 len(self._index), len(self.session_dirs))

        # Lazy per-session cache: session_idx → np.ndarray (T, 32, 32, 32) token IDs
        self._cache: dict[int, np.ndarray] = {}

    def _load_session(self, session_idx: int) -> np.ndarray:
        if session_idx in self._cache:
            return self._cache[session_idx]

        from utils.smelt import load_b_arrays
        session_dir = self.session_dirs[session_idx]
        log.info("Loading session %s (idx=%d)", session_dir.name, session_idx)

        px, py, pz, blocks_u16 = load_b_arrays(session_dir)
        # blocks_u16: uint16 (T, 32, 32, 32) raw block-state IDs
        tokens = self._map_tokens(blocks_u16)  # int32 (T, 32, 32, 32)
        self._cache[session_idx] = tokens
        return tokens

    def _map_tokens(self, blocks_u16: np.ndarray) -> np.ndarray:
        """Map uint16 block-state IDs → compact token IDs."""
        # Vectorised: build lookup table up to max state ID
        max_id = int(blocks_u16.max())
        lut = np.zeros(max_id + 1, dtype=np.int32)
        for sid in range(max_id + 1):
            lut[sid] = self.state_to_token.get(sid, 0)
        return lut[blocks_u16].astype(np.int32)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        session_idx, tick_idx = self._index[idx]
        tokens = self._load_session(session_idx)
        grid = tokens[tick_idx]  # (32, 32, 32) int32
        return torch.from_numpy(grid).long()
