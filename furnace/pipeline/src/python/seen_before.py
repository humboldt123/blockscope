"""
seen_before.py — post-pass that computes s[32,32,32] per tick.

s[i,j,k] at tick t = 1 if the absolute world coord for cell (i,j,k)
was visible (m==1) at some earlier tick t' < t AND held the same block-state.

Implementation walks ticks chronologically using numpy vectorisation:
  - The "seen" state is a flat dict keyed by encoded (wx,wy,wz).
  - Each tick we check cells against the dict in a vectorised batch,
    then update the dict with visible cells.

Memory: ~100k entries for a 10-min session — fine in RAM.
Flag for revisit if sessions grow beyond 30 min.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

WINDOW = 32


def compute_seen_before(
    px_arr: np.ndarray,   # int32[T]
    py_arr: np.ndarray,
    pz_arr: np.ndarray,
    m_all: np.ndarray,    # uint8[T,32,32,32]
    b_all: np.ndarray,    # int32[T,32,32,32]
) -> np.ndarray:
    """
    Returns s_all: uint8[T,32,32,32].
    Tick 0 is guaranteed all-zero (nothing seen before).
    """
    T = m_all.shape[0]
    s_all = np.zeros((T, WINDOW, WINDOW, WINDOW), dtype=np.uint8)

    # Cell offset grid within the window: shape (32,32,32,3)
    idx = np.indices((WINDOW, WINDOW, WINDOW), dtype=np.int32)  # (3,32,32,32)
    idx = np.moveaxis(idx, 0, -1)                                 # (32,32,32,3)
    idx_flat = idx.reshape(-1, 3)                                  # (32768,3)

    # "seen" dict: encoded_world_coord (int64) -> block_state_id (int32)
    # Encoding: (wx+32768)*65536*65536 + (wy+32768)*65536 + (wz+32768)
    # This packs 3 signed 16-bit coords into a 48-bit int64 (unique for any
    # world that fits in a 32768-block radius, which covers all vanilla worlds).
    seen_key   = np.empty(0, dtype=np.int64)
    seen_state = np.empty(0, dtype=np.int32)

    def encode(wx, wy, wz):
        return ((wx + 32768).astype(np.int64) * (65536 * 65536)
              + (wy + 32768).astype(np.int64) * 65536
              + (wz + 32768).astype(np.int64))

    for t in range(T):
        ox = int(px_arr[t]) - 15
        oy = int(py_arr[t]) - 15
        oz = int(pz_arr[t]) - 15

        # World coords of every cell in the window
        wx = idx_flat[:, 0] + ox   # (32768,)
        wy = idx_flat[:, 1] + oy
        wz = idx_flat[:, 2] + oz
        keys = encode(wx, wy, wz)  # (32768,)

        b_flat = b_all[t].reshape(-1).astype(np.int32)  # (32768,)
        m_flat = m_all[t].reshape(-1)                    # (32768,)

        # Check which cells are in seen_key with matching state
        if len(seen_key) > 0:
            # Sort-based lookup (faster than Python dict for large arrays)
            sort_idx = np.searchsorted(seen_key, keys)
            sort_idx = np.clip(sort_idx, 0, len(seen_key) - 1)
            match_key   = seen_key[sort_idx]   == keys
            match_state = seen_state[sort_idx] == b_flat
            s_mask = match_key & match_state
            s_all[t] = s_mask.reshape(WINDOW, WINDOW, WINDOW).astype(np.uint8)

        # Update seen with visible cells at tick t
        vis = np.where(m_flat)[0]
        if len(vis) > 0:
            new_keys   = keys[vis]
            new_states = b_flat[vis]

            if len(seen_key) == 0:
                seen_key   = new_keys
                seen_state = new_states
                order      = np.argsort(seen_key)
                seen_key   = seen_key[order]
                seen_state = seen_state[order]
            else:
                # Merge: update existing entries, insert new ones
                all_keys   = np.concatenate([seen_key,   new_keys])
                all_states = np.concatenate([seen_state, new_states])
                # Keep last occurrence (new values override old)
                order      = np.argsort(all_keys, kind="stable")
                all_keys   = all_keys[order]
                all_states = all_states[order]
                # Deduplicate keeping last (rightmost after stable sort = new)
                keep = np.concatenate([
                    all_keys[:-1] != all_keys[1:],
                    np.array([True])
                ])
                seen_key   = all_keys[keep]
                seen_state = all_states[keep]

        if t % 200 == 0:
            log.info("seen_before: tick %d/%d  dict_size=%d", t, T, len(seen_key))

    log.info("seen_before done. %d unique world coords tracked", len(seen_key))
    return s_all
