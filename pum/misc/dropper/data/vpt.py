"""VPT contractor dataset: frame sequences + per-tick action labels.

Loads pre-extracted 224x224 frames from vpt_frames/ and parses per-tick
action labels from vpt_contractor/ JSONL files.

# NOTE: mod-collected dataset format is being guessed. When real mod data arrives,
# the loader will need to be updated. The expected format is similar to VPT but
# with per-slot inventory labels per tick (like inv_synth meta.jsonl). Design the
# interface so adding a ModDataset with the same __getitem__ signature is easy.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

from vocab import (
    ACTION_KEYS, camera_to_bin, NUM_BINARY_ACTIONS,
    NUM_MOUSE_BUTTONS, NUM_HOTBAR_SLOTS,
)

VPT_CONTRACTOR_DIR = Path("/data/vvm33/vpt_contractor")
VPT_FRAMES_DIR = Path("/data/vvm33/vpt_frames")

_TICK_CACHE = None


def _load_tick_cache() -> dict[str, int]:
    global _TICK_CACHE
    if _TICK_CACHE is None:
        cache_path = VPT_CONTRACTOR_DIR / "_tick_count_cache.json"
        with open(cache_path, encoding="utf-8", errors="replace") as f:
            _TICK_CACHE = json.load(f)
    return _TICK_CACHE


def _list_sessions(min_ticks: int = 100) -> list[str]:
    tick_cache = _load_tick_cache()
    sessions = []
    for name, count in tick_cache.items():
        if count >= min_ticks:
            sessions.append(name)
    sessions.sort()
    return sessions


def _parse_tick_actions(tick: dict) -> dict[str, torch.Tensor]:
    """Convert a single VPT JSONL tick to action tensors."""
    keys_held = set(tick.get("keyboard", {}).get("keys", []))
    binary = torch.zeros(NUM_BINARY_ACTIONS, dtype=torch.float)
    for i, key in enumerate(ACTION_KEYS):
        if key in keys_held:
            binary[i] = 1.0

    mouse = tick.get("mouse", {})
    mouse_buttons = torch.zeros(NUM_MOUSE_BUTTONS, dtype=torch.float)
    for b in mouse.get("buttons", []):
        if 0 <= b < NUM_MOUSE_BUTTONS:
            mouse_buttons[b] = 1.0

    dx = mouse.get("dx", 0.0)
    dy = mouse.get("dy", 0.0)
    camera_dx_bin = camera_to_bin(dx)
    camera_dy_bin = camera_to_bin(dy)

    hotbar = tick.get("hotbar", 0)

    return {
        "binary": binary,
        "mouse_buttons": mouse_buttons,
        "camera_dx": torch.tensor(camera_dx_bin, dtype=torch.long),
        "camera_dy": torch.tensor(camera_dy_bin, dtype=torch.long),
        "camera_dx_raw": torch.tensor(dx, dtype=torch.float),
        "camera_dy_raw": torch.tensor(dy, dtype=torch.float),
        "hotbar": torch.tensor(hotbar, dtype=torch.long),
    }


class VPTFrameSequenceDataset(Dataset):
    """Yields (frames, actions) sequences of length T from VPT sessions.

    Each sample is a contiguous window of T frames with corresponding actions.
    Windows are pre-computed as (session_idx, start_tick) pairs.
    """

    def __init__(
        self,
        seq_len: int = 16,
        stride: int = 8,
        split: str = "train",
        val_sessions: int = 100,
        min_ticks: int = 100,
        transform=None,
        load_mode: str = "full",
    ):
        self.seq_len = seq_len
        self.load_mode = load_mode  # "full" = all T frames, "last" = only last frame
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        all_sessions = _list_sessions(min_ticks)
        if split == "val":
            self.sessions = all_sessions[:val_sessions]
        elif split == "train":
            self.sessions = all_sessions[val_sessions:]
        else:
            self.sessions = all_sessions

        tick_cache = _load_tick_cache()
        self.windows = []
        for sess in self.sessions:
            n_ticks = tick_cache[sess]
            n_frames = min(n_ticks, len(list(Path(VPT_FRAMES_DIR / sess).glob("*.jpg"))) if (VPT_FRAMES_DIR / sess).exists() else 0)
            for start in range(0, n_frames - seq_len, stride):
                self.windows.append((sess, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sess, start = self.windows[idx]
        frames_dir = VPT_FRAMES_DIR / sess
        jsonl_path = VPT_CONTRACTOR_DIR / f"{sess}.jsonl"

        if self.load_mode == "last":
            # Baseline optimisation: only load the final frame.
            t = start + self.seq_len - 1
            img_path = frames_dir / f"{t:05d}.jpg"
            img = Image.open(img_path).convert("RGB")
            frames = torch.stack([self.transform(img)])  # (1, 3, 224, 224)
        else:
            frames = []
            for t in range(start, start + self.seq_len):
                img_path = frames_dir / f"{t:05d}.jpg"
                img = Image.open(img_path).convert("RGB")
                frames.append(self.transform(img))
            frames = torch.stack(frames)  # (T, 3, 224, 224)  # (T, 3, 224, 224)

        # Load action labels for this window
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            all_ticks = [json.loads(line) for line in f]

        actions_list = []
        for t in range(start, start + self.seq_len):
            if t < len(all_ticks):
                actions_list.append(_parse_tick_actions(all_ticks[t]))
            else:
                actions_list.append(_parse_tick_actions({}))

        # Stack action tensors
        actions = {
            "binary": torch.stack([a["binary"] for a in actions_list]),           # (T, 11)
            "mouse_buttons": torch.stack([a["mouse_buttons"] for a in actions_list]),  # (T, 2)
            "camera_dx": torch.stack([a["camera_dx"] for a in actions_list]),     # (T,)
            "camera_dy": torch.stack([a["camera_dy"] for a in actions_list]),     # (T,)
            "camera_dx_raw": torch.stack([a["camera_dx_raw"] for a in actions_list]),
            "camera_dy_raw": torch.stack([a["camera_dy_raw"] for a in actions_list]),
            "hotbar": torch.stack([a["hotbar"] for a in actions_list]),           # (T,)
        }

        return frames, actions


class VPTFrameSequenceDatasetFast(Dataset):
    """Faster version that pre-indexes frame counts using directory listing cache
    instead of globbing at init time.
    """

    def __init__(
        self,
        seq_len: int = 16,
        stride: int = 8,
        split: str = "train",
        val_sessions: int = 100,
        min_ticks: int = 100,
        transform=None,
        load_mode: str = "full",
    ):
        self.seq_len = seq_len
        self.load_mode = load_mode  # "full" = all T frames, "last" = only last frame
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        all_sessions = _list_sessions(min_ticks)
        if split == "val":
            self.sessions = all_sessions[:val_sessions]
        elif split == "train":
            self.sessions = all_sessions[val_sessions:]
        else:
            self.sessions = all_sessions

        tick_cache = _load_tick_cache()
        self.windows = []
        self._jsonl_cache = {}

        for sess in self.sessions:
            frames_dir = VPT_FRAMES_DIR / sess
            if not frames_dir.exists():
                continue
            # Use tick count as upper bound for frames, verify first/last frame exists
            n_ticks = tick_cache[sess]
            # Quick check: does last expected frame exist?
            while n_ticks > 0 and not (frames_dir / f"{n_ticks - 1:05d}.jpg").exists():
                n_ticks -= 1
            for start in range(0, n_ticks - seq_len, stride):
                self.windows.append((sess, start))

    def _load_jsonl(self, sess: str) -> list[dict]:
        if sess not in self._jsonl_cache:
            jsonl_path = VPT_CONTRACTOR_DIR / f"{sess}.jsonl"
            with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                self._jsonl_cache[sess] = [json.loads(line) for line in f]
            # Keep cache bounded
            if len(self._jsonl_cache) > 50:
                oldest = next(iter(self._jsonl_cache))
                del self._jsonl_cache[oldest]
        return self._jsonl_cache[sess]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sess, start = self.windows[idx]
        frames_dir = VPT_FRAMES_DIR / sess
        all_ticks = self._load_jsonl(sess)

        if self.load_mode == "last":
            # Baseline optimisation: only load the final frame.
            t = start + self.seq_len - 1
            img_path = frames_dir / f"{t:05d}.jpg"
            img = Image.open(img_path).convert("RGB")
            frames = torch.stack([self.transform(img)])  # (1, 3, 224, 224)
        else:
            frames = []
            for t in range(start, start + self.seq_len):
                img_path = frames_dir / f"{t:05d}.jpg"
                img = Image.open(img_path).convert("RGB")
                frames.append(self.transform(img))
            frames = torch.stack(frames)  # (T, 3, 224, 224)

        actions_list = []
        for t in range(start, start + self.seq_len):
            if t < len(all_ticks):
                actions_list.append(_parse_tick_actions(all_ticks[t]))
            else:
                actions_list.append(_parse_tick_actions({}))

        actions = {
            "binary": torch.stack([a["binary"] for a in actions_list]),
            "mouse_buttons": torch.stack([a["mouse_buttons"] for a in actions_list]),
            "camera_dx": torch.stack([a["camera_dx"] for a in actions_list]),
            "camera_dy": torch.stack([a["camera_dy"] for a in actions_list]),
            "camera_dx_raw": torch.stack([a["camera_dx_raw"] for a in actions_list]),
            "camera_dy_raw": torch.stack([a["camera_dy_raw"] for a in actions_list]),
            "hotbar": torch.stack([a["hotbar"] for a in actions_list]),
        }

        return frames, actions
