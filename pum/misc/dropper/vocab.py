import json
from pathlib import Path
from typing import Optional

DATA_ROOT = Path("/data/vvm33")
VOCAB_PATH = DATA_ROOT / "_vocab.json"
NAME_MAP_PATH = DATA_ROOT / "_name_map.json"

_vocab: Optional[dict[str, int]] = None
_id_to_name: Optional[dict[int, str]] = None
_name_map: Optional[dict[str, str]] = None


def load_vocab() -> dict[str, int]:
    global _vocab
    if _vocab is None:
        with open(VOCAB_PATH) as f:
            _vocab = json.load(f)
    return _vocab


def load_name_map() -> dict[str, str]:
    global _name_map
    if _name_map is None:
        with open(NAME_MAP_PATH) as f:
            _name_map = json.load(f)
    return _name_map


def name_to_id(name: str) -> int:
    vocab = load_vocab()
    nm = load_name_map()
    canonical = nm.get(name, name)
    return vocab.get(canonical, vocab.get(name, 0))


def id_to_name(item_id: int) -> str:
    global _id_to_name
    if _id_to_name is None:
        vocab = load_vocab()
        _id_to_name = {v: k for k, v in vocab.items()}
    return _id_to_name.get(item_id, "unknown")


def num_classes() -> int:
    return len(load_vocab())


AIR_ID = 0

# Action space constants derived from VPT contractor data.
# 11 binary keyboard actions tracked as held/not-held per tick.
ACTION_KEYS = [
    "key.keyboard.w",
    "key.keyboard.a",
    "key.keyboard.s",
    "key.keyboard.d",
    "key.keyboard.space",
    "key.keyboard.left.shift",
    "key.keyboard.e",
    "key.keyboard.q",
    "key.keyboard.f",
    "key.keyboard.1",
    "key.keyboard.2",
]

ACTION_KEY_NAMES = [
    "forward", "left", "back", "right", "jump", "sneak",
    "inventory", "drop", "swap_hands", "hotbar_1", "hotbar_2",
]

NUM_BINARY_ACTIONS = len(ACTION_KEYS)

# Mouse buttons: 0 = attack/left, 1 = use/right
NUM_MOUSE_BUTTONS = 2

# Camera: dx, dy as continuous values, discretized into bins for prediction.
CAMERA_BINS = 121  # -60 to +60 in 1-unit bins
CAMERA_BIN_MIN = -60
CAMERA_BIN_MAX = 60

# Hotbar: 10 discrete slots (0-9)
NUM_HOTBAR_SLOTS = 10


def camera_to_bin(value: float) -> int:
    clamped = max(CAMERA_BIN_MIN, min(CAMERA_BIN_MAX, value))
    return int(clamped - CAMERA_BIN_MIN)


def bin_to_camera(b: int) -> float:
    return float(b + CAMERA_BIN_MIN)
