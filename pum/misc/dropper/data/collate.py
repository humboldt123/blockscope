"""Custom collate functions for variable-structure batches."""

import torch
from torch.utils.data import default_collate


def synth_collate(batch):
    """Collate for SynthInventoryDataset.

    Handles the slot_type_map dict (not stackable) by returning it as a list.
    Also handles raw PIL images (for SigLIP2 naFlex) by leaving them as a list.
    """
    first = batch[0]["pixel_values"]
    if isinstance(first, torch.Tensor):
        pixel_values = torch.stack([s["pixel_values"] for s in batch])
    else:
        # PIL Images or other non-tensor types (e.g. for naFlex processor)
        pixel_values = [s["pixel_values"] for s in batch]

    targets = torch.stack([s["target"] for s in batch])
    gui_types = [s["gui_type"] for s in batch]
    slot_type_maps = [s["slot_type_map"] for s in batch]
    filenames = [s["filename"] for s in batch]
    bboxes = [s["bbox"] for s in batch]

    return {
        "pixel_values": pixel_values,
        "target": targets,
        "gui_types": gui_types,
        "slot_type_maps": slot_type_maps,
        "filenames": filenames,
        "bboxes": bboxes,
    }


def vpt_collate(batch):
    """Collate for VPT frame sequence datasets.

    batch: list of (frames_tensor, actions_dict) tuples.
    """
    frames = torch.stack([b[0] for b in batch])  # (B, T, 3, 224, 224)
    action_keys = batch[0][1].keys()
    actions = {}
    for key in action_keys:
        actions[key] = torch.stack([b[1][key] for b in batch])
    return frames, actions
