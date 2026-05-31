"""Reusable wandb logging helpers for the dropper project.

Every experiment imports from here so visualization is consistent.
"""

import torch
import numpy as np
import wandb
from PIL import Image
from typing import Optional

from vocab import id_to_name, ACTION_KEY_NAMES


def log_inventory_predictions(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    n: int = 8,
    prefix: str = "val",
):
    """Run model on n images, log a wandb.Table with predictions vs ground truth.

    Args:
        model: Model with forward that returns per-patch logits (B, num_patches, num_classes).
        images: (B, C, H, W) tensor, already preprocessed for SigLIP.
        labels: (B, num_patches) tensor with item class IDs, -100 for ignored patches.
        step: Current training step.
        n: Number of samples to log.
        prefix: Table name prefix.
    """
    model.eval()
    device = next(model.parameters()).device
    imgs = images[:n].to(device)
    labs = labels[:n].to(device)

    with torch.no_grad():
        logits = model(imgs)
        preds = logits.argmax(dim=-1)

    table = wandb.Table(columns=["image", "predicted_items", "ground_truth_items", "correct"])

    for i in range(min(n, imgs.shape[0])):
        img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
        wimg = wandb.Image(Image.fromarray(img_np))

        mask = labs[i] != -100
        gt_ids = labs[i][mask].cpu().tolist()
        pred_ids = preds[i][mask].cpu().tolist()

        gt_names = [id_to_name(g) for g in gt_ids if g != 0]
        pred_names = [id_to_name(p) for p in pred_ids if p != 0]

        correct = sum(1 for g, p in zip(gt_ids, pred_ids) if g == p)
        total = len(gt_ids)
        acc_str = f"{correct}/{total}" if total > 0 else "n/a"

        table.add_data(wimg, ", ".join(pred_names[:20]), ", ".join(gt_names[:20]), acc_str)

    wandb.log({f"{prefix}/inventory_predictions": table}, step=step)
    model.train()


def log_action_predictions(
    model,
    frames: torch.Tensor,
    actions: dict[str, torch.Tensor],
    step: int,
    n: int = 8,
    prefix: str = "val",
):
    """Log predicted vs ground truth actions as a wandb.Table.

    Args:
        model: Policy model that returns action dict.
        frames: (B, T, C, H, W) tensor of frame sequences.
        actions: Dict with keys "binary" (B, T, 11), "camera_dx" (B, T), "camera_dy" (B, T).
        step: Current training step.
        n: Number of sequences to log.
        prefix: Table name prefix.
    """
    model.eval()
    device = next(model.parameters()).device
    batch_frames = frames[:n].to(device)

    with torch.no_grad():
        pred = model(batch_frames)

    columns = ["frame_idx"] + [f"gt_{k}" for k in ACTION_KEY_NAMES] + [f"pred_{k}" for k in ACTION_KEY_NAMES] + ["gt_camera", "pred_camera"]
    table = wandb.Table(columns=columns)

    binary_gt = actions["binary"][:n].cpu()  # (B, T, 11) or (B, 11)
    binary_pred = (pred["binary_logits"][:n].cpu().sigmoid() > 0.5).int()  # (B, 11)
    cam_dx_gt = actions["camera_dx"][:n].cpu()  # (B, T) or (B,)
    cam_dy_gt = actions["camera_dy"][:n].cpu()
    cam_dx_pred = pred["camera_dx_logits"][:n].cpu().argmax(dim=-1)  # (B,)
    cam_dy_pred = pred["camera_dy_logits"][:n].cpu().argmax(dim=-1)

    # Ground truth may have temporal dim; predictions are always for the last frame.
    gt_has_time = binary_gt.dim() == 3

    for b in range(min(n, batch_frames.shape[0])):
        gt_row = binary_gt[b, -1].tolist() if gt_has_time else binary_gt[b].tolist()
        pred_row = binary_pred[b].tolist()
        gt_cam = f"({cam_dx_gt[b, -1]:.0f}, {cam_dy_gt[b, -1]:.0f})" if gt_has_time else f"({cam_dx_gt[b]:.0f}, {cam_dy_gt[b]:.0f})"
        pred_cam = f"({cam_dx_pred[b]:.0f}, {cam_dy_pred[b]:.0f})"
        table.add_data(b, *gt_row, *pred_row, gt_cam, pred_cam)

    wandb.log({f"{prefix}/action_predictions": table}, step=step)
    model.train()


def log_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    step: int,
    title: str = "item_confusion",
):
    """Log a wandb confusion matrix for item classification.

    Args:
        preds: (N,) predicted class indices.
        labels: (N,) ground truth class indices.
        class_names: List of class name strings.
        step: Current training step.
        title: Name for the logged artifact.
    """
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Only include classes that actually appear
    present = np.unique(np.concatenate([preds_np, labels_np]))
    present_names = [class_names[i] if i < len(class_names) else str(i) for i in present]

    wandb.log({
        f"{title}": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels_np.tolist(),
            preds=preds_np.tolist(),
            class_names=class_names,
            title=title,
        )
    }, step=step)


def log_attention_maps(
    model,
    image: torch.Tensor,
    step: int,
    layer_idx: int = -1,
    prefix: str = "attention",
):
    """Extract and log SigLIP attention maps overlaid on input image.

    Args:
        model: Model with a .encoder attribute that is a SigLIP wrapper.
        image: (C, H, W) single image tensor, preprocessed.
        step: Current training step.
        layer_idx: Which transformer layer's attention to extract (-1 = last).
        prefix: wandb log key prefix.
    """
    model.eval()
    device = next(model.parameters()).device
    img = image.unsqueeze(0).to(device)

    attn_weights = []
    hooks = []

    def hook_fn(module, input, output):
        if hasattr(output, 'attn_weights') and output.attn_weights is not None:
            attn_weights.append(output.attn_weights)
        elif isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights.append(output[1])

    encoder = model.encoder if hasattr(model, 'encoder') else model
    layers = None
    if hasattr(encoder, 'model'):
        vision = encoder.model
        if hasattr(vision, 'vision_model'):
            layers = vision.vision_model.encoder.layers
        elif hasattr(vision, 'encoder'):
            layers = vision.encoder.layers

    if layers is not None:
        target_layer = layers[layer_idx]
        if hasattr(target_layer, 'self_attn'):
            h = target_layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(h)

    with torch.no_grad():
        _ = encoder(img) if not hasattr(model, 'encoder') else model.encoder(img)

    for h in hooks:
        h.remove()

    if not attn_weights:
        model.train()
        return

    attn = attn_weights[0]
    if attn.dim() == 4:
        attn = attn.mean(dim=1)  # average over heads: (1, N, N)
    attn_map = attn[0].cpu().numpy()

    # CLS token attention over patches (if CLS exists), else mean attention
    if attn_map.shape[0] > 196:
        spatial_attn = attn_map[0, 1:]  # CLS attending to patches
    else:
        spatial_attn = attn_map.mean(axis=0)

    if len(spatial_attn) >= 196:
        spatial_attn = spatial_attn[:196]
    side = int(np.sqrt(len(spatial_attn)))
    attn_2d = spatial_attn.reshape(side, side)

    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    attn_resized = np.array(Image.fromarray((attn_2d * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR))

    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)

    # Overlay: blend attention heatmap with image
    import cv2
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_np.astype(float) + 0.4 * heatmap.astype(float)).clip(0, 255).astype(np.uint8)

    wandb.log({
        f"{prefix}/attention_overlay": wandb.Image(Image.fromarray(overlay), caption="Last-layer attention"),
        f"{prefix}/attention_raw": wandb.Image(Image.fromarray(attn_resized), caption="Raw attention map"),
    }, step=step)

    model.train()


def log_per_action_accuracy(
    binary_preds: torch.Tensor,
    binary_labels: torch.Tensor,
    step: int,
    prefix: str = "val",
):
    """Log per-action precision, recall, F1 as a wandb bar chart.

    Args:
        binary_preds: (N, 11) binary predictions.
        binary_labels: (N, 11) binary ground truth.
        step: Current training step.
        prefix: wandb log key prefix.
    """
    preds = binary_preds.cpu().float()
    labels = binary_labels.cpu().float()

    metrics = {}
    for i, name in enumerate(ACTION_KEY_NAMES):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum().float()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum().float()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum().float()

        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f"{prefix}/precision_{name}"] = precision
        metrics[f"{prefix}/recall_{name}"] = recall
        metrics[f"{prefix}/f1_{name}"] = f1

    wandb.log(metrics, step=step)
    return metrics


def log_camera_histogram(
    pred_dx: torch.Tensor,
    pred_dy: torch.Tensor,
    gt_dx: torch.Tensor,
    gt_dy: torch.Tensor,
    step: int,
    prefix: str = "val",
):
    """Log 2D histogram of predicted vs ground truth camera deltas."""
    data = [[float(px), float(py), float(gx), float(gy)]
            for px, py, gx, gy in zip(
                pred_dx.cpu().flatten()[:2000],
                pred_dy.cpu().flatten()[:2000],
                gt_dx.cpu().flatten()[:2000],
                gt_dy.cpu().flatten()[:2000],
            )]
    table = wandb.Table(data=data, columns=["pred_dx", "pred_dy", "gt_dx", "gt_dy"])
    wandb.log({
        f"{prefix}/camera_pred_scatter": wandb.plot.scatter(table, "pred_dx", "pred_dy", title="Predicted Camera"),
        f"{prefix}/camera_gt_scatter": wandb.plot.scatter(table, "gt_dx", "gt_dy", title="Ground Truth Camera"),
    }, step=step)


def log_gradient_norms(named_params, step: int, prefix: str = "grad"):
    """Log gradient norms per named component for detecting gradient conflicts."""
    component_norms = {}
    for name, param in named_params:
        if param.grad is not None:
            component = name.split(".")[0]
            norm = param.grad.data.norm(2).item()
            if component not in component_norms:
                component_norms[component] = []
            component_norms[component].append(norm ** 2)

    metrics = {}
    for component, norms in component_norms.items():
        metrics[f"{prefix}/{component}_grad_norm"] = np.sqrt(sum(norms))
    wandb.log(metrics, step=step)
