from __future__ import annotations

import torch


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return numerator / (denominator + eps)


def compute_binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    fov_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute binary segmentation metrics.

    Args:
        prediction: Probability tensor or binary tensor shaped [B, 1, H, W].
        target: Binary mask tensor shaped [B, 1, H, W].
        threshold: Threshold used when prediction is not already binary.
        fov_mask: Optional binary FOV mask. Metrics are computed only where it is 1.
    """
    with torch.no_grad():
        pred = (prediction >= threshold).bool()
        gt = (target >= 0.5).bool()

        if fov_mask is not None:
            valid = (fov_mask >= 0.5).bool()
            pred = pred[valid]
            gt = gt[valid]
        else:
            pred = pred.reshape(-1)
            gt = gt.reshape(-1)

        pred = pred.bool()
        gt = gt.bool()

        tp = torch.sum(pred & gt).float()
        tn = torch.sum(~pred & ~gt).float()
        fp = torch.sum(pred & ~gt).float()
        fn = torch.sum(~pred & gt).float()

        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        iou = _safe_div(tp, tp + fp + fn)
        accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
        sensitivity = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        precision = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

        return {
            "dice": float(dice.item()),
            "iou": float(iou.item()),
            "accuracy": float(accuracy.item()),
            "sensitivity": float(sensitivity.item()),
            "specificity": float(specificity.item()),
            "precision": float(precision.item()),
            "f1": float(f1.item()),
        }
