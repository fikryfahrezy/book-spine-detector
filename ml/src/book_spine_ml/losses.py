from __future__ import annotations

import torch
import torch.nn.functional as F


def separator_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    separator_class_weight: float = 8.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = torch.tensor(
        [1.0, separator_class_weight], device=logits.device, dtype=logits.dtype
    )
    cross_entropy = F.cross_entropy(logits, targets, weight=weights)

    probabilities = logits.softmax(dim=1)[:, 1]
    target_float = (targets == 1).to(probabilities.dtype)
    intersection = (probabilities * target_float).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - dice.mean()

    total = cross_entropy + dice_weight * dice_loss
    return total, {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "dice_loss": float(dice_loss.detach().cpu()),
    }

