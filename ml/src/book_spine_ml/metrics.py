from __future__ import annotations

import torch


def binary_segmentation_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    predictions = logits.argmax(dim=1) == 1
    truth = targets == 1

    intersection = (predictions & truth).sum().float()
    union = (predictions | truth).sum().float()
    predicted_count = predictions.sum().float()
    truth_count = truth.sum().float()

    dice = (2 * intersection + 1) / (predicted_count + truth_count + 1)
    iou = (intersection + 1) / (union + 1)

    return {"dice": float(dice.cpu()), "iou": float(iou.cpu())}

