from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


class SeparatorNet(nn.Module):
    """Mobile-friendly semantic segmenter returning separator logits."""

    def __init__(self, pretrained_backbone: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        backbone_weights = (
            MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        )
        self.network = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)["out"]


def create_model(config: dict, *, for_export: bool = False) -> SeparatorNet:
    model_config = config.get("model", {})
    return SeparatorNet(
        pretrained_backbone=(
            bool(model_config.get("pretrained_backbone", True)) and not for_export
        ),
        num_classes=int(model_config.get("num_classes", 2)),
    )

