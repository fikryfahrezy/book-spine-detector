from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import image_to_tensor, letterbox_image
from .model import create_model
from .postprocess import CountResult, Orientation, count_from_probability
from .runtime import resolve_device


@dataclass(frozen=True)
class Prediction:
    result: CountResult
    probability: np.ndarray
    overlay: Image.Image


class InferenceEngine:
    def __init__(self, checkpoint_path: str | Path, device_name: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device_name)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.config = checkpoint["config"]
        self.model = create_model(self.config, for_export=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.image_size = int(self.config["data"].get("image_size", 640))

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        *,
        orientation: Orientation = "auto",
        pixel_threshold: float | None = None,
        peak_height: float | None = None,
        peak_prominence: float | None = None,
        minimum_peak_distance: int | None = None,
    ) -> Prediction:
        original = image.convert("RGB")
        prepared, metadata = letterbox_image(
            original, self.image_size, fill=(0, 0, 0)
        )
        tensor = image_to_tensor(prepared).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probability = logits.softmax(dim=1)[0, 1].detach().cpu().numpy()
        crop = probability[
            metadata.pad_top : metadata.pad_top + metadata.resized_height,
            metadata.pad_left : metadata.pad_left + metadata.resized_width,
        ]

        options = self.config.get("postprocess", {})
        result = count_from_probability(
            crop,
            orientation=orientation,
            pixel_threshold=float(
                options.get("pixel_threshold", 0.45)
                if pixel_threshold is None
                else pixel_threshold
            ),
            peak_height=float(
                options.get("peak_height", 0.30) if peak_height is None else peak_height
            ),
            peak_prominence=float(
                options.get("peak_prominence", 0.08)
                if peak_prominence is None
                else peak_prominence
            ),
            smoothing_window=int(options.get("smoothing_window", 9)),
            minimum_peak_distance=int(
                options.get("minimum_peak_distance", 8)
                if minimum_peak_distance is None
                else minimum_peak_distance
            ),
        )
        overlay = draw_count_overlay(original, result, metadata.scale)
        return Prediction(result=result, probability=crop, overlay=overlay)


def draw_count_overlay(
    image: Image.Image, result: CountResult, scale: float
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    color = (20, 220, 90)

    for index, position in enumerate(result.boundary_positions, start=1):
        original_position = int(round(position / scale))
        if result.orientation == "vertical":
            draw.line([(original_position, 0), (original_position, height)], fill=color, width=3)
            draw.text((original_position + 4, 6), str(index), fill=color)
        else:
            draw.line([(0, original_position), (width, original_position)], fill=color, width=3)
            draw.text((6, original_position + 4), str(index), fill=color)

    label = f"Count: {result.count}  Confidence: {result.confidence:.2f}"
    draw.rectangle([(8, 8), (370, 42)], fill=(0, 0, 0))
    draw.text((16, 16), label, fill=(255, 255, 255))
    return overlay

