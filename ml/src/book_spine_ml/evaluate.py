from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import SeparatorDataset
from .metrics import binary_segmentation_metrics
from .model import create_model
from .postprocess import count_from_probability
from .runtime import resolve_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    device_name: str | None = None,
) -> dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    device = resolve_device(device_name)
    model = create_model(config, for_export=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    data_config = config["data"]
    dataset = SeparatorDataset(
        data_config["root"], split, int(data_config.get("image_size", 640)), augment=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    postprocess_options = config.get("postprocess", {})

    dice_total = 0.0
    iou_total = 0.0
    absolute_count_error = 0.0
    exact_counts = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"evaluate {split}"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(image)
            metrics = binary_segmentation_metrics(logits, mask)
            dice_total += metrics["dice"]
            iou_total += metrics["iou"]

            predicted_probability = logits.softmax(dim=1)[0, 1].cpu().numpy()
            target_probability = mask[0].float().cpu().numpy()
            predicted_count = count_from_probability(
                predicted_probability, **postprocess_options
            ).count
            target_count = count_from_probability(
                target_probability, **postprocess_options
            ).count
            error = abs(predicted_count - target_count)
            absolute_count_error += error
            exact_counts += int(error == 0)

    count = len(dataset)
    return {
        "samples": count,
        "dice": dice_total / count,
        "iou": iou_total / count,
        "count_mae": absolute_count_error / count,
        "exact_count_accuracy": exact_counts / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained separator model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_checkpoint(args.checkpoint, args.split, args.device), indent=2
        )
    )


if __name__ == "__main__":
    main()

