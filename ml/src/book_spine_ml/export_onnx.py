from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import create_model


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = create_model(config, for_export=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(config["data"].get("image_size", 640))
    dummy_input = torch.randn(1, 3, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["separator_logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "separator_logits": {0: "batch"}},
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a separator checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()

