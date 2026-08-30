from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import IMAGE_EXTENSIONS


def parse_yolo_shape(line: str, width: int, height: int) -> list[tuple[float, float]]:
    values = [float(value) for value in line.strip().split()]
    if len(values) < 5:
        raise ValueError(f"Unsupported YOLO annotation: {line!r}")
    coordinates = values[1:]

    if len(coordinates) == 4:
        center_x, center_y, box_width, box_height = coordinates
        left = (center_x - box_width / 2) * width
        right = (center_x + box_width / 2) * width
        top = (center_y - box_height / 2) * height
        bottom = (center_y + box_height / 2) * height
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    if len(coordinates) % 2:
        raise ValueError(f"Expected coordinate pairs: {line!r}")
    return [
        (coordinates[index] * width, coordinates[index + 1] * height)
        for index in range(0, len(coordinates), 2)
    ]


def longest_boundary_edges(
    polygon: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(polygon) < 4:
        return []
    points = np.asarray(polygon, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    major_axis, minor_axis = vectors[0], vectors[1]
    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis

    major_min, major_max = major_projection.min(), major_projection.max()
    minor_min, minor_max = minor_projection.min(), minor_projection.max()

    def point(major: float, minor: float) -> tuple[float, float]:
        value = center + major * major_axis + minor * minor_axis
        return float(value[0]), float(value[1])

    return [
        (point(major_min, minor_min), point(major_max, minor_min)),
        (point(major_min, minor_max), point(major_max, minor_max)),
    ]


def convert_annotation(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_mask_path: Path,
    *,
    line_width: int,
) -> int:
    image = Image.open(image_path).convert("RGB")
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    instances = 0
    effective_line_width = max(
        1, round(line_width * max(image.size) / 640)
    )

    if label_path.exists():
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            polygon = parse_yolo_shape(raw_line, *image.size)
            for start, end in longest_boundary_edges(polygon):
                draw.line([start, end], fill=255, width=effective_line_width)
            instances += 1

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_image_path)
    mask.save(output_mask_path)
    return instances


def prepare_yolo_dataset(
    source: str | Path,
    destination: str | Path,
    *,
    line_width: int = 5,
    prefix: str = "",
) -> dict[str, int]:
    source = Path(source)
    destination = Path(destination)
    summary: dict[str, int] = {}

    for split in ("train", "valid", "val", "test"):
        source_split = source / split
        if not source_split.exists():
            continue
        destination_split = "val" if split in {"valid", "val"} else split
        image_dir = source_split / "images"
        label_dir = source_split / "labels"
        if not image_dir.exists():
            continue

        converted = 0
        total_instances = 0
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            output_name = f"{prefix}{image_path.name}"
            output_stem = Path(output_name).stem
            total_instances += convert_annotation(
                image_path,
                label_path,
                destination / destination_split / "images" / output_name,
                destination / destination_split / "masks" / f"{output_stem}.png",
                line_width=line_width,
            )
            converted += 1
        summary[f"{destination_split}_images"] = converted
        summary[f"{destination_split}_instances"] = total_instances

    if not summary:
        raise FileNotFoundError(
            f"No YOLO split folders found in {source}. Expected train/images and train/labels."
        )

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO detection/OBB/segmentation export to separator masks."
    )
    parser.add_argument("--source", required=True, help="Roboflow/YOLO export directory")
    parser.add_argument("--destination", required=True, help="Processed dataset directory")
    parser.add_argument("--line-width", type=int, default=5)
    parser.add_argument(
        "--prefix", default="", help="Prefix output filenames when merging multiple sources"
    )
    args = parser.parse_args()
    summary = prepare_yolo_dataset(
        args.source,
        args.destination,
        line_width=args.line_width,
        prefix=args.prefix,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
