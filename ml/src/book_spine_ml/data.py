from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LetterboxMetadata:
    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int


def letterbox_image(
    image: Image.Image,
    size: int,
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
    fill: int | tuple[int, int, int] = 0,
) -> tuple[Image.Image, LetterboxMetadata]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = image.resize((resized_width, resized_height), resample=resample)

    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    canvas = Image.new(image.mode, (size, size), color=fill)
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, LetterboxMetadata(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class SeparatorDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, augment: bool) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.mask_dir}. "
                "See data/README.md or run book-spine-prepare-yolo."
            )

        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(f"No image/mask pairs found for split '{split}' in {self.root}")

        self.color_jitter = ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.15:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-4.0, 4.0)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            image = self.color_jitter(image)

        image, _ = letterbox_image(image, self.image_size, fill=(0, 0, 0))
        mask, _ = letterbox_image(
            mask,
            self.image_size,
            resample=Image.Resampling.NEAREST,
            fill=0,
        )

        image_tensor = image_to_tensor(image)
        mask_tensor = (TF.pil_to_tensor(mask).squeeze(0) > 0).long()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "path": str(image_path),
        }


def preview_sample(dataset: SeparatorDataset, index: int = 0):
    """Return an image/mask figure for notebook inspection."""
    import matplotlib.pyplot as plt

    sample = dataset[index]
    image = sample["image"].clone()
    for channel, mean, std in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        channel.mul_(std).add_(mean)
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    mask = sample["mask"].numpy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title(Path(sample["path"]).name)
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap="spring", alpha=0.55)
    axes[1].set_title("Separator target")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure
