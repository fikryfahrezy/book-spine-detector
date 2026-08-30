from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import SeparatorDataset
from .losses import separator_loss
from .metrics import binary_segmentation_metrics
from .model import create_model
from .runtime import resolve_device, seed_everything


def _run_epoch(
    model,
    loader,
    *,
    device,
    loss_options,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    sample_count = 0

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            batch_size = images.shape[0]

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss, _ = separator_loss(logits, masks, **loss_options)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            metrics = binary_segmentation_metrics(logits.detach(), masks)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["dice"] += metrics["dice"] * batch_size
            totals["iou"] += metrics["iou"] * batch_size
            sample_count += batch_size

    return {name: value / max(1, sample_count) for name, value in totals.items()}


def train_from_config(
    config_path: str | Path,
    device_name: str | None = None,
    resume_path: str | Path | None = None,
) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(device_name)

    data_config = config["data"]
    training_config = config["training"]
    data_root = Path(data_config["root"])
    image_size = int(data_config.get("image_size", 640))
    workers = int(data_config.get("num_workers", 2))

    train_dataset = SeparatorDataset(data_root, "train", image_size, augment=True)
    validation_dataset = SeparatorDataset(data_root, "val", image_size, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_options = {
        "separator_class_weight": float(
            training_config.get("separator_class_weight", 8.0)
        ),
        "dice_weight": float(training_config.get("dice_weight", 1.0)),
    }

    output_dir = Path(training_config.get("output_dir", "runs/separator-baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    epochs = int(training_config.get("epochs", 40))
    patience = int(training_config.get("patience", 8))
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict] = []
    start_epoch = 1

    configured_resume = resume_path or training_config.get("resume")
    if configured_resume:
        configured_resume = Path(configured_resume)
        resume_checkpoint = torch.load(configured_resume, map_location="cpu")
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if use_amp and resume_checkpoint.get("scaler_state"):
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_loss = float(resume_checkpoint.get("metrics", {}).get("loss", best_loss))
        best_checkpoint_path = output_dir / "best.pt"
        if best_checkpoint_path.exists():
            previous_best = torch.load(best_checkpoint_path, map_location="cpu")
            best_loss = min(
                best_loss,
                float(previous_best.get("metrics", {}).get("loss", best_loss)),
            )
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resuming from {configured_resume} at epoch {start_epoch}")

    print(f"Training on {device}; {len(train_dataset)} train / {len(validation_dataset)} val")
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            loss_options=loss_options,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device=device,
            loss_options=loss_options,
            use_amp=use_amp,
        )
        scheduler.step(validation_metrics["loss"])

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": validation_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} | "
            f"val dice {validation_metrics['dice']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "config": config,
            "metrics": validation_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if stale_epochs >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    return {
        "device": str(device),
        "best_validation_loss": best_loss,
        "checkpoint": str(output_dir / "best.pt"),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train separator segmentation model")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--resume", default=None, help="Path to last.pt to resume")
    args = parser.parse_args()
    train_from_config(args.config, args.device, args.resume)


if __name__ == "__main__":
    main()
