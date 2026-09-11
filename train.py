#!/usr/bin/env python3
"""Train OCFR-UIE with the paper's single-stage paired-image protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from tqdm import tqdm

from ocfr_uie import build_model
from ocfr_uie.checkpoint import load_checkpoint, save_checkpoint
from ocfr_uie.config import TrainConfig
from ocfr_uie.data import build_manifest_loader, load_split_manifest
from ocfr_uie.evaluation.metrics import get_psnr, get_ssim, to_numpy_batch
from ocfr_uie.losses import OCFRTrainingLoss


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(deterministic)


def should_replace_selected_checkpoint(
    ssim: float,
    psnr: float,
    selected_ssim: float | None,
    selected_psnr: float,
    strict_best_ssim: float,
    tie_epsilon: float,
) -> tuple[bool, str]:
    if selected_ssim is None:
        return True, "first_validation"
    tie_floor = strict_best_ssim - tie_epsilon
    if ssim < tie_floor:
        return False, "below_strict_ssim_tie_window"
    if selected_ssim < tie_floor or ssim > selected_ssim + tie_epsilon:
        return True, "higher_ssim"
    if abs(ssim - selected_ssim) <= tie_epsilon and psnr > selected_psnr:
        return True, "ssim_tie_higher_psnr"
    return False, "not_selected"


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    description: str = "val",
) -> tuple[float, float]:
    model.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for raw, target, _ in tqdm(loader, desc=description, leave=False):
        prediction = model(raw.to(device, non_blocking=True)).clamp(0.0, 1.0)
        for prediction_item, target_item in zip(
            to_numpy_batch(prediction),
            to_numpy_batch(target),
        ):
            psnr_values.append(get_psnr(prediction_item, target_item))
            ssim_values.append(get_ssim(prediction_item, target_item))
    if not psnr_values:
        raise RuntimeError("validation produced no samples")
    return float(np.mean(ssim_values)), float(np.mean(psnr_values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCFR-UIE")
    parser.add_argument("--config", default="configs/uieb_ocfr.json")
    parser.add_argument("--split-file", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--initialization-checkpoint", default="")
    parser.add_argument("--dinov2-repo", default="")
    parser.add_argument("--dinov2-checkpoint", default="")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    if args.split_file:
        config.split_file = args.split_file
    if args.data_root:
        config.data_root = args.data_root
    if args.output_root:
        config.output_root = args.output_root
    if args.run_name:
        config.run_name = args.run_name
    if args.initialization_checkpoint:
        config.initialization_checkpoint = args.initialization_checkpoint
    if args.dinov2_repo:
        config.dinov2_repo = args.dinov2_repo
    if args.dinov2_checkpoint:
        config.dinov2_checkpoint = args.dinov2_checkpoint
    config.validate()

    seed_everything(config.seed, config.deterministic)
    if not torch.cuda.is_available():
        raise RuntimeError("OCFR-UIE training requires CUDA")
    device = torch.device("cuda")
    model = build_model(config.feature_channels).to(device)
    initialization: dict[str, object] = {}
    if config.initialization_checkpoint:
        initialization = load_checkpoint(
            model,
            config.initialization_checkpoint,
            strict=True,
            map_location=device,
        )
        print(f"initialization: {Path(config.initialization_checkpoint).resolve()}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if config.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dinov2_layers = tuple(
        int(layer.strip()) for layer in config.dinov2_layers.split(",") if layer.strip()
    )
    criterion = OCFRTrainingLoss(
        charbonnier_weight=config.charbonnier_weight,
        hvi_weight=config.hvi_weight,
        ssim_weight=config.ssim_weight,
        perceptual_weight=config.perceptual_weight,
        dinov2_dpc_weight=config.dinov2_dpc_weight,
        edge_weight=config.edge_weight,
        dinov2_repo=config.dinov2_repo,
        dinov2_checkpoint=config.dinov2_checkpoint,
        dinov2_model=config.dinov2_model,
        dinov2_layers=dinov2_layers,
        dinov2_feature_size=config.dinov2_feature_size,
        dinov2_chunk_size=config.dinov2_chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 1e-4,
    )

    # Decouple data order from model and frozen-loss construction.
    seed_everything(config.seed, config.deterministic)
    manifest = load_split_manifest(config.split_file, data_root=config.data_root or None)
    train_loader = build_manifest_loader(
        manifest.path,
        "train",
        data_root=manifest.data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        training=True,
        resize=config.resize,
        num_workers=config.num_workers,
    )
    val_loader = build_manifest_loader(
        manifest.path,
        "val",
        data_root=manifest.data_root,
        batch_size=config.eval_batch_size,
        image_size=config.image_size,
        training=False,
        resize=config.resize,
        num_workers=1,
    )
    run_dir = Path(config.output_root) / config.run_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(f"run_dir: {run_dir.resolve()}")
    print(f"split_manifest: {manifest.path}")
    print(f"parameters: {parameter_count}")
    print(
        f"architecture: {model.module.architecture_version if isinstance(model, torch.nn.DataParallel) else model.architecture_version}"
    )
    print(f"activation_checkpointing: {config.activation_checkpointing}")
    print(f"checkpoint_metric: {config.checkpoint_metric}")
    print(
        "loss_weights: "
        f"charbonnier={config.charbonnier_weight:g}, "
        f"hvi={config.hvi_weight:g}, "
        f"ssim={config.ssim_weight:g}, "
        f"vgg={config.perceptual_weight:g}, "
        f"edge={config.edge_weight:g}, "
        f"dpc={config.dinov2_dpc_weight:g}"
    )

    selected_ssim: float | None = None
    selected_psnr = -float("inf")
    strict_best_ssim = -float("inf")
    strict_best_psnr = -float("inf")
    records_path = run_dir / "metrics.jsonl"
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        component_sums: dict[str, float] = {}
        batch_count = 0
        progress = tqdm(train_loader, desc=f"train {epoch}/{config.epochs}", leave=False)
        for raw, target, _ in progress:
            raw = raw.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = (
                activation_checkpoint(
                    model,
                    raw,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                if config.activation_checkpointing
                else model(raw)
            )
            loss, components = criterion(prediction, target, raw)
            loss.backward()
            optimizer.step()
            batch_count += 1
            loss_value = float(loss.detach().item())
            component_sums["total"] = component_sums.get("total", 0.0) + loss_value
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.item())
            progress.set_postfix(loss=f"{loss_value:.5f}")
        if batch_count == 0:
            raise RuntimeError("training epoch produced no batches")

        val_ssim, val_psnr = validate(
            model,
            val_loader,
            device,
            description="val",
        )
        strict_ssim_improved = val_ssim > strict_best_ssim
        strict_psnr_improved = val_psnr > strict_best_psnr
        strict_best_ssim = max(strict_best_ssim, val_ssim)
        strict_best_psnr = max(strict_best_psnr, val_psnr)
        selected, reason = should_replace_selected_checkpoint(
            val_ssim,
            val_psnr,
            selected_ssim,
            selected_psnr,
            strict_best_ssim,
            config.ssim_tie_epsilon,
        )
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        metadata = {
            "architecture": type(base_model).__name__,
            "architecture_version": base_model.architecture_version,
            "feature_channels": config.feature_channels,
            "epoch": epoch,
            "validation_ssim": val_ssim,
            "validation_psnr": val_psnr,
            "selection_ssim": val_ssim,
            "selection_psnr": val_psnr,
            "checkpoint_metric": config.checkpoint_metric,
            "selection_reason": reason,
            "config": config.to_dict(),
            "initialization": initialization,
        }
        if strict_ssim_improved:
            save_checkpoint(run_dir / "best_ssim.pth", model, meta=metadata)
        if strict_psnr_improved:
            save_checkpoint(run_dir / "best_psnr.pth", model, meta=metadata)
        if selected:
            selected_ssim, selected_psnr = val_ssim, val_psnr
            save_checkpoint(run_dir / "best_model.pth", model, meta=metadata)
        save_checkpoint(run_dir / "latest.pth", model, meta=metadata)

        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "loss": {name: value / batch_count for name, value in component_sums.items()},
            "validation_ssim": val_ssim,
            "validation_psnr": val_psnr,
            "selection_ssim": val_ssim,
            "selection_psnr": val_psnr,
            "checkpoint_metric": config.checkpoint_metric,
            "selected": selected,
            "selection_reason": reason,
            "elapsed_seconds": time.time() - start_time,
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch={epoch:03d} loss={record['loss']['total']:.6f} "
            f"val_psnr={val_psnr:.4f} val_ssim={val_ssim:.6f} "
            + f"selected={selected} reason={reason}"
        )

    print(f"completed: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
