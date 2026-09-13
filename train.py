#!/usr/bin/env python3
"""Train OCFR-UIE with the UIEB or LSUI recipe used in the paper."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from tqdm import tqdm

from ocfr_uie import build_model
from ocfr_uie.checkpoint import load_checkpoint, save_checkpoint
from ocfr_uie.config import TrainConfig
from ocfr_uie.data import build_manifest_loader
from ocfr_uie.losses import OCFRTrainingLoss
from ocfr_uie.metrics import float_numpy_batch, psnr, ssim
from ocfr_uie.quality import BoundedURankerObjective, combine_two_objective_cagrad


ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve(path: str) -> str:
    value = Path(path).expanduser()
    return str(value if value.is_absolute() else ROOT / value)


def build_loss(config: TrainConfig) -> OCFRTrainingLoss:
    layers = tuple(int(item) for item in config.dinov2_layers.split(","))
    return OCFRTrainingLoss(
        charbonnier_weight=1.0,
        hvi_weight=0.5,
        ssim_weight=0.1,
        perceptual_weight=0.1,
        edge_weight=0.1,
        dinov2_dpc_weight=0.00065,
        dinov2_repo=config.dinov2_repo,
        dinov2_checkpoint=config.dinov2_checkpoint,
        dinov2_model=config.dinov2_model,
        dinov2_layers=layers,
        dinov2_feature_size=config.dinov2_feature_size,
        dinov2_chunk_size=config.dinov2_chunk_size,
    )


@torch.no_grad()
def validate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for raw, target, _ in tqdm(loader, desc="validation", leave=False):
        prediction = model(raw.to(device, non_blocking=True)).clamp(0.0, 1.0)
        for pred_item, target_item in zip(
            float_numpy_batch(prediction), float_numpy_batch(target)
        ):
            psnr_values.append(psnr(pred_item, target_item))
            ssim_values.append(ssim(pred_item, target_item))
    return float(np.mean(ssim_values)), float(np.mean(psnr_values))


def select_checkpoint(
    value_ssim: float,
    value_psnr: float,
    selected_ssim: float | None,
    selected_psnr: float,
    best_ssim: float,
    tolerance: float,
) -> bool:
    if selected_ssim is None:
        return True
    lower_bound = best_ssim - tolerance
    if value_ssim < lower_bound:
        return False
    if selected_ssim < lower_bound or value_ssim > selected_ssim + tolerance:
        return True
    return abs(value_ssim - selected_ssim) <= tolerance and value_psnr > selected_psnr


def paired_training(
    config: TrainConfig,
    data_root: str,
    device: torch.device,
    run_dir: Path,
) -> Path:
    model = build_model(config.feature_channels).to(device)
    load_checkpoint(model, config.initialization, strict=True, map_location=device)
    if config.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    criterion = build_loss(config).to(device)
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

    seed_everything(config.seed)
    train_loader = build_manifest_loader(
        config.split_dir,
        "train",
        data_root=data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        training=True,
        resize=True,
        num_workers=config.num_workers,
    )
    val_loader = build_manifest_loader(
        config.split_dir,
        "val",
        data_root=data_root,
        batch_size=config.eval_batch_size,
        image_size=config.image_size,
        training=False,
        resize=True,
        num_workers=1,
    )

    selected_ssim: float | None = None
    selected_psnr = -float("inf")
    best_ssim = -float("inf")
    checkpoint_path = run_dir / "paired_best.pth"
    for epoch in range(1, config.epochs + 1):
        model.train()
        mean_loss = 0.0
        batches = 0
        for raw, target, _ in tqdm(
            train_loader, desc=f"paired {epoch}/{config.epochs}", leave=False
        ):
            raw = raw.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = criterion(model(raw), target, raw)
            loss.backward()
            optimizer.step()
            mean_loss += float(loss.detach())
            batches += 1

        value_ssim, value_psnr = validate(model, val_loader, device)
        best_ssim = max(best_ssim, value_ssim)
        selected = select_checkpoint(
            value_ssim,
            value_psnr,
            selected_ssim,
            selected_psnr,
            best_ssim,
            config.ssim_tie_epsilon,
        )
        if selected:
            selected_ssim, selected_psnr = value_ssim, value_psnr
            save_checkpoint(
                checkpoint_path,
                model,
                meta={
                    "architecture": "OCFRUIE",
                    "feature_channels": config.feature_channels,
                    "dataset": config.dataset,
                    "epoch": epoch,
                    "validation_psnr": value_psnr,
                    "validation_ssim": value_ssim,
                    "config": config.to_dict(),
                },
            )
        scheduler.step()
        print(
            f"epoch={epoch:03d} loss={mean_loss / batches:.6f} "
            f"val_psnr={value_psnr:.4f} val_ssim={value_ssim:.6f} "
            f"selected={selected}"
        )
    return checkpoint_path


def gradients(loss: torch.Tensor, parameters: list[torch.nn.Parameter]):
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, values)
    ]


def add_gradients(destination, source, scale: float) -> None:
    for target, value in zip(destination, source):
        target.add_(value, alpha=scale)


def quality_refinement(
    config: TrainConfig,
    data_root: str,
    device: torch.device,
    run_dir: Path,
    checkpoint: Path,
) -> Path:
    seed_everything(config.seed)
    model = build_model(config.feature_channels).to(device)
    load_checkpoint(model, checkpoint, strict=True, map_location=device)
    parameters = list(model.parameters())
    fidelity = build_loss(config).to(device)
    quality = BoundedURankerObjective(
        device=device, chunk_size=config.uranker_chunk_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.refinement_learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    train_loader = build_manifest_loader(
        config.split_dir,
        "train",
        data_root=data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        training=True,
        resize=True,
        num_workers=config.num_workers,
    )
    total_steps = config.refinement_schedule_epochs * len(train_loader)
    global_step = 0

    for epoch in range(1, config.refinement_epochs + 1):
        model.train()
        quality.train()
        for raw, target, _ in tqdm(
            train_loader,
            desc=f"refinement {epoch}/{config.refinement_epochs}",
            leave=False,
        ):
            raw = raw.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            count = raw.shape[0]
            fidelity_grad = [torch.zeros_like(parameter) for parameter in parameters]
            quality_grad = [torch.zeros_like(parameter) for parameter in parameters]
            for start in range(0, count, config.refinement_micro_batch_size):
                end = start + config.refinement_micro_batch_size
                raw_part = raw[start:end]
                target_part = target[start:end]
                fraction = raw_part.shape[0] / count

                fidelity_loss, _ = fidelity(model(raw_part), target_part, raw_part)
                add_gradients(
                    fidelity_grad, gradients(fidelity_loss, parameters), fraction
                )
                quality_loss, _ = quality(model(raw_part))
                add_gradients(
                    quality_grad, gradients(quality_loss, parameters), fraction
                )

            combined, _ = combine_two_objective_cagrad(
                fidelity_grad,
                quality_grad,
                conflict_aversion=config.cagrad_conflict_aversion,
            )
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, combined):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            progress = global_step / max(total_steps - 1, 1)
            learning_rate = config.refinement_min_learning_rate + 0.5 * (
                config.refinement_learning_rate - config.refinement_min_learning_rate
            ) * (1.0 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)
            optimizer.step()
            global_step += 1

    final_path = run_dir / "final.pth"
    save_checkpoint(
        final_path,
        model,
        meta={
            "architecture": "OCFRUIE",
            "feature_channels": config.feature_channels,
            "dataset": config.dataset,
            "quality_refinement_steps": global_step,
            "config": config.to_dict(),
        },
    )
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uieb.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--dinov2-checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    config = TrainConfig.from_json(resolve(args.config))
    config.split_dir = resolve(config.split_dir)
    config.initialization = resolve(config.initialization)
    config.output_dir = resolve(args.output_dir or config.output_dir)
    config.dinov2_repo = str(Path(args.dinov2_repo).expanduser().resolve())
    config.dinov2_checkpoint = str(Path(args.dinov2_checkpoint).expanduser().resolve())
    config.validate()

    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.cuda.set_per_process_memory_fraction(0.45, device=0)
    for required in (config.initialization, config.dinov2_checkpoint):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    if not Path(config.dinov2_repo).is_dir():
        raise FileNotFoundError(config.dinov2_repo)

    seed_everything(config.seed)
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    device = torch.device("cuda")
    paired_checkpoint = paired_training(config, args.data_root, device, run_dir)
    final_checkpoint = quality_refinement(
        config, args.data_root, device, run_dir, paired_checkpoint
    )
    print(f"final checkpoint: {final_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
