#!/usr/bin/env python3
"""Evaluate OCFR-UIE on a paired split with the unified metric interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ocfr_uie import build_model
from ocfr_uie.checkpoint import (
    load_checkpoint,
    model_kwargs_from_metadata,
    read_checkpoint_metadata,
)
from ocfr_uie.data import build_manifest_loader, load_split_manifest
from ocfr_uie.evaluation import (
    UnifiedEvaluator,
    format_metrics,
    normalize_metric_names,
    write_evaluation_outputs,
)


def _save_batch(prediction: torch.Tensor, names: list[str], output_dir: Path) -> None:
    prediction = prediction.detach().float().clamp(0.0, 1.0).cpu().numpy()
    for image, name in zip(prediction, names):
        array = np.transpose(image, (1, 2, 0))
        Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(
            output_dir / f"{Path(name).stem}.png"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="OCFR-UIE paired evaluation")
    parser.add_argument(
        "--split-file",
        default="splits/UIEB",
        type=Path,
        help="Directory containing train.txt, val.txt, and test.txt",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--data-root",
        default=None,
        type=Path,
        help="Optional override for the manifest's top-level dataset_root",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--feature-channels",
        default=0,
        type=int,
        help="Model width; zero infers feature_channels from checkpoint metadata.",
    )
    parser.add_argument("--output-dir", default="outputs/test", type=Path)
    parser.add_argument(
        "--metrics",
        default="psnr,ssim,uiqm,uciqe",
        help="Comma-separated metrics; UCIQE uses the original CIELab definition.",
    )
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=24, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the fixed-resolution CUDA evaluation graph.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    metric_names = normalize_metric_names(args.metrics)
    manifest = load_split_manifest(args.split_file, data_root=args.data_root)
    loader = build_manifest_loader(
        manifest.path,
        args.split,
        data_root=manifest.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        training=False,
        resize=True,
        num_workers=args.num_workers,
    )
    checkpoint_metadata = read_checkpoint_metadata(args.checkpoint)
    model_kwargs = model_kwargs_from_metadata(checkpoint_metadata)
    if args.feature_channels:
        model_kwargs["feature_channels"] = args.feature_channels
    model = build_model(**model_kwargs).to(device).eval()
    report = load_checkpoint(model, args.checkpoint, strict=True, map_location=device)
    if args.compile:
        if device.type != "cuda":
            raise ValueError("--compile requires CUDA")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    evaluator = UnifiedEvaluator(metric_names, device=device, iqa_size=args.image_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_output = args.output_dir / "images"
    if args.save_images:
        image_output.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for raw, target, names in tqdm(loader, desc="test"):
            prediction = model(raw.to(device, non_blocking=True)).clamp(0.0, 1.0)
            rows.extend(
                evaluator.evaluate_tensor_batch(
                    prediction,
                    target,
                    datasets=[manifest.dataset] * len(names),
                    names=list(names),
                )
            )
            if args.save_images:
                _save_batch(prediction, list(names), image_output)
    overall, sample_average, per_dataset = evaluator.summarize(rows)
    write_evaluation_outputs(
        args.output_dir,
        rows,
        overall,
        sample_average,
        per_dataset,
    )
    metadata = {
        "images": len(rows),
        "dataset": manifest.dataset,
        "split": args.split,
        "split_manifest": str(manifest.path),
        "dataset_root": str(manifest.data_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": report["meta"],
        **model_kwargs,
        "metrics": metric_names,
        "image_size": args.image_size,
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"images: {len(rows)}")
    print(format_metrics(sample_average, metric_names))
    print(f"saved: {(args.output_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
