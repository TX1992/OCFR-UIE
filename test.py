#!/usr/bin/env python3
"""Evaluate the released OCFR-UIE checkpoints on UIEB or LSUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ocfr_uie import build_model
from ocfr_uie.checkpoint import load_checkpoint
from ocfr_uie.data import build_manifest_loader
from ocfr_uie.metrics import float_numpy_batch, psnr, quantized_numpy_batch, ssim


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "UIEB": ("splits/UIEB", "pretrained/ocfr_uie_uieb_best.pth", 1),
    "LSUI": ("splits/LSUI", "pretrained/ocfr_uie_lsui_best.pth", 24),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="UIEB")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override the verified dataset-specific evaluation batch size.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    split_dir, default_checkpoint, verified_batch_size = DATASETS[args.dataset]
    batch_size = args.batch_size or verified_batch_size
    checkpoint = Path(args.checkpoint or ROOT / default_checkpoint).resolve()
    output_dir = Path(args.output_dir or ROOT / "outputs" / args.dataset.lower())
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    loader = build_manifest_loader(
        ROOT / split_dir,
        "test",
        data_root=args.data_root,
        batch_size=batch_size,
        image_size=256,
        training=False,
        resize=True,
        num_workers=args.num_workers,
    )
    model = build_model(feature_channels=28).to(device).eval()
    load_checkpoint(model, checkpoint, strict=True, map_location=device)

    if args.save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    with torch.inference_mode():
        for raw, target, names in tqdm(loader, desc=f"test {args.dataset}"):
            prediction = model(raw.to(device, non_blocking=True))
            predictions = quantized_numpy_batch(prediction)
            targets = float_numpy_batch(target)
            for prediction_item, target_item, name in zip(
                predictions, targets, names
            ):
                psnr_values.append(psnr(prediction_item, target_item))
                ssim_values.append(ssim(prediction_item, target_item))
                if args.save_images:
                    Image.fromarray(
                        np.rint(prediction_item * 255.0).astype(np.uint8), mode="RGB"
                    ).save(output_dir / f"{Path(name).stem}.png")

    result = {
        "dataset": args.dataset,
        "images": len(psnr_values),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "checkpoint": str(checkpoint),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
