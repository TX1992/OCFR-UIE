#!/usr/bin/env python3
"""Enhance a directory of RGB images with OCFR-UIE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ocfr_uie import build_model
from ocfr_uie.checkpoint import (
    load_checkpoint,
    model_kwargs_from_metadata,
    read_checkpoint_metadata,
)
from ocfr_uie.data import IMAGE_EXTENSIONS, pil_to_tensor


def _pad_to_multiple_of_four(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_height = (-height) % 4
    pad_width = (-width) % 4
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    return image, (height, width)


def _save_tensor(image: torch.Tensor, path: Path) -> None:
    array = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCFR-UIE directory inference")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resize", default=0, type=int, help="Square size; 0 keeps native resolution"
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile a fixed-resolution CUDA graph; requires --resize.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    metadata = read_checkpoint_metadata(args.checkpoint)
    model = build_model(**model_kwargs_from_metadata(metadata)).to(device).eval()
    load_checkpoint(model, args.checkpoint, strict=True, map_location=device)
    if args.compile:
        if device.type != "cuda":
            raise ValueError("--compile requires CUDA")
        if args.resize <= 0:
            raise ValueError("--compile requires a fixed positive --resize")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)
    pattern = "**/*" if args.recursive else "*"
    paths = sorted(
        path
        for path in args.input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"no input images found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for path in tqdm(paths, desc="infer"):
            image = Image.open(path).convert("RGB")
            if args.resize > 0:
                image = image.resize((args.resize, args.resize), Image.Resampling.BILINEAR)
            tensor, original_size = _pad_to_multiple_of_four(pil_to_tensor(image).unsqueeze(0))
            prediction = model(tensor.to(device))[0, :, : original_size[0], : original_size[1]]
            relative = path.relative_to(args.input_dir) if args.recursive else Path(path.name)
            output_path = args.output_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _save_tensor(prediction, output_path.with_suffix(".png"))
    print(f"saved {len(paths)} images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
