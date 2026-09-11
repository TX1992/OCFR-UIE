"""Fixed-split paired-image loading for OCFR-UIE."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class ManifestPair:
    input_path: Path
    target_path: Path
    name: str


@dataclass(frozen=True)
class SplitManifest:
    path: Path
    dataset: str
    data_root: Path
    splits: dict[str, tuple[ManifestPair, ...]]

    def pairs(self, split: str) -> tuple[ManifestPair, ...]:
        if split not in self.splits:
            raise KeyError(f"split {split!r} is not present in {self.path}")
        return self.splits[split]


def _resolve_image(root: Path, value: str, field: str, list_path: Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{list_path}: {field} must be a safe path relative to data_root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{list_path}: {field} escapes data_root: {value}") from error
    if not path.is_file():
        raise FileNotFoundError(f"split {field} does not exist: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"split {field} is not a supported image: {path}")
    return path


def load_split_manifest(
    split_directory: str | Path,
    *,
    data_root: str | Path | None = None,
) -> SplitManifest:
    """Load tab-separated relative image pairs from train/val/test.txt."""

    split_directory = Path(split_directory).expanduser().resolve()
    if not split_directory.is_dir():
        raise FileNotFoundError(f"split directory does not exist: {split_directory}")
    if data_root in (None, ""):
        raise ValueError(f"{split_directory}: --data-root is required")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    splits: dict[str, tuple[ManifestPair, ...]] = {}
    names_by_split: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        list_path = split_directory / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"missing split list: {list_path}")
        pairs: list[ManifestPair] = []
        names: set[str] = set()
        for line_number, line in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = [field.strip() for field in line.split("\t")]
            if len(fields) != 2 or not all(fields):
                raise ValueError(
                    f"{list_path}:{line_number}: expected INPUT<TAB>GT relative paths"
                )
            input_path = _resolve_image(root, fields[0], "input", list_path)
            target_path = _resolve_image(root, fields[1], "gt", list_path)
            if input_path.name != target_path.name:
                raise ValueError(f"{list_path}:{line_number}: paired filenames must match")
            name = input_path.name
            if name in names:
                raise ValueError(f"{list_path}: duplicate sample name: {name}")
            names.add(name)
            pairs.append(ManifestPair(input_path, target_path, name))
        if not pairs:
            raise ValueError(f"split list is empty: {list_path}")
        splits[split] = tuple(pairs)
        names_by_split[split] = names

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = names_by_split[left] & names_by_split[right]
            if overlap:
                raise ValueError(
                    f"{split_directory}: {left}/{right} overlap: {sorted(overlap)[:8]}"
                )
    return SplitManifest(split_directory, split_directory.name, root, splits)


class PairedImageDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[ManifestPair],
        image_size: int = 256,
        training: bool = False,
        resize: bool = True,
    ) -> None:
        super().__init__()
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.resize = bool(resize)
        if not self.pairs:
            raise RuntimeError("paired image dataset is empty")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        raw = load_rgb(pair.input_path)
        target = load_rgb(pair.target_path)
        if target.size != raw.size:
            target = target.resize(raw.size, Image.Resampling.BILINEAR)
        if self.resize:
            size = (self.image_size, self.image_size)
            raw = raw.resize(size, Image.Resampling.BILINEAR)
            target = target.resize(size, Image.Resampling.BILINEAR)

        # Preserve the validated training RNG sequence even when resize makes
        # the crop cover the complete image.
        if self.training:
            width, height = raw.size
            if width < self.image_size or height < self.image_size:
                raise ValueError(f"image {pair.input_path} is smaller than the crop size")
            left = random.randrange(width - self.image_size + 1)
            top = random.randrange(height - self.image_size + 1)
            box = (left, top, left + self.image_size, top + self.image_size)
            raw, target = raw.crop(box), target.crop(box)
            if random.random() < 0.5:
                raw, target = ImageOps.flip(raw), ImageOps.flip(target)
            if random.random() < 0.5:
                raw, target = ImageOps.mirror(raw), ImageOps.mirror(target)
            if random.random() < 0.5:
                angle = random.choice((90, 180, 270))
                raw, target = raw.rotate(angle), target.rotate(angle)
        return pil_to_tensor(raw), pil_to_tensor(target), pair.name


def build_manifest_loader(
    split_directory: str | Path,
    split: str,
    *,
    data_root: str | Path | None = None,
    batch_size: int,
    image_size: int = 256,
    training: bool = False,
    resize: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    manifest = load_split_manifest(split_directory, data_root=data_root)
    dataset = PairedImageDataset(
        manifest.pairs(split),
        image_size=image_size,
        training=training,
        resize=resize,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=False,
    )
