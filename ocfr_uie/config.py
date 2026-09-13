"""Configuration for the released UIEB and LSUI training recipes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    dataset: str
    split_dir: str
    initialization: str
    output_dir: str
    seed: int = 7
    feature_channels: int = 28
    image_size: int = 256
    epochs: int = 100
    batch_size: int = 24
    eval_batch_size: int = 24
    num_workers: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    ssim_tie_epsilon: float = 0.002
    data_parallel: bool = False
    dinov2_repo: str = ""
    dinov2_checkpoint: str = ""
    dinov2_model: str = "dinov2_vitb14"
    dinov2_layers: str = "3,7"
    dinov2_feature_size: int = 224
    dinov2_chunk_size: int = 4
    refinement_epochs: int = 1
    refinement_schedule_epochs: int = 10
    refinement_learning_rate: float = 1e-5
    refinement_min_learning_rate: float = 1e-6
    refinement_micro_batch_size: int = 4
    uranker_chunk_size: int = 4
    cagrad_conflict_aversion: float = 0.4
    grad_clip: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown configuration keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in {"UIEB", "LSUI"}:
            raise ValueError("dataset must be UIEB or LSUI")
        if self.feature_channels % 4 or self.image_size % 4:
            raise ValueError("feature_channels and image_size must be divisible by four")
        if min(self.epochs, self.batch_size, self.eval_batch_size) <= 0:
            raise ValueError("epochs and batch sizes must be positive")
        if self.refinement_epochs <= 0:
            raise ValueError("refinement_epochs must be positive")
        if self.refinement_epochs > self.refinement_schedule_epochs:
            raise ValueError("refinement_epochs exceeds the refinement schedule")
        if self.batch_size % self.refinement_micro_batch_size:
            raise ValueError("batch_size must be divisible by refinement_micro_batch_size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
