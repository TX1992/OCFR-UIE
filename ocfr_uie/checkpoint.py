"""OCFR-UIE checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a state-dict or checkpoint mapping")
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }
    if not state:
        raise ValueError("checkpoint does not contain tensor parameters")
    return state


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    raw = torch.load(checkpoint_path, map_location=map_location)
    metadata = raw.get("meta", {}) if isinstance(raw, Mapping) else {}
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    checkpoint_version = metadata.get("architecture_version")
    model_version = getattr(model, "architecture_version", None)
    if checkpoint_version and model_version and checkpoint_version != model_version:
        raise ValueError(
            f"checkpoint architecture {checkpoint_version!r} does not match "
            f"model architecture {model_version!r}"
        )
    incompatible = model.load_state_dict(_unwrap_state_dict(raw), strict=strict)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "meta": metadata,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"model": model_to_save.state_dict(), "meta": dict(meta or {})},
        temporary,
    )
    temporary.replace(path)
    return path
