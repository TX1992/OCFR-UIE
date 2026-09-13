"""Frozen URanker objective and two-task CAGrad used for quality calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class BoundedURankerObjective(nn.Module):
    """Official differentiable URanker score with bounded minimization."""

    def __init__(self, *, device: torch.device, chunk_size: int = 4) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        import pyiqa

        self.chunk_size = int(chunk_size)
        self.uranker = freeze_module(
            pyiqa.create_metric(
                "uranker",
                device=device,
                as_loss=True,
                loss_reduction="none",
            )
        )

    def train(self, mode: bool = True) -> "BoundedURankerObjective":
        super().train(mode)
        self.uranker.eval()
        return self

    def scores(self, prediction: torch.Tensor) -> torch.Tensor:
        prediction = prediction.clamp(0.0, 1.0)
        scores = []
        for chunk in prediction.split(self.chunk_size):
            score = self.uranker(chunk).reshape(chunk.shape[0], -1).mean(1)
            scores.append(score)
        return torch.cat(scores)

    def forward(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        score = self.scores(prediction)
        quality = torch.sigmoid(-score).mean()
        return quality, {
            "uranker_score": score.detach().mean(),
            "quality": quality.detach(),
        }


@dataclass(frozen=True)
class CAGradDiagnostics:
    cosine: float
    fidelity_norm: float
    quality_norm: float
    quality_weight: float
    conflict_scale: float


def _dot(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> torch.Tensor:
    if len(left) != len(right) or not left:
        raise ValueError("gradient lists must be non-empty and have equal length")
    total = left[0].new_zeros(())
    for left_item, right_item in zip(left, right):
        if left_item.shape != right_item.shape:
            raise ValueError("paired gradients must have matching shapes")
        total = total + torch.sum(left_item.float() * right_item.float())
    return total


def combine_two_objective_cagrad(
    fidelity_gradients: Sequence[torch.Tensor],
    quality_gradients: Sequence[torch.Tensor],
    *,
    conflict_aversion: float = 0.4,
    epsilon: float = 1e-8,
) -> tuple[list[torch.Tensor], CAGradDiagnostics]:
    """Official two-objective CAGrad dual without flattening gradients."""

    if not 0.0 <= conflict_aversion < 1.0:
        raise ValueError("conflict_aversion must be in [0, 1)")
    g11 = float(_dot(fidelity_gradients, fidelity_gradients).detach())
    g12 = float(_dot(fidelity_gradients, quality_gradients).detach())
    g22 = float(_dot(quality_gradients, quality_gradients).detach())
    fidelity_norm = float(np.sqrt(max(g11, 0.0)))
    quality_norm = float(np.sqrt(max(g22, 0.0)))
    cosine = float(
        np.clip(g12 / max(fidelity_norm * quality_norm, epsilon), -1.0, 1.0)
    )
    g0_norm = float(
        np.sqrt(0.25 * max(g11 + 2.0 * g12 + g22, 0.0) + epsilon)
    )
    coefficient = conflict_aversion * g0_norm

    def dual_objective(quality_weight: float) -> float:
        fidelity_weight = 1.0 - quality_weight
        gw_dot_g0 = 0.5 * (
            fidelity_weight * (g11 + g12)
            + quality_weight * (g12 + g22)
        )
        gw_squared = (
            fidelity_weight**2 * g11
            + 2.0 * fidelity_weight * quality_weight * g12
            + quality_weight**2 * g22
        )
        return gw_dot_g0 + coefficient * np.sqrt(max(gw_squared, 0.0) + epsilon)

    result = minimize_scalar(
        dual_objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"CAGrad dual optimization failed: {result.message}")
    quality_weight = float(np.clip(result.x, 0.0, 1.0))
    fidelity_weight = 1.0 - quality_weight
    gw_squared = (
        fidelity_weight**2 * g11
        + 2.0 * fidelity_weight * quality_weight * g12
        + quality_weight**2 * g22
    )
    conflict_scale = coefficient / (
        np.sqrt(max(gw_squared, 0.0) + epsilon) + epsilon
    )
    denominator = 1.0 + conflict_aversion
    combined = []
    for fidelity, quality in zip(fidelity_gradients, quality_gradients):
        average = 0.5 * (fidelity + quality)
        weighted = fidelity_weight * fidelity + quality_weight * quality
        combined.append((average + conflict_scale * weighted) / denominator)
    return combined, CAGradDiagnostics(
        cosine=cosine,
        fidelity_norm=fidelity_norm,
        quality_norm=quality_norm,
        quality_weight=quality_weight,
        conflict_scale=float(conflict_scale),
    )
