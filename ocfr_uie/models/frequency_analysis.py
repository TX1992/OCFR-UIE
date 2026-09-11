"""Optically conditioned frequency analysis and wavelet restoration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import (
    BasicBlock,
    SepConv,
    deterministic_adaptive_avg_pool2d,
    deterministic_reflect_pad2d,
)
from .adaptive_decoding import FrequencyAdaptiveDualPathDecoder


def _make_statistics_gate(channels: int) -> nn.ModuleDict:
    hidden = max(8, channels // 8)
    return nn.ModuleDict(
        {
            "net": nn.Sequential(
                nn.Conv2d(2 * channels, hidden, 1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden, channels, 1, bias=True),
                nn.Sigmoid(),
            )
        }
    )


def _apply_statistics_gate(gate: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
    mean = deterministic_adaptive_avg_pool2d(x, 1)
    variance = (x - mean).square().mean(dim=(2, 3), keepdim=True)
    descriptor = torch.cat(
        [mean, torch.sqrt(variance + 1e-6)],
        dim=1,
    )
    return x * gate.net(descriptor)


def _dual_scale_kernel_mapping() -> torch.Tensor:
    """Map normalized 3x3 kernels to local and dilated 5x5 supports."""

    mapping = torch.zeros(2, 9, 25)
    index = 0
    for row in range(3):
        for column in range(3):
            mapping[0, index, (row + 1) * 5 + column + 1] = 1.0
            mapping[1, index, (2 * row) * 5 + 2 * column] = 1.0
            index += 1
    return mapping


class HaarFrequencyInteractionBlock(nn.Module):
    """Fixed Haar analysis followed by learned frequency-component interaction."""

    def __init__(
        self,
        feature_channels: int,
        *,
        structure_refinement: bool = False,
    ) -> None:
        super().__init__()
        channels = feature_channels
        self.structure_refinement = bool(structure_refinement)
        kernels = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ]
        )
        self.awsi = nn.ModuleDict(
            {
                "ll_branch": nn.Sequential(
                    SepConv(channels, channels, 3, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(channels, channels, 1, bias=False),
                ),
                "hf_reduce": nn.Conv2d(
                    3 * channels,
                    channels,
                    1,
                    bias=False,
                ),
                "hf_branch": nn.Sequential(
                    nn.LeakyReLU(0.1, inplace=True),
                    SepConv(channels, channels, 3, bias=False),
                ),
                "hf_gate": nn.Sequential(
                    nn.Conv2d(2 * channels, channels, 1, bias=True),
                    nn.LeakyReLU(0.1, inplace=True),
                    SepConv(channels, channels, 3, bias=True),
                    nn.Sigmoid(),
                ),
                "post": SepConv(channels, channels, 3, bias=False),
            }
        )
        self.awsi.register_buffer(
            "haar_kernel",
            kernels[:, None].repeat(channels, 1, 1, 1),
        )
        if self.structure_refinement:
            self.refine = FrequencyAdaptiveDualPathDecoder(channels)
        else:
            self.refine = nn.ModuleDict(
                {
                    "local1": BasicBlock(channels, channels, 3),
                    "local2": BasicBlock(channels, channels, 3),
                }
            )

    def _wavelet_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        bands = F.conv2d(
            x,
            self.awsi.haar_kernel,
            stride=2,
            groups=channels,
        ).view(batch, channels, 4, height // 2, width // 2)
        low = self.awsi.ll_branch(bands[:, :, 0])
        high = self.awsi.hf_reduce(
            torch.cat(
                [bands[:, :, index] for index in range(1, 4)],
                dim=1,
            )
        )
        high = self.awsi.hf_branch(high)
        fused = self.awsi.post(low + self.awsi.hf_gate(torch.cat([low, high], dim=1)) * high)
        size = (height, width)
        return (
            F.interpolate(
                fused,
                size,
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                high,
                size,
                mode="bilinear",
                align_corners=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        wavelet, high = self._wavelet_features(x)
        x = x + wavelet
        if self.structure_refinement:
            x = self.refine(x, high)
        else:
            x = self.refine.local2(self.refine.local1(x))
        return 0.5 * (x + residual)


class DualSupportFrequencyPartition(nn.Module):
    """Convexly mix normalized local and dilated low-pass candidates."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 9, hidden, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.kernel_logits = nn.Parameter(torch.zeros(2, 9))
        self.register_buffer(
            "kernel_mapping",
            _dual_scale_kernel_mapping(),
            persistent=False,
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def kernels(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.kernel_logits, dim=1)
        kernels = torch.bmm(
            probabilities.unsqueeze(1),
            self.kernel_mapping.to(dtype=probabilities.dtype),
        ).squeeze(1)
        return kernels.to(dtype=x.dtype).view(2, 1, 5, 5)

    def forward(
        self,
        x: torch.Tensor,
        degradation_statistics: torch.Tensor,
        degradation_map: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape
        kernel_bank = self.kernels(x).repeat(channels, 1, 1, 1)
        candidates = F.conv2d(
            deterministic_reflect_pad2d(x, (2, 2, 2, 2)),
            kernel_bank,
            groups=channels,
        ).view(batch, channels, 2, height, width)

        gate_size = (max(1, height // 8), max(1, width // 8))
        content = deterministic_adaptive_avg_pool2d(x, gate_size)
        condition = torch.cat(
            [
                degradation_statistics.expand(-1, -1, *gate_size),
                deterministic_adaptive_avg_pool2d(degradation_map, gate_size),
            ],
            dim=1,
        )
        mixing = torch.sigmoid(
            F.interpolate(
                self.gate(torch.cat([content, condition], dim=1)),
                (height, width),
                mode="bilinear",
                align_corners=False,
            )
        )
        return torch.lerp(
            candidates[:, :, 1],
            candidates[:, :, 0],
            mixing,
        )


class SumPreservingFrequencyExchange(nn.Module):
    """Exchange one shared correction while preserving the component sum."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 8)
        self.transport = SepConv(channels, channels, 3, bias=True)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels + 1, hidden, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_size = (
            max(1, low.shape[-2] // 8),
            max(1, low.shape[-1] // 8),
        )
        low_grid = deterministic_adaptive_avg_pool2d(low, gate_size)
        high_grid = deterministic_adaptive_avg_pool2d(high, gate_size)
        high_magnitude = deterministic_adaptive_avg_pool2d(
            high.abs(),
            gate_size,
        ).mean(dim=1, keepdim=True)
        coherence = (high_grid.abs().mean(dim=1, keepdim=True) / (high_magnitude + 1e-6)).clamp(
            0.0, 1.0
        )
        gate = torch.sigmoid(
            F.interpolate(
                self.gate(
                    torch.cat(
                        [low_grid, high_grid, coherence],
                        dim=1,
                    )
                ),
                low.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        delta = torch.sigmoid(self.scale) * gate * torch.tanh(self.transport(high - low))
        return low + delta, high - delta


class ConditionedComplementaryFrequencyBlock(nn.Module):
    """Analyze and exchange complementary responses under degradation conditions."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        channels = feature_channels

        def branch() -> nn.Sequential:
            return nn.Sequential(
                SepConv(channels, channels, 3, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.frequency = nn.ModuleDict(
            {
                "low_branch": branch(),
                "high_branch": branch(),
                "high_gate": _make_statistics_gate(channels),
                "band_fuse": nn.Sequential(
                    nn.Conv2d(2 * channels, channels, 1, bias=True),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
                "band_gate": _make_statistics_gate(channels),
                "post": SepConv(channels, channels, 3, bias=True),
                "partition": DualSupportFrequencyPartition(channels),
                "router": SumPreservingFrequencyExchange(channels),
            }
        )
        self.frequency.register_parameter(
            "scale",
            nn.Parameter(torch.tensor(-0.7)),
        )
        self.refine = nn.ModuleDict(
            {
                "local1": BasicBlock(channels, channels, 3),
                "local2": BasicBlock(channels, channels, 3),
            }
        )

    def _restore_bands(
        self,
        x: torch.Tensor,
        degradation_statistics: torch.Tensor,
        degradation_map: torch.Tensor,
    ) -> torch.Tensor:
        low = self.frequency.partition(x, degradation_statistics, degradation_map)
        low_feature = self.frequency.low_branch(low)
        high_feature = _apply_statistics_gate(
            self.frequency.high_gate,
            self.frequency.high_branch(x - low),
        )
        low_feature, high_feature = self.frequency.router(
            low_feature,
            high_feature,
        )
        restored = self.frequency.band_fuse(torch.cat([low_feature, high_feature], dim=1))
        restored = _apply_statistics_gate(self.frequency.band_gate, restored)
        return 0.5 * torch.sigmoid(self.frequency.scale) * self.frequency.post(restored)

    def forward(
        self,
        x: torch.Tensor,
        degradation_statistics: torch.Tensor,
        degradation_map: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        if self.training and x.requires_grad:
            frequency_update = checkpoint(
                self._restore_bands,
                x,
                degradation_statistics,
                degradation_map,
                use_reentrant=False,
            )
        else:
            frequency_update = self._restore_bands(
                x,
                degradation_statistics,
                degradation_map,
            )
        x = self.refine.local2(self.refine.local1(x + frequency_update))
        return 0.5 * (x + residual)
