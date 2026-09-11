"""Frequency-adaptive dual-path decoding used by OCFR-UIE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BasicBlock, SimpleGate, deterministic_adaptive_avg_pool2d


class FrequencyAdaptiveDualPathDecoder(nn.Module):
    """Blend detail and context paths using high-frequency evidence."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 8)
        self.detail_candidate = BasicBlock(channels, channels, 3)
        self.context_candidate = nn.ModuleDict(
            {
                "expand": nn.Conv2d(channels, 2 * channels, 1, bias=True),
                "local": nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=1,
                    groups=channels,
                    bias=True,
                ),
                "dilated": nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=2,
                    dilation=2,
                    groups=channels,
                    bias=True,
                ),
                "gate": SimpleGate(),
                "project": nn.Conv2d(channels, channels, 1, bias=True),
            }
        )
        self.channel_reliability = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        # Retain the released state-dict key for strict checkpoint compatibility.
        self.structure_weights = nn.Parameter(torch.zeros(3))

    def spatial_reliability(self, high: torch.Tensor) -> torch.Tensor:
        coarse = F.interpolate(
            F.avg_pool2d(high, kernel_size=2, stride=2),
            high.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse_magnitude = coarse.abs().mean(dim=1, keepdim=True)
        inconsistency = (high - coarse).abs().mean(dim=1, keepdim=True)
        correlation = (high * coarse).mean(dim=1, keepdim=True)
        correlation = correlation / torch.sqrt(
            high.square().mean(dim=1, keepdim=True) * coarse.square().mean(dim=1, keepdim=True)
            + 1e-6
        )
        global_magnitude = coarse_magnitude.mean(dim=(2, 3), keepdim=True)
        factors = torch.cat(
            [
                coarse_magnitude / (coarse_magnitude + global_magnitude + 1e-6),
                coarse_magnitude / (coarse_magnitude + inconsistency + 1e-6),
                0.5 * (correlation.clamp(-1.0, 1.0) + 1.0),
            ],
            dim=1,
        ).clamp_min(1e-6)
        weights = torch.softmax(self.structure_weights, dim=0).view(1, 3, 1, 1)
        return torch.exp((weights * torch.log(factors)).sum(dim=1, keepdim=True))

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        local, dilated = self.context_candidate.expand(x).chunk(2, dim=1)
        local = self.context_candidate.local(local)
        dilated = self.context_candidate.dilated(dilated)
        context = self.context_candidate.gate(torch.cat([local, dilated], dim=1))
        return x + self.context_candidate.project(context)

    def forward(self, x: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        if x.shape != high.shape:
            raise ValueError("decoder feature and high-band evidence must have equal shapes")
        detail = self.detail_candidate(x)
        context = self._context(x)
        spatial = self.spatial_reliability(high)
        descriptor = torch.cat(
            [
                deterministic_adaptive_avg_pool2d(high.abs(), 1),
                high.abs().amax(dim=(2, 3), keepdim=True),
            ],
            dim=1,
        )
        channel = self.channel_reliability(descriptor)
        reliability = 0.5 * (spatial + channel)
        return reliability * detail + (1.0 - reliability) * context
