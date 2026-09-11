"""Underwater degradation and relative optical condition encoding."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import deterministic_adaptive_avg_pool2d


OpticalState = Dict[str, torch.Tensor]


class UnderwaterDegradationConditionEncoder(nn.Module):
    """Encode global and spatial degradation statistics for feature modulation."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 16,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        hidden_channels = max(8, hidden_channels)
        self.global_proj = nn.Sequential(
            nn.Conv2d(6, hidden_channels, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, feature_channels, 1, bias=True),
        )
        self.lowfreq_proj = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, feature_channels, 3, padding=1, bias=True),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(feature_channels, feature_channels, 1, bias=True)
        self.scale = nn.Parameter(torch.tensor(-1.2))

    def _build_condition(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = raw.shape
        mean_rgb = raw.mean(dim=(2, 3), keepdim=True)
        gray = mean_rgb.mean(dim=1, keepdim=True)
        balance = torch.clamp(gray / (mean_rgb + self.eps), 0.25, 4.0)
        global_condition = torch.cat([mean_rgb, balance], dim=1)
        low = deterministic_adaptive_avg_pool2d(raw, (max(1, height // 16), max(1, width // 16)))
        return global_condition, low

    def forward(
        self, raw: torch.Tensor, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_condition, low = self._build_condition(raw)
        global_modulation = self.global_proj(global_condition)
        low_modulation = self.lowfreq_proj(low)
        confidence = self.confidence(low)
        low_modulation = F.interpolate(
            low_modulation, size=feature.shape[-2:], mode="bilinear", align_corners=False
        )
        confidence = F.interpolate(
            confidence, size=feature.shape[-2:], mode="bilinear", align_corners=False
        )
        modulation = torch.tanh(global_modulation + confidence * low_modulation)
        modulation = self.out_proj(modulation)
        scale = 0.25 * torch.sigmoid(self.scale)
        output = feature * (1.0 + scale * modulation)
        return output, global_condition, confidence


class RelativeOpticalStateEstimator(nn.Module):
    """Analytically anchored and bounded underwater optical state."""

    state_dim = 11

    def __init__(self, hidden_channels: int = 12, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        hidden_channels = max(8, int(hidden_channels))
        self.global_correction = nn.Sequential(
            nn.Conv2d(9, hidden_channels, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 6, 1, bias=True),
        )
        self.q_correction = nn.Sequential(
            nn.Conv2d(6, hidden_channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        self._initialize_bounded_corrections()

    def _initialize_bounded_corrections(self) -> None:
        for layer in (self.global_correction[-1], self.q_correction[-1]):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.bias)

    def _ambient_prior(self, raw: torch.Tensor) -> torch.Tensor:
        grid = deterministic_adaptive_avg_pool2d(
            raw, (min(16, raw.shape[-2]), min(16, raw.shape[-1]))
        )
        dark = grid.amin(dim=1, keepdim=True)
        mean = dark.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt((dark - mean).square().mean(dim=(2, 3), keepdim=True) + self.eps)
        score = 4.0 * (dark - mean) / (std + self.eps)
        weight = torch.softmax(score.flatten(2), dim=-1).view_as(score)
        return (grid * weight).sum(dim=(2, 3), keepdim=True)

    def forward(self, raw: torch.Tensor) -> OpticalState:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError("RelativeOpticalStateEstimator expects BCHW RGB input")
        smooth = F.avg_pool2d(raw, kernel_size=7, stride=1, padding=3)
        red, green, blue = smooth[:, 0:1], smooth[:, 1:2], smooth[:, 2:3]
        luma = 0.299 * red + 0.587 * green + 0.114 * blue
        dark = smooth.amin(dim=1, keepdim=True)
        spread = smooth.amax(dim=1, keepdim=True) - dark
        red_deficit = F.relu(0.5 * (green + blue) - red)

        q_base = torch.clamp(
            0.60 * (1.0 - luma) + 0.25 * red_deficit + 0.15 * spread,
            0.0,
            1.0,
        )
        q_input = torch.cat([smooth, luma, dark, red_deficit], dim=1)
        q = torch.clamp(q_base + 0.10 * torch.tanh(self.q_correction(q_input)), 0.0, 1.0)

        ambient_prior = self._ambient_prior(raw)
        mean_rgb = raw.mean(dim=(2, 3), keepdim=True)
        std_rgb = torch.sqrt((raw - mean_rgb).square().mean(dim=(2, 3), keepdim=True) + self.eps)
        correction = self.global_correction(torch.cat([mean_rgb, std_rgb, ambient_prior], dim=1))
        ambient_delta, rho_logits = correction.chunk(2, dim=1)
        ambient = torch.clamp(ambient_prior + 0.05 * torch.tanh(ambient_delta), 0.0, 1.0)

        rho_sigmoid = torch.sigmoid(rho_logits)
        rho_b = 0.55 + 0.20 * rho_sigmoid[:, 2:3]
        rho_g = rho_b + 0.15 + 0.15 * rho_sigmoid[:, 1:2]
        rho_r = rho_g + 0.18 + 0.17 * rho_sigmoid[:, 0:1]
        rho = torch.cat([rho_r, rho_g, rho_b], dim=1)
        transmission = torch.exp(-rho * q)

        q_mean = q.mean(dim=(2, 3), keepdim=True)
        q_std = torch.sqrt((q - q_mean).square().mean(dim=(2, 3), keepdim=True) + self.eps)
        transmission_mean = transmission.mean(dim=(2, 3), keepdim=True)
        token = torch.cat([ambient, rho, q_mean, q_std, transmission_mean], dim=1)
        if token.shape[1] != self.state_dim:
            raise RuntimeError(f"invalid optical token width: {token.shape[1]}")
        return {
            "token": token,
            "q": q,
            "transmission": transmission,
            "ambient": ambient,
            "rho": rho,
        }


class UnderwaterOpticalConditionEncoder(nn.Module):
    """Jointly encode observed degradation and relative optical state."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.optical = RelativeOpticalStateEstimator()
        self.degradation = UnderwaterDegradationConditionEncoder(feature_channels)

    def forward(
        self,
        raw: torch.Tensor,
        stem_feature: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        OpticalState,
    ]:
        state = self.optical(raw)
        feature, degradation_vector, degradation_map = self.degradation(raw, stem_feature)
        degradation_mean = deterministic_adaptive_avg_pool2d(degradation_map, 1)
        degradation_variance = (degradation_map - degradation_mean).square().mean(
            dim=(2, 3), keepdim=True
        )
        degradation_statistics = torch.cat(
            [
                degradation_vector,
                degradation_mean,
                torch.sqrt(degradation_variance + 1e-6),
            ],
            dim=1,
        )
        return feature, degradation_statistics, degradation_map, state
