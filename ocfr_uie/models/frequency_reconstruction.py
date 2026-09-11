"""Optically guided Haar-frequency reconstruction with detail preservation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .optical_condition import RelativeOpticalStateEstimator, OpticalState


def _haar_matrices(channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    analysis = 0.5 * torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    analysis = analysis.repeat(channels, 1).reshape(
        4 * channels,
        4,
        1,
        1,
    )
    synthesis = (
        analysis.reshape(channels, 4, 4).transpose(1, 2).reshape(4 * channels, 4, 1, 1).contiguous()
    )
    return analysis, synthesis


class OpticalStateGuidedFrequencyReconstruction(nn.Module):
    """Fuse full-frequency and detail-preserving candidates before synthesis."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.feature_channels = int(feature_channels)
        channels = self.feature_channels
        state_dim = RelativeOpticalStateEstimator.state_dim

        analysis, synthesis = _haar_matrices(channels)
        self.register_buffer("haar_analysis", analysis, persistent=False)
        self.register_buffer("haar_synthesis", synthesis, persistent=False)

        self.band_correction = nn.Conv2d(
            2 * channels,
            4 * channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.low_selector = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            groups=channels,
            bias=True,
        )
        self.candidate_selector = nn.Conv2d(
            4 * channels,
            4,
            kernel_size=1,
            bias=True,
        )
        self.low_token = nn.Conv2d(state_dim, channels, 1, bias=False)
        self.low_range = nn.Conv2d(1, channels, 1, bias=False)
        self.candidate_token = nn.Conv2d(state_dim, 4, 1, bias=False)
        self.candidate_range = nn.Conv2d(1, 4, 1, bias=False)
        self._initialize_selectors()

    def _initialize_selectors(self) -> None:
        nn.init.zeros_(self.low_selector.weight)
        nn.init.zeros_(self.low_selector.bias)
        nn.init.zeros_(self.candidate_selector.weight)
        nn.init.zeros_(self.candidate_selector.bias)
        nn.init.zeros_(self.low_token.weight)
        nn.init.zeros_(self.low_range.weight)
        nn.init.zeros_(self.candidate_token.weight)
        nn.init.zeros_(self.candidate_range.weight)

    def analysis(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.feature_channels:
            raise ValueError("Haar analysis received an invalid feature tensor")
        if feature.shape[-2] % 2 or feature.shape[-1] % 2:
            raise ValueError("Haar analysis requires even spatial dimensions")
        components = F.pixel_unshuffle(feature, downscale_factor=2)
        bands = F.conv2d(
            components,
            self.haar_analysis,
            groups=self.feature_channels,
        )
        return bands.reshape(
            feature.shape[0],
            self.feature_channels,
            4,
            feature.shape[-2] // 2,
            feature.shape[-1] // 2,
        )

    def synthesis(self, bands: torch.Tensor) -> torch.Tensor:
        if bands.ndim != 5 or bands.shape[1] != self.feature_channels or bands.shape[2] != 4:
            raise ValueError("Haar synthesis received an invalid frequency-component tensor")
        components = F.conv2d(
            bands.flatten(1, 2),
            self.haar_synthesis,
            groups=self.feature_channels,
        )
        return F.pixel_shuffle(components, upscale_factor=2)

    @staticmethod
    def _range_map(
        state: OpticalState,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        q = state["q"]
        expected_size = (2 * spatial_size[0], 2 * spatial_size[1])
        if q.shape[-2:] != expected_size:
            raise ValueError("optical range map must be twice the reconstruction resolution")
        return F.avg_pool2d(q, kernel_size=2, stride=2)

    def forward(
        self,
        deep: torch.Tensor,
        skip: torch.Tensor,
        state: OpticalState,
    ) -> torch.Tensor:
        if deep.ndim != 4 or skip.ndim != 4:
            raise ValueError("reconstruction inputs must be BCHW tensors")
        if deep.shape[0] != skip.shape[0]:
            raise ValueError("deep and skip batches must match")
        if deep.shape[1] != 2 * self.feature_channels:
            raise ValueError("deep feature width does not match reconstruction")
        if skip.shape[1] != self.feature_channels:
            raise ValueError("skip feature width does not match reconstruction")
        if skip.shape[-2:] != (
            2 * deep.shape[-2],
            2 * deep.shape[-1],
        ):
            raise ValueError("skip resolution must be twice the deep resolution")

        encoder_bands = self.analysis(skip)
        correction = self.band_correction(deep).reshape_as(encoder_bands)
        spatial_candidate = encoder_bands + correction

        range_map = self._range_map(state, deep.shape[-2:])
        low_logits = (
            self.low_selector(correction[:, :, 0])
            + self.low_token(state["token"])
            + self.low_range(range_map)
        )
        low_weight = torch.sigmoid(low_logits)
        frequency_low = encoder_bands[:, :, 0] + low_weight * correction[:, :, 0]
        frequency_candidate = torch.cat(
            [
                frequency_low.unsqueeze(2),
                encoder_bands[:, :, 1:],
            ],
            dim=2,
        )

        difference = spatial_candidate - frequency_candidate
        candidate_logits = (
            self.candidate_selector(difference.flatten(1, 2))
            + self.candidate_token(state["token"])
            + self.candidate_range(range_map)
        )
        spatial_weight = 0.5 + 0.5 * torch.sigmoid(candidate_logits)
        fused_bands = frequency_candidate + spatial_weight.unsqueeze(1) * difference
        return self.synthesis(fused_bands)
