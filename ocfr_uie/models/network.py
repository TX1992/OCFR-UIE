"""Complete OCFR-UIE network."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import Downsample, Upsample
from .frequency_analysis import (
    ConditionedComplementaryFrequencyBlock,
    HaarFrequencyInteractionBlock,
)
from .frequency_reconstruction import OpticalStateGuidedFrequencyReconstruction
from .optical_condition import UnderwaterOpticalConditionEncoder


class OCFRUIE(nn.Module):
    """Optically conditioned frequency reconstruction for RGB enhancement."""

    architecture_version = "ocfr-uie-v1"
    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 28,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("OCFR-UIE accepts RGB input only")
        if feature_channels % 4:
            raise ValueError("feature_channels must be divisible by four")
        self.in_channels = int(in_channels)
        self.feature_channels = int(feature_channels)

        channels = self.feature_channels
        self.condition_encoder = UnderwaterOpticalConditionEncoder(channels)
        self.stem = nn.Conv2d(in_channels, channels, 3, 1, 1)

        self.encoder1 = HaarFrequencyInteractionBlock(channels)
        self.down1 = Downsample(channels)
        self.encoder2 = ConditionedComplementaryFrequencyBlock(2 * channels)
        self.down2 = Downsample(2 * channels)
        self.bottleneck = ConditionedComplementaryFrequencyBlock(4 * channels)

        self.up1 = Upsample(4 * channels)
        self.decoder1 = HaarFrequencyInteractionBlock(
            2 * channels,
            structure_refinement=True,
        )
        self.reconstruction = OpticalStateGuidedFrequencyReconstruction(channels)
        self.decoder2 = HaarFrequencyInteractionBlock(
            channels,
            structure_refinement=True,
        )
        self.output = nn.Conv2d(channels, in_channels, 3, 1, 1)

    @staticmethod
    def _validate_input(raw: torch.Tensor) -> None:
        if raw.ndim != 4 or raw.shape[1] != 3:
            raise ValueError("OCFR-UIE expects a BCHW RGB tensor")
        if raw.shape[-2] % 4 or raw.shape[-1] % 4:
            raise ValueError("input height and width must be divisible by four")

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        self._validate_input(raw)
        stem = self.stem(raw)
        shallow, degradation_statistics, degradation_map, optical_state = self.condition_encoder(
            raw,
            stem,
        )
        encoder_1 = self.encoder1(shallow)
        encoder_2 = self.encoder2(
            self.down1(encoder_1),
            degradation_statistics,
            degradation_map,
        )
        bottleneck = self.bottleneck(
            self.down2(encoder_2),
            degradation_statistics,
            degradation_map,
        )

        deep = self.decoder1(self.up1(bottleneck) + encoder_2)
        reconstructed = self.reconstruction(deep, encoder_1, optical_state)
        return self.output(self.decoder2(reconstructed)) + raw


def build_model(feature_channels: int = 28) -> OCFRUIE:
    return OCFRUIE(feature_channels=feature_channels)
