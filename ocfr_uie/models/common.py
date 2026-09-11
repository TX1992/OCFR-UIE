"""Reusable lightweight operators used by OCFR-UIE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def deterministic_reflect_pad2d(
    x: torch.Tensor, padding: int | tuple[int, int, int, int]
) -> torch.Tensor:
    """Reflection-pad BCHW tensors using deterministic slice operations."""

    if isinstance(padding, int):
        left = right = top = bottom = padding
    else:
        if len(padding) != 4:
            raise ValueError("2-D padding must contain left, right, top, bottom")
        left, right, top, bottom = padding
    if min(left, right, top, bottom) < 0:
        raise ValueError("reflection padding must be non-negative")
    height, width = x.shape[-2:]
    if left >= width or right >= width or top >= height or bottom >= height:
        raise ValueError("reflection padding must be smaller than the input extent")

    horizontal = [x[..., 1 : left + 1].flip(-1)] if left else []
    horizontal.append(x)
    if right:
        horizontal.append(x[..., width - right - 1 : width - 1].flip(-1))
    x = torch.cat(horizontal, dim=-1)

    vertical = [x[..., 1 : top + 1, :].flip(-2)] if top else []
    vertical.append(x)
    if bottom:
        vertical.append(x[..., height - bottom - 1 : height - 1, :].flip(-2))
    return torch.cat(vertical, dim=-2)


class _DeterministicAdaptiveAvgPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        output_height, output_width = output_size
        input_height, input_width = x.shape[-2:]
        ctx.input_height = input_height
        ctx.input_width = input_width
        ctx.output_height = output_height
        ctx.output_width = output_width
        ctx.block_height = input_height // output_height
        ctx.block_width = input_width // output_width
        ctx.non_overlapping = input_height % output_height == 0 and input_width % output_width == 0
        return F.adaptive_avg_pool2d(x, output_size)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        if ctx.non_overlapping:
            gradient = gradient.repeat_interleave(ctx.block_height, dim=-2)
            gradient = gradient.repeat_interleave(ctx.block_width, dim=-1)
            return gradient / float(ctx.block_height * ctx.block_width), None

        input_gradient = gradient.new_zeros(*gradient.shape[:-2], ctx.input_height, ctx.input_width)
        for output_y in range(ctx.output_height):
            input_y_start = (output_y * ctx.input_height) // ctx.output_height
            input_y_end = (
                (output_y + 1) * ctx.input_height + ctx.output_height - 1
            ) // ctx.output_height
            for output_x in range(ctx.output_width):
                input_x_start = (output_x * ctx.input_width) // ctx.output_width
                input_x_end = (
                    (output_x + 1) * ctx.input_width + ctx.output_width - 1
                ) // ctx.output_width
                area = (input_y_end - input_y_start) * (input_x_end - input_x_start)
                contribution = gradient[..., output_y, output_x, None, None] / float(area)
                input_gradient[..., input_y_start:input_y_end, input_x_start:input_x_end] += (
                    contribution
                )
        return input_gradient, None


def deterministic_adaptive_avg_pool2d(
    x: torch.Tensor, output_size: int | tuple[int, int]
) -> torch.Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return _DeterministicAdaptiveAvgPool2d.apply(x, output_size)


class SepConv(nn.Module):
    """Depthwise convolution followed by pointwise projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv1(x)
        output = self.conv2(output)
        return output


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class BasicBlock(nn.Module):
    """Half-instance-normalized residual block from the lightweight backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        relu_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("BasicBlock requires an even output width")
        self.identity = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_1 = SepConv(in_channels, out_channels, kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_channels, out_channels, kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_channels // 2, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(x)
        first, second = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(first), second], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out = out + self.identity(x)
        return out


class Downsample(nn.Module):
    """Convolutional projection followed by 2x pixel unshuffle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Convolutional projection followed by 2x pixel shuffle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
