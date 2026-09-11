# Released Results

This file records the paper results at `256 x 256`. UCIQE denotes the original CIELab formulation.

## UIEB

| Split | PSNR | SSIM | Uranker | UIQM | UCIQE | MUSIQ | PAQ2PIQ | NIQE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation (89) | 24.7211 | 0.9234 | 2.2309 | 2.5282 | 0.6136 | 45.5287 | 73.7302 | 4.8061 |
| Test (89) | 24.0553 | 0.9139 | 2.3090 | 2.5701 | 0.6222 | 46.8160 | 74.6380 | 4.4241 |

## LSUI

| Split | PSNR | SSIM | Uranker | UIQM | UCIQE | MUSIQ | PAQ2PIQ | NIQE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation (428) | 30.5600 | 0.9312 | 2.0276 | - | - | - | - | - |
| Test (428) | 30.3819 | 0.9332 | 1.9444 | 2.5238 | 0.5956 | 44.5823 | 72.0587 | 5.0451 |

## EUVP

| Dataset split | PSNR | SSIM |
|---|---:|---:|
| EUVP-Dark test (555) | 22.8009 | 0.9124 |
| EUVP-Scene test (219) | 29.6470 | 0.9060 |

## Complexity

| Parameters | FLOPs at 256 | Mean latency | Throughput |
|---:|---:|---:|---:|
| 0.6238M | 5.3908G | 12.4956 ms | 80.03 images/s |

Latency was measured with batch size 1 using CUDA events on an RTX 4090. The optical-condition
encoder is included in all complexity values.
