# OCFR-UIE

Minimal training and evaluation code for UIEB and LSUI. The package contains
the fixed train/validation/test splits and the final checkpoints reported in
the paper. It does not include ablation or auxiliary experiment code.

## Installation

```bash
pip install -r requirements.txt
```

Training additionally requires the official DINOv2 repository and the
`dinov2_vitb14_pretrain.pth` checkpoint. Uranker weights are obtained by
`pyiqa` on first use.

## Evaluation

```bash
python test.py --dataset UIEB --data-root /path/to/UIEB
python test.py --dataset LSUI --data-root /path/to/LSUI
```

The commands use the verified evaluation batch size for each dataset (1 for
UIEB and 24 for LSUI). Predictions are evaluated after conversion to 8-bit
RGB, matching the reported paired-image results:

| Dataset | PSNR | SSIM |
|---|---:|---:|
| UIEB | 24.0553 | 0.9139 |
| LSUI | 30.3819 | 0.9332 |

Add `--save-images` to retain restored PNG files.

## Training

The command runs paired training followed by the short quality-refinement
stage used for the released model.

```bash
python train.py \
  --config configs/uieb.json \
  --data-root /path/to/UIEB \
  --dinov2-repo /path/to/dinov2 \
  --dinov2-checkpoint /path/to/dinov2_vitb14_pretrain.pth

python train.py \
  --config configs/lsui.json \
  --data-root /path/to/LSUI \
  --dinov2-repo /path/to/dinov2 \
  --dinov2-checkpoint /path/to/dinov2_vitb14_pretrain.pth
```

Dataset roots must match the relative paths in `splits/UIEB` or
`splits/LSUI`. Training writes only `paired_best.pth`, `final.pth`, and the
resolved configuration to the selected output directory.
