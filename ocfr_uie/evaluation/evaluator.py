import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .metrics import (
    DEFAULT_METRICS,
    PYIQA_METRICS,
    build_pyiqa_metrics,
    get_psnr,
    get_ssim,
    get_uciqe,
    get_uiqm,
    normalize_metric_names,
    resize_for_iqa,
    to_numpy_batch,
)


def mean_dict(rows: Sequence[Mapping[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and row[key] != ""]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def summarize_rows(rows: Sequence[Mapping[str, object]], metrics: Sequence[str] = DEFAULT_METRICS):
    keys = [key for key in metrics if any(key in row for row in rows)]
    by_dataset: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "all")), []).append(row)
    per_set = {dataset: mean_dict(items, keys) for dataset, items in sorted(by_dataset.items())}
    overall = {
        key: float(np.mean([per_set[d][key] for d in sorted(per_set) if key in per_set[d]]))
        for key in keys
    }
    sample_avg = mean_dict(rows, keys)
    return overall, sample_avg, per_set


def format_metrics(values: Mapping[str, float], metrics: Sequence[str] = DEFAULT_METRICS) -> str:
    labels = {
        "uranker": "Uranker",
        "musiq": "MUSIQ",
        "paq2piq": "PAQ2PIQ",
        "niqe": "NIQE",
        "uiqm": "UIQM",
        "uciqe": "UCIQE",
        "psnr": "PSNR",
        "ssim": "SSIM",
    }
    parts = []
    for key in metrics:
        if key in values:
            parts.append(f"{labels.get(key, key)}={float(values[key]):.4f}")
    return ", ".join(parts)


class UnifiedEvaluator:
    def __init__(
        self,
        metrics: Optional[Sequence[str] | str] = None,
        device: str | torch.device = "cuda",
        iqa_size: Optional[int] = None,
        pyiqa_metrics: Optional[Mapping[str, object]] = None,
    ):
        self.metrics = normalize_metric_names(metrics)
        self.device = torch.device(
            device if torch.cuda.is_available() or str(device) == "cpu" else "cpu"
        )
        self.iqa_size = int(iqa_size) if iqa_size else None
        needed_pyiqa = [name for name in self.metrics if name in PYIQA_METRICS]
        self.pyiqa_metrics = dict(pyiqa_metrics or {})
        missing = [name for name in needed_pyiqa if name not in self.pyiqa_metrics]
        if missing:
            self.pyiqa_metrics.update(build_pyiqa_metrics(missing, self.device))

    @torch.no_grad()
    def evaluate_tensor_batch(
        self,
        pred: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
        datasets: Optional[Sequence[str]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        pred = pred.detach().float().clamp(0.0, 1.0)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        pred_for_iqa = resize_for_iqa(pred.to(self.device), self.iqa_size)

        gt_batch = None
        if gt is not None:
            gt_batch = gt.detach().float().clamp(0.0, 1.0)
            if gt_batch.dim() == 3:
                gt_batch = gt_batch.unsqueeze(0)

        batch_scores: Dict[str, np.ndarray] = {}
        for name, metric in self.pyiqa_metrics.items():
            if name in self.metrics:
                score = metric(pred_for_iqa).float().reshape(pred_for_iqa.size(0), -1).mean(dim=1)
                batch_scores[name] = score.detach().cpu().numpy().astype(np.float64)

        pred_np = to_numpy_batch(pred)
        gt_np = to_numpy_batch(gt_batch) if gt_batch is not None else None
        rows: List[Dict[str, object]] = []
        for idx in range(pred_np.shape[0]):
            row: Dict[str, object] = {
                "dataset": str(datasets[idx]) if datasets is not None else "all",
                "name": str(names[idx]) if names is not None else str(idx),
            }
            for metric_name, values in batch_scores.items():
                row[metric_name] = float(values[idx])
            if "uiqm" in self.metrics:
                row["uiqm"] = get_uiqm(pred_np[idx])
            if "uciqe" in self.metrics:
                row["uciqe"] = get_uciqe(pred_np[idx])
            if gt_np is not None:
                if "psnr" in self.metrics:
                    row["psnr"] = get_psnr(pred_np[idx], gt_np[idx])
                if "ssim" in self.metrics:
                    row["ssim"] = get_ssim(pred_np[idx], gt_np[idx])
            rows.append(row)
        return rows

    def summarize(self, rows: Sequence[Mapping[str, object]]):
        return summarize_rows(rows, self.metrics)


def write_evaluation_outputs(out_dir: str | Path, rows, overall, sample_avg, per_set):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "name", "input_path", "pred_path", "gt_path"] + [
        key for key in DEFAULT_METRICS if any(key in row for row in rows)
    ]
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    summary = {"overall": overall, "sample_avg": sample_avg, "per_set": per_set}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = ["scope", "dataset"] + [
            key for key in DEFAULT_METRICS if key in overall or key in sample_avg
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"scope": "overall", "dataset": "mean_by_dataset", **overall})
        writer.writerow({"scope": "sample_avg", "dataset": "all_samples", **sample_avg})
        for dataset, values in per_set.items():
            writer.writerow({"scope": "per_set", "dataset": dataset, **values})
    return out_dir
