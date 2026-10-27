#!/usr/bin/env python
"""Batch evaluation of MOLEXA on the held-out test set.

The evaluator mirrors the repository's test-data transform and model setup,
while keeping all implementation in this new file.  It does not modify the
model, dataset, or checkpoint.  The test loader is Windows-safe and keeps the
authors' ``ptc1``/``ptc2`` ordering convention.

The paper reports:
  * MAE: mean absolute coordinate error;
  * DE: absolute error of every inter-atomic distance (all i < j pairs);
  * AE: absolute error of every angle i-j-k (all distinct atoms, with j as
    the centre; swapping i and k gives the same angle).

The paper's headline numbers are stored in the comparison output.  Because
the paper describes DE/AE in terms of per-molecule pairs/triplets but does not
provide an executable aggregation function, this script writes both macro
(mean of per-sample means/maxima) and micro (pooled pair/triplet) values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ATOM_TO_Z = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
}

PAPER_REFERENCE = {
    "less_than_8_atoms": {
        "mae_au": 0.52,
        "de_mean_au": 0.98,
        "de_max_au": 2.11,
        "ae_mean_deg": 13.97,
        "ae_max_deg": 38.39,
    },
    "8_or_9_atoms": {
        "mae_au": 0.66,
        "de_mean_au": 1.16,
        "de_max_au": 3.43,
        "ae_mean_deg": 14.12,
        "ae_max_deg": 57.91,
    },
}


def _finite_mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


class TestSet:
    """Windows-safe equivalent of ``utils.DataLoaders_noSplit`` for test data."""

    def __init__(self, data_dir: Path, max_samples: int | None = None):
        self.frames: dict[Path, pd.DataFrame] = {}
        self.entries: list[tuple[Path, str, int, object]] = []

        pkl_paths = sorted(data_dir.rglob("*.pkl"))
        for pkl_path in pkl_paths:
            frame = pd.read_pickle(pkl_path)
            self.frames[pkl_path] = frame
            molecule_type = pkl_path.stem
            for row_position, row_label in enumerate(frame.index):
                self.entries.append((pkl_path, molecule_type, row_position, row_label))

        if max_samples is not None and max_samples > 0:
            self.entries = self.entries[:max_samples]
        if not self.entries:
            raise FileNotFoundError(f"No test .pkl samples found under {data_dir}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Data:
        pkl_path, molecule_type, row_position, _ = self.entries[index]
        row = self.frames[pkl_path].iloc[row_position]
        num_atoms = int(molecule_type.split("_")[0])

        # This is the same ordering used by TransformPd_multInput in the
        # author's loader: ptc1 and ptc2 first, then the remaining atoms.
        ptc1 = int(row["ptc1"])
        ptc2 = int(row["ptc2"])
        indexes = [ptc1, ptc2] + [
            atom_index
            for atom_index in range(1, num_atoms + 1)
            if atom_index not in (ptc1, ptc2)
        ]

        atoms = [ATOM_TO_Z[row[f"atom_{atom_index}"]] for atom_index in indexes]
        charges = [int(row[f"q_{atom_index}"]) for atom_index in indexes]
        coordinates = [
            [row[f"{axis}_{atom_index}"] for axis in ("x", "y", "z")]
            for atom_index in indexes
        ]
        momenta = [
            [row[f"{axis}_{atom_index}"] for axis in ("px", "py", "pz")]
            for atom_index in indexes
        ]

        y = torch.tensor(coordinates, dtype=torch.float32)
        y[torch.isnan(y)] = 0
        return Data(
            z=torch.tensor(atoms, dtype=torch.long),
            q=torch.tensor(charges, dtype=torch.long),
            y=y,
            pos=torch.tensor(momenta, dtype=torch.float32),
            natoms=torch.tensor([num_atoms], dtype=torch.long),
            sample_index=torch.tensor([index], dtype=torch.long),
        )


def pair_errors(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float, float, int]:
    """Return mean, maximum, sum, and count of all pairwise distance errors."""
    if pred.size(0) < 2:
        return math.nan, math.nan, 0.0, 0
    errors = (torch.pdist(pred) - torch.pdist(truth)).abs()
    return float(errors.mean()), float(errors.max()), float(errors.sum()), int(errors.numel())


def angle_errors(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float, float, int]:
    """Return mean, maximum, sum, and count of all i-j-k angle errors."""
    values: list[torch.Tensor] = []
    n_atoms = pred.size(0)
    for center in range(n_atoms):
        endpoints = [index for index in range(n_atoms) if index != center]
        for left_position, left in enumerate(endpoints):
            for right in endpoints[left_position + 1 :]:
                pred_left = pred[left] - pred[center]
                pred_right = pred[right] - pred[center]
                truth_left = truth[left] - truth[center]
                truth_right = truth[right] - truth[center]

                pred_norm = torch.linalg.vector_norm(pred_left) * torch.linalg.vector_norm(pred_right)
                truth_norm = torch.linalg.vector_norm(truth_left) * torch.linalg.vector_norm(truth_right)
                if float(pred_norm) < 1e-12 or float(truth_norm) < 1e-12:
                    continue

                pred_cos = torch.clamp(torch.dot(pred_left, pred_right) / pred_norm, -1.0, 1.0)
                truth_cos = torch.clamp(torch.dot(truth_left, truth_right) / truth_norm, -1.0, 1.0)
                values.append((torch.acos(pred_cos) - torch.acos(truth_cos)).abs())

    if not values:
        return math.nan, math.nan, 0.0, 0
    errors = torch.stack(values) * (180.0 / math.pi)
    return float(errors.mean()), float(errors.max()), float(errors.sum()), int(errors.numel())


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_model(source_dir: Path, checkpoint_path: Path, device: torch.device):
    """Construct the author's model and load a .chk/.pth checkpoint."""
    source_dir = source_dir.resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from utils.misc import GetModel

    error_bins = torch.arange(0, 10, 0.05, device=device)
    model, _ = GetModel(
        device=device,
        err_bin_center=error_bins,
        denoise=True,
        sample=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif hasattr(checkpoint, "state_dict"):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")

    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"checkpoint warning: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    model.to(device)
    model.eval()
    return model


def sample_metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int | None]:
    pred = pred.to(torch.float64)
    truth = truth.to(torch.float64)
    coordinate_errors = (pred - truth).abs()
    de_mean, de_max, de_sum, de_count = pair_errors(pred, truth)
    ae_mean, ae_max, ae_sum, ae_count = angle_errors(pred, truth)
    return {
        "n_atoms": int(pred.size(0)),
        "mae_au": float(coordinate_errors.mean()),
        "coordinate_abs_sum": float(coordinate_errors.sum()),
        "coordinate_count": int(coordinate_errors.numel()),
        "de_mean_au": de_mean,
        "de_max_au": de_max,
        "de_abs_sum": de_sum,
        "de_count": de_count,
        "ae_mean_deg": ae_mean,
        "ae_max_deg": ae_max,
        "ae_abs_sum": ae_sum,
        "ae_count": ae_count,
    }


def aggregate(rows: list[dict], label: str) -> dict[str, float | int | str | None]:
    coordinate_sum = sum(float(row["coordinate_abs_sum"]) for row in rows)
    coordinate_count = sum(int(row["coordinate_count"]) for row in rows)
    de_sum = sum(float(row["de_abs_sum"]) for row in rows)
    de_count = sum(int(row["de_count"]) for row in rows)
    ae_sum = sum(float(row["ae_abs_sum"]) for row in rows)
    ae_count = sum(int(row["ae_count"]) for row in rows)

    return {
        "group": label,
        "samples": len(rows),
        "mae_macro_au": _finite_mean(row["mae_au"] for row in rows),
        "mae_micro_au": coordinate_sum / coordinate_count if coordinate_count else None,
        "de_macro_mean_au": _finite_mean(row["de_mean_au"] for row in rows),
        "de_micro_mean_au": de_sum / de_count if de_count else None,
        "de_macro_max_au": _finite_mean(row["de_max_au"] for row in rows),
        "de_global_max_au": max(
            (
                float(row["de_max_au"])
                for row in rows
                if row["de_max_au"] is not None
                and math.isfinite(float(row["de_max_au"]))
            ),
            default=None,
        ),
        "ae_macro_mean_deg": _finite_mean(row["ae_mean_deg"] for row in rows),
        "ae_micro_mean_deg": ae_sum / ae_count if ae_count else None,
        "ae_macro_max_deg": _finite_mean(row["ae_max_deg"] for row in rows),
        "ae_global_max_deg": max(
            (
                float(row["ae_max_deg"])
                for row in rows
                if row["ae_max_deg"] is not None
                and math.isfinite(float(row["ae_max_deg"]))
            ),
            default=None,
        ),
    }


def comparison_rows(aggregates: list[dict]) -> list[dict]:
    rows = []
    mapping = {
        "mae_macro_au": "mae_au",
        "de_macro_mean_au": "de_mean_au",
        "de_macro_max_au": "de_max_au",
        "ae_macro_mean_deg": "ae_mean_deg",
        "ae_macro_max_deg": "ae_max_deg",
    }
    for aggregate_row in aggregates:
        reference = PAPER_REFERENCE.get(aggregate_row["group"])
        if reference is None:
            continue
        for reproduced_key, paper_key in mapping.items():
            reproduced = aggregate_row[reproduced_key]
            paper_value = reference[paper_key]
            rows.append({
                "group": aggregate_row["group"],
                "metric": reproduced_key,
                "reproduced": reproduced,
                "paper": paper_value,
                "difference": reproduced - paper_value if reproduced is not None else None,
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="source_code/molexa")
    parser.add_argument("--data-dir", type=Path, required=True, help="test directory containing .pkl files")
    parser.add_argument("--checkpoint", type=Path, required=True, help="molexa.chk or compatible .pth")
    parser.add_argument("--output-dir", type=Path, required=True, help="separate result directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=0, help="0 means all test samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = TestSet(args.data_dir, max_samples=args.test_samples or None)
    # Batch size one preserves per-molecule boundaries, matches the notebook,
    # and keeps peak memory reasonable on an 8 GB RTX 4060.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(args.source_dir, args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict] = []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            sample_indices = batch.sample_index.reshape(-1).tolist()
            batch = batch.to(device)
            prediction = model(batch).reshape(-1, 3).cpu()
            truth = batch.y.reshape(-1, 3).cpu()
            pointers = batch.ptr.cpu().tolist()
            for local_index, sample_index in enumerate(sample_indices):
                start, end = pointers[local_index], pointers[local_index + 1]
                metrics = sample_metrics(prediction[start:end], truth[start:end])
                pkl_path, molecule_type, row_position, row_label = dataset.entries[int(sample_index)]
                sample_rows.append({
                    "sample_index": int(sample_index),
                    "file": str(pkl_path),
                    "molecule_type": molecule_type,
                    "row_position": row_position,
                    "row_label": row_label,
                    **metrics,
                })
            if batch_number % 25 == 0:
                print(f"evaluated {batch_number}/{len(loader)} test samples")

    sample_fields = [
        "sample_index", "file", "molecule_type", "row_position", "row_label", "n_atoms",
        "mae_au", "de_mean_au", "de_max_au", "ae_mean_deg", "ae_max_deg",
    ]
    with (args.output_dir / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in sample_fields} for row in sample_rows)

    groups = [
        ("overall", sample_rows),
        ("less_than_8_atoms", [row for row in sample_rows if row["n_atoms"] < 8]),
        ("8_or_9_atoms", [row for row in sample_rows if row["n_atoms"] in (8, 9)]),
    ]
    aggregate_rows = [aggregate(rows, label) for label, rows in groups]
    aggregate_fields = list(aggregate_rows[0])
    with (args.output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    comparison = comparison_rows(aggregate_rows)
    with (args.output_dir / "paper_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "metric", "reproduced", "paper", "difference"])
        writer.writeheader()
        writer.writerows(comparison)

    summary = {
        "device": str(device),
        "seed": args.seed,
        "test_samples": len(sample_rows),
        "source_dir": str(args.source_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "aggregates": aggregate_rows,
        "paper_reference": PAPER_REFERENCE,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
