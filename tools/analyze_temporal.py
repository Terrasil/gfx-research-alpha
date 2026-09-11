#!/usr/bin/env python3
"""Temporal residual analysis for the controlled RODD motion sequences.

Each rendered pose is compared with its own depth-peeling reference first.
Temporal metrics are then computed on that residual sequence. This avoids
counting legitimate scene motion as reconstruction flicker.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REFERENCE = "depth-peeling-reference-24"
FRAME_RE = re.compile(r"-f(\d+)$")


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.readline().decode("ascii").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Unsupported PFM header in {path}: {header}")
        width, height = map(int, stream.readline().decode("ascii").split())
        scale = float(stream.readline().decode("ascii").strip())
        endian = "<" if scale < 0.0 else ">"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(stream, endian + "f4", width * height * channels)
    if data.size != width * height * channels:
        raise ValueError(f"Truncated PFM: {path}")
    return data.reshape(height, width, channels).astype(np.float64, copy=False)


def luminance(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] == 1:
        return image[..., 0]
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def rmse(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else math.nan


def sequence_metrics(method_frames: list[np.ndarray], reference_frames: list[np.ndarray]) -> dict[str, float]:
    method = np.stack(method_frames, axis=0)
    reference = np.stack(reference_frames, axis=0)
    residual = method - reference
    if residual.shape[0] < 2:
        return {}

    residual_delta = np.diff(residual, axis=0)
    reference_delta = np.diff(reference, axis=0)
    residual_variance = np.var(residual, axis=0)
    spectrum = np.fft.rfft(residual - np.mean(residual, axis=0, keepdims=True), axis=0)
    power = np.abs(spectrum) ** 2
    non_dc_power = float(np.mean(np.sum(power[1:], axis=0))) if power.shape[0] > 1 else 0.0
    total_residual_energy = float(np.mean(residual * residual))

    return {
        "mean_spatial_residual_rmse": rmse(residual),
        "residual_frame_delta_rmse": rmse(residual_delta),
        "residual_temporal_variance_rms": float(np.sqrt(np.mean(residual_variance))),
        "flicker_non_dc_energy": non_dc_power,
        "flicker_to_residual_energy": non_dc_power / max(total_residual_energy, 1e-24),
        "reference_frame_delta_rmse": rmse(reference_delta),
        "residual_delta_to_reference_motion": rmse(residual_delta) / max(rmse(reference_delta), 1e-12),
    }


def analyze(results: Path, output: Path) -> pd.DataFrame:
    quality_path = results / "quality.csv"
    captures = results / "captures"
    if not quality_path.exists():
        raise FileNotFoundError(quality_path)
    quality = pd.read_csv(quality_path)
    temporal = quality[quality["sweep"].isin(["temporal-camera", "temporal-object"])].copy()
    rows: list[dict] = []

    group_columns = ["sweep", "method", "dither"]
    if "rodd_variant" in temporal.columns:
        group_columns.append("rodd_variant")

    for group_values, group in temporal[temporal["method"] != REFERENCE].groupby(group_columns, dropna=False):
        group = group.copy()
        group["frame"] = group["pair_id"].map(lambda value: int(FRAME_RE.search(str(value)).group(1)) if FRAME_RE.search(str(value)) else -1)
        group = group[group["frame"] >= 0].sort_values("frame")
        method_frames: list[np.ndarray] = []
        reference_frames: list[np.ndarray] = []
        alpha_method_frames: list[np.ndarray] = []
        alpha_reference_frames: list[np.ndarray] = []

        for _, row in group.iterrows():
            pair_id = str(row["pair_id"])
            method_name = str(row["method"])
            method_path = captures / f"{pair_id}-{method_name}.pfm"
            reference_path = captures / f"{pair_id}-{REFERENCE}.pfm"
            if not method_path.exists() or not reference_path.exists():
                continue
            method_frames.append(luminance(read_pfm(method_path)))
            reference_frames.append(luminance(read_pfm(reference_path)))

            alpha_method_path = captures / f"{pair_id}-{method_name}-alpha-estimate.pfm"
            alpha_reference_path = captures / f"{pair_id}-alpha-reference.pfm"
            if alpha_method_path.exists() and alpha_reference_path.exists():
                alpha_method_frames.append(luminance(read_pfm(alpha_method_path)))
                alpha_reference_frames.append(luminance(read_pfm(alpha_reference_path)))

        if len(method_frames) < 2:
            continue
        metrics = sequence_metrics(method_frames, reference_frames)
        alpha_metrics = sequence_metrics(alpha_method_frames, alpha_reference_frames) if len(alpha_method_frames) == len(method_frames) else {}
        if isinstance(group_values, tuple):
            descriptor = dict(zip(group_columns, group_values))
        else:
            descriptor = {group_columns[0]: group_values}
        rows.append({
            **descriptor,
            "frames": len(method_frames),
            **metrics,
            "alpha_residual_frame_delta_rmse": alpha_metrics.get("residual_frame_delta_rmse", math.nan),
            "alpha_temporal_variance_rms": alpha_metrics.get("residual_temporal_variance_rms", math.nan),
            "alpha_flicker_to_residual_energy": alpha_metrics.get("flicker_to_residual_energy", math.nan),
        })

    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "temporal_summary.csv", index=False)

    if not result.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.7))
        labels = [f"{s}/{d}/{m}" for s, d, m in zip(result["sweep"], result["dither"], result["method"])]
        ax.bar(labels, result["residual_frame_delta_rmse"])
        ax.set_ylabel("Frame-to-frame residual RMSE")
        ax.set_title("Temporal reconstruction residual under controlled motion")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "temporal_residual_delta.png", dpi=190)
        plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results / "analysis"
    result = analyze(args.results, output)
    print(f"Analyzed {len(result)} temporal method sequences -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
