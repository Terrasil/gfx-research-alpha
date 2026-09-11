#!/usr/bin/env python3
"""Frequency-transfer and ordered-pattern residual analysis for captured HDR frames.

Run after --benchmark-suite/--benchmark-paper. The script uses the PFM captures
written by the renderer and their matched depth-peeling references. It reports
radial amplitude transfer H(f)=A_method(f)/A_reference(f), plus the fraction of
luminance error that is phase-locked to the 4x4 ordered pattern.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REFERENCE = "depth-peeling-reference-24"


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
    image = data.reshape(height, width, channels)
    return image.astype(np.float64, copy=False)


def luminance(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] == 1:
        return image[..., 0]
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def radial_amplitude(signal: np.ndarray, bins: int = 96) -> tuple[np.ndarray, np.ndarray]:
    signal = signal - float(np.mean(signal))
    window_y = np.hanning(signal.shape[0])
    window_x = np.hanning(signal.shape[1])
    window = window_y[:, None] * window_x[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(signal * window)))

    fy = np.fft.fftshift(np.fft.fftfreq(signal.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(signal.shape[1]))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    max_radius = 0.5
    edges = np.linspace(0.0, max_radius, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        if np.any(mask):
            values[index] = float(np.mean(spectrum[mask]))
    return centers, values


def phase_locked_residual_ratio(error: np.ndarray) -> float:
    total = float(np.sum(error * error))
    if total <= 1e-24:
        return 0.0
    phase_energy = 0.0
    for py in range(4):
        for px in range(4):
            phase = error[py::4, px::4]
            if phase.size == 0:
                continue
            mean = float(np.mean(phase))
            phase_energy += mean * mean * phase.size
    return float(np.clip(phase_energy / total, 0.0, 1.0))


def band_mean(freq: np.ndarray, transfer: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq < hi) & np.isfinite(transfer)
    return float(np.mean(transfer[mask])) if np.any(mask) else np.nan


def analyze(results: Path, output: Path) -> pd.DataFrame:
    quality_path = results / "quality.csv"
    captures = results / "captures"
    if not quality_path.exists():
        raise FileNotFoundError(quality_path)
    data = pd.read_csv(quality_path)
    rows: list[dict] = []
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []

    for _, row in data.iterrows():
        method = str(row["method"])
        if method == REFERENCE:
            continue
        pair_id = str(row["pair_id"])
        method_path = captures / f"{pair_id}-{method}.pfm"
        reference_path = captures / f"{pair_id}-{REFERENCE}.pfm"
        if not method_path.exists() or not reference_path.exists():
            continue

        image = luminance(read_pfm(method_path))
        reference = luminance(read_pfm(reference_path))
        if image.shape != reference.shape:
            continue

        freq, method_amp = radial_amplitude(image)
        _, reference_amp = radial_amplitude(reference)
        floor = max(float(np.nanmax(reference_amp)) * 1e-6, 1e-12)
        transfer = method_amp / np.maximum(reference_amp, floor)
        valid = reference_amp > floor
        transfer = np.where(valid, transfer, np.nan)
        error = image - reference

        variant = str(row.get("rodd_variant", "full"))
        label = f"{pair_id}/{method}/{variant}"
        rows.append({
            "pair_id": pair_id,
            "sweep": row.get("sweep", ""),
            "method": method,
            "rodd_variant": variant,
            "dither": row.get("dither", ""),
            "alpha": row.get("alpha", np.nan),
            "geometry": row.get("geometry", ""),
            "transfer_low_0_0p08": band_mean(freq, transfer, 0.0, 0.08),
            "transfer_mid_0p08_0p20": band_mean(freq, transfer, 0.08, 0.20),
            "transfer_high_0p20_0p50": band_mean(freq, transfer, 0.20, 0.50),
            "ordered_phase_residual_ratio": phase_locked_residual_ratio(error),
        })
        curves.append((label, freq, transfer))

    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "spectrum_summary.csv", index=False)

    focus = [curve for curve in curves if "rodd-experimental" in curve[0]]
    if focus:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for label, freq, transfer in focus[:18]:
            ax.plot(freq, transfer, linewidth=1.0, alpha=0.75, label=label)
        ax.axhline(1.0, linewidth=1.0)
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 1.8)
        ax.set_xlabel("Spatial frequency [cycles/pixel]")
        ax.set_ylabel("Amplitude transfer H(f)")
        ax.set_title("RODD radial frequency transfer from captured HDR frames")
        ax.grid(True, alpha=0.25)
        if len(focus) <= 10:
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / "rodd_frequency_transfer.png", dpi=190)
        plt.close(fig)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results / "analysis"
    result = analyze(args.results, output)
    print(f"Analyzed {len(result)} captured method/reference pairs -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
