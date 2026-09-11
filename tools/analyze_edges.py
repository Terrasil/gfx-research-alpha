#!/usr/bin/env python3
"""Edge and transition analysis for captured alpha/coverage fields.

The renderer writes scalar PFM maps for the reference opacity and each captured
method's coverage/continuous-alpha estimate. This script measures a symmetric
edge-location error, 10-90% transition width, overshoot and undershoot on local
profiles normal to reference edges. It is intended for the controlled panel,
depth-step, angled-plate and alpha-pattern captures, not as a perceptual metric.
"""
from __future__ import annotations

import argparse
import math
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
    image = data.reshape(height, width, channels).astype(np.float64, copy=False)
    return image[..., 0]


def gradient_field(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def edge_mask(image: np.ndarray) -> np.ndarray:
    _, _, magnitude = gradient_field(image)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=bool)
    positive = finite[finite > 1e-8]
    if positive.size == 0:
        return np.zeros_like(image, dtype=bool)
    threshold = max(0.015, 0.30 * float(np.percentile(positive, 90.0)))
    return magnitude >= threshold


def chamfer_distance_to_edges(edges: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance transform using an 8-neighbour chamfer."""
    height, width = edges.shape
    inf = float(height + width + 16)
    distance = np.where(edges, 0.0, inf).astype(np.float64)
    diagonal = math.sqrt(2.0)
    for y in range(height):
        for x in range(width):
            value = distance[y, x]
            if y > 0:
                value = min(value, distance[y - 1, x] + 1.0)
                if x > 0:
                    value = min(value, distance[y - 1, x - 1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y - 1, x + 1] + diagonal)
            if x > 0:
                value = min(value, distance[y, x - 1] + 1.0)
            distance[y, x] = value
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            value = distance[y, x]
            if y + 1 < height:
                value = min(value, distance[y + 1, x] + 1.0)
                if x > 0:
                    value = min(value, distance[y + 1, x - 1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y + 1, x + 1] + diagonal)
            if x + 1 < width:
                value = min(value, distance[y, x + 1] + 1.0)
            distance[y, x] = value
    return distance


def symmetric_edge_location_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_edges = edge_mask(reference)
    est_edges = edge_mask(estimate)
    if not np.any(ref_edges) or not np.any(est_edges):
        return math.nan
    to_est = chamfer_distance_to_edges(est_edges)
    to_ref = chamfer_distance_to_edges(ref_edges)
    return 0.5 * (float(np.mean(to_est[ref_edges])) + float(np.mean(to_ref[est_edges])))


def bilinear(image: np.ndarray, x: float, y: float) -> float:
    height, width = image.shape
    x = float(np.clip(x, 0.0, width - 1.0))
    y = float(np.clip(y, 0.0, height - 1.0))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    a = (1.0 - tx) * image[y0, x0] + tx * image[y0, x1]
    b = (1.0 - tx) * image[y1, x0] + tx * image[y1, x1]
    return float((1.0 - ty) * a + ty * b)


def crossing_position(t: np.ndarray, values: np.ndarray, target: float, increasing: bool) -> float:
    for index in range(len(t) - 1):
        a = values[index]
        b = values[index + 1]
        crossed = (a <= target <= b) if increasing else (a >= target >= b)
        if not crossed:
            continue
        delta = b - a
        if abs(delta) <= 1e-12:
            return float(t[index])
        u = (target - a) / delta
        return float(t[index] + u * (t[index + 1] - t[index]))
    return math.nan


def transition_statistics(reference: np.ndarray, estimate: np.ndarray, max_profiles: int = 1600) -> dict[str, float]:
    gx, gy, magnitude = gradient_field(reference)
    edges = edge_mask(reference)
    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return {"transition_width_ref": math.nan, "transition_width_est": math.nan, "transition_width_error": math.nan, "overshoot": math.nan, "undershoot": math.nan, "profiles": 0}
    stride = max(1, xs.size // max_profiles)
    t = np.linspace(-8.0, 8.0, 65)
    widths_ref: list[float] = []
    widths_est: list[float] = []
    overshoots: list[float] = []
    undershoots: list[float] = []

    for y, x in zip(ys[::stride], xs[::stride]):
        mag = float(magnitude[y, x])
        if mag <= 1e-8:
            continue
        nx = float(gx[y, x] / mag)
        ny = float(gy[y, x] / mag)
        ref_profile = np.array([bilinear(reference, x + nx * dt, y + ny * dt) for dt in t])
        est_profile = np.array([bilinear(estimate, x + nx * dt, y + ny * dt) for dt in t])
        low = float(np.percentile(ref_profile, 5.0))
        high = float(np.percentile(ref_profile, 95.0))
        amplitude = high - low
        if amplitude < 0.05:
            continue
        increasing = ref_profile[-1] >= ref_profile[0]
        target10 = low + 0.1 * amplitude
        target90 = low + 0.9 * amplitude
        r10 = crossing_position(t, ref_profile, target10, increasing)
        r90 = crossing_position(t, ref_profile, target90, increasing)
        e10 = crossing_position(t, est_profile, target10, increasing)
        e90 = crossing_position(t, est_profile, target90, increasing)
        if math.isfinite(r10) and math.isfinite(r90):
            widths_ref.append(abs(r90 - r10))
        if math.isfinite(e10) and math.isfinite(e90):
            widths_est.append(abs(e90 - e10))
        overshoots.append(max(0.0, float(np.max(est_profile)) - high) / amplitude)
        undershoots.append(max(0.0, low - float(np.min(est_profile))) / amplitude)

    ref_width = float(np.median(widths_ref)) if widths_ref else math.nan
    est_width = float(np.median(widths_est)) if widths_est else math.nan
    return {
        "transition_width_ref": ref_width,
        "transition_width_est": est_width,
        "transition_width_error": est_width - ref_width if math.isfinite(ref_width) and math.isfinite(est_width) else math.nan,
        "overshoot": float(np.mean(overshoots)) if overshoots else math.nan,
        "undershoot": float(np.mean(undershoots)) if undershoots else math.nan,
        "profiles": min(len(overshoots), max_profiles),
    }


def analyze(results: Path, output: Path) -> pd.DataFrame:
    quality_path = results / "quality.csv"
    captures = results / "captures"
    if not quality_path.exists():
        raise FileNotFoundError(quality_path)
    quality = pd.read_csv(quality_path)
    rows: list[dict] = []

    for _, row in quality.iterrows():
        method = str(row["method"])
        if method == REFERENCE:
            continue
        pair_id = str(row["pair_id"])
        reference_path = captures / f"{pair_id}-alpha-reference.pfm"
        if not reference_path.exists():
            continue
        reference = read_pfm(reference_path)
        stem = f"{pair_id}-{method}"
        for kind in ("coverage", "alpha"):
            estimate_path = captures / f"{stem}-{kind}-estimate.pfm"
            if not estimate_path.exists():
                continue
            estimate = read_pfm(estimate_path)
            if estimate.shape != reference.shape:
                continue
            transition = transition_statistics(reference, estimate)
            rows.append({
                "pair_id": pair_id,
                "sweep": row.get("sweep", ""),
                "method": method,
                "rodd_variant": row.get("rodd_variant", "full"),
                "dither": row.get("dither", ""),
                "alpha_mode": row.get("alpha_mode", ""),
                "geometry": row.get("geometry", ""),
                "alpha": row.get("alpha", np.nan),
                "field": kind,
                "edge_location_error_px": symmetric_edge_location_error(reference, estimate),
                **transition,
            })

    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "edge_summary.csv", index=False)

    if not result.empty:
        focus = result[(result["field"] == "coverage") & (result["method"] == "rodd-experimental")]
        if not focus.empty:
            fig, ax = plt.subplots(figsize=(7.8, 4.6))
            labels = [f"{d}/{v}" for d, v in zip(focus["dither"], focus["rodd_variant"])]
            ax.scatter(labels, focus["edge_location_error_px"], alpha=0.8)
            ax.set_ylabel("Symmetric edge-location error [px]")
            ax.set_title("RODD edge localization on captured coverage fields")
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / "rodd_edge_location.png", dpi=190)
            plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results / "analysis"
    result = analyze(args.results, output)
    print(f"Analyzed {len(result)} captured alpha/coverage fields -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
