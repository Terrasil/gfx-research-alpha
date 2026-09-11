#!/usr/bin/env python3
"""Search a 4x4 ordered-threshold permutation for the RODD decoder.

The objective is reconstruction-aware rather than a generic blue-noise score. It
couples four terms that are useful to the later inverse problem:
1. affine survivor-set conditioning;
2. predicted affine reconstruction variance at all target phases;
3. nested spatial dispersion of the forward coverage prefixes;
4. low-frequency energy of the nested binary masks.

The result is still a candidate. Static matrix scores are not a substitute for
final image-space quality, temporal stability and GPU timing measurements.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

BAYER = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.int32)

SHIPPED = np.array([
    [3, 11, 0, 8],
    [13, 9, 12, 4],
    [2, 10, 1, 15],
    [14, 5, 6, 7],
], dtype=np.int32)

COORDS = np.array([(x - 1.5, y - 1.5) for y in range(4) for x in range(4)], dtype=np.float64)
PHI_ALL = np.column_stack((np.ones(16), COORDS))
TORUS_D2 = np.zeros((16, 16), dtype=np.float64)
for i in range(16):
    xi, yi = i % 4, i // 4
    for j in range(16):
        xj, yj = j % 4, j // 4
        dx = min(abs(xi - xj), 4 - abs(xi - xj))
        dy = min(abs(yi - yj), 4 - abs(yi - yj))
        TORUS_D2[i, j] = dx * dx + dy * dy

LOW_FREQUENCY_BINS = ((0, 1), (0, 3), (1, 0), (3, 0))


def rank_weight(rank: int) -> float:
    if rank <= 5:
        return 3.0
    if rank <= 8:
        return 2.0
    return 1.0


def low_frequency_ratio(mask: np.ndarray) -> float:
    centered = mask.astype(np.float64) - float(mask.mean())
    spectrum = np.abs(np.fft.fft2(centered)) ** 2
    total = float(spectrum.sum())
    if total <= 1e-12:
        return 0.0
    low = sum(float(spectrum[y, x]) for y, x in LOW_FREQUENCY_BINS)
    return low / total


def metrics(pattern: np.ndarray) -> tuple[float, float, float, float, float, float]:
    flat = pattern.reshape(-1)
    log_condition = 0.0
    dispersion = 0.0
    smallest_sv = 0.0
    reconstruction_risk = 0.0
    low_frequency = 0.0
    total_weight = 0.0
    worst_condition = 1.0

    for rank in range(3, 16):
        ids = np.where(flat < rank)[0]
        A = PHI_ALL[ids]
        singular = np.linalg.svd(A, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        weight = rank_weight(rank)
        log_condition += weight * math.log(condition)
        smallest_sv += weight * float(singular[-1])
        worst_condition = max(worst_condition, condition)

        normal = A.T @ A
        covariance = np.linalg.pinv(normal, rcond=1e-12)
        leverage = np.einsum("ij,jk,ik->i", PHI_ALL, covariance, PHI_ALL)
        reconstruction_risk += weight * float(np.mean(leverage))

        distances = TORUS_D2[np.ix_(ids, ids)].copy()
        np.fill_diagonal(distances, np.inf)
        dispersion += weight * float(np.mean(np.min(distances, axis=1)))

        mask = (pattern < rank).astype(np.float64)
        low_frequency += weight * low_frequency_ratio(mask)
        total_weight += weight

    return (
        log_condition / total_weight,
        dispersion / total_weight,
        smallest_sv / total_weight,
        worst_condition,
        reconstruction_risk / total_weight,
        low_frequency / total_weight,
    )




def per_rank_metrics(pattern: np.ndarray) -> list[dict[str, float]]:
    flat = pattern.reshape(-1)
    rows: list[dict[str, float]] = []
    for rank in range(3, 16):
        ids = np.where(flat < rank)[0]
        A = PHI_ALL[ids]
        singular = np.linalg.svd(A, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        normal = A.T @ A
        covariance = np.linalg.pinv(normal, rcond=1e-12)
        leverage = np.einsum("ij,jk,ik->i", PHI_ALL, covariance, PHI_ALL)
        distances = TORUS_D2[np.ix_(ids, ids)].copy()
        np.fill_diagonal(distances, np.inf)
        rows.append({
            "rank": rank,
            "coverage": rank / 16.0,
            "condition_number": condition,
            "smallest_singular": float(singular[-1]),
            "mean_reconstruction_risk": float(np.mean(leverage)),
            "nested_dispersion": float(np.mean(np.min(distances, axis=1))),
            "low_frequency_ratio": low_frequency_ratio((pattern < rank).astype(np.float64)),
        })
    return rows


def objective(
    pattern: np.ndarray,
    dispersion_weight: float,
    reconstruction_weight: float,
    low_frequency_weight: float,
) -> float:
    log_condition, dispersion, _, worst, reconstruction_risk, low_frequency = metrics(pattern)
    return (
        log_condition
        + 0.035 * math.log(worst)
        + reconstruction_weight * math.log(max(reconstruction_risk, 1e-12))
        + low_frequency_weight * low_frequency
        - dispersion_weight * dispersion
    )


def search(
    trials: int,
    seed: int,
    dispersion_weight: float,
    reconstruction_weight: float,
    low_frequency_weight: float,
    minimum_bayer_dispersion: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bayer_metrics = metrics(BAYER)
    minimum_dispersion = bayer_metrics[1] * minimum_bayer_dispersion
    best = SHIPPED.copy()
    best_score = objective(best, dispersion_weight, reconstruction_weight, low_frequency_weight)

    for _ in range(trials):
        candidate = rng.permutation(16).reshape(4, 4)
        candidate_metrics = metrics(candidate)
        if candidate_metrics[1] < minimum_dispersion:
            continue
        score = objective(candidate, dispersion_weight, reconstruction_weight, low_frequency_weight)
        if score < best_score:
            best = candidate.copy()
            best_score = score
    return best


def describe(name: str, pattern: np.ndarray) -> None:
    log_condition, dispersion, smallest_sv, worst, reconstruction_risk, low_frequency = metrics(pattern)
    print(name)
    print(pattern)
    print(f"mean_log_condition={log_condition:.8f}")
    print(f"nested_dispersion={dispersion:.8f}")
    print(f"mean_smallest_singular={smallest_sv:.8f}")
    print(f"worst_condition={worst:.8f}")
    print(f"mean_reconstruction_risk={reconstruction_risk:.8f}")
    print(f"mean_low_frequency_ratio={low_frequency:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--dispersion-weight", type=float, default=0.50)
    parser.add_argument("--reconstruction-weight", type=float, default=0.45)
    parser.add_argument("--low-frequency-weight", type=float, default=0.35)
    parser.add_argument("--min-bayer-dispersion", type=float, default=0.88)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, help="CSV with per-rank conditioning/reconstruction metrics")
    args = parser.parse_args()

    best = search(
        args.trials,
        args.seed,
        args.dispersion_weight,
        args.reconstruction_weight,
        args.low_frequency_weight,
        args.min_bayer_dispersion,
    )
    describe("Bayer 4x4", BAYER)
    print()
    describe("Previous shipped RODD candidate", SHIPPED)
    print()
    describe("Search result", best)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.output, best, fmt="%d")

    if args.report:
        import csv
        args.report.parent.mkdir(parents=True, exist_ok=True)
        patterns = (("bayer4", BAYER), ("previous-rodd4", SHIPPED), ("search-result", best))
        with args.report.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "pattern", "rank", "coverage", "condition_number", "smallest_singular",
                "mean_reconstruction_risk", "nested_dispersion", "low_frequency_ratio",
            ])
            writer.writeheader()
            for name, pattern in patterns:
                for row in per_rank_metrics(pattern):
                    writer.writerow({"pattern": name, **row})


if __name__ == "__main__":
    main()
