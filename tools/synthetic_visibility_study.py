#!/usr/bin/env python3
"""Synthetic two-layer visibility study for the publication candidate.

Produces deterministic CSV data for the censored ordinal alpha estimator using
translated Bayer8 thresholds and surface-decorrelated static hash thresholds.
No GPU or image assets are required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def bayer_rank(x: int, y: int, size: int = 8) -> int:
    levels = {4: 2, 8: 3, 16: 4}[size]
    weight = size * size // 4
    rank = 0
    for bit in range(levels):
        xb = (x >> bit) & 1
        yb = (y >> bit) & 1
        code = (0 if xb == 0 else 2) if yb == 0 else (3 if xb == 0 else 1)
        rank += code * weight
        weight //= 4
    return rank


def bayer_threshold(x: int, y: int, seed: int, size: int = 8) -> float:
    levels = {4: 2, 8: 3, 16: 4}[size]
    mask = size - 1
    sx = seed & mask
    sy = (seed >> levels) & mask
    return (bayer_rank((x + sx) & mask, (y + sy) & mask, size) + 0.5) / (size * size)


def hash_threshold(x: int, y: int, seed: int) -> float:
    mask32 = 0xFFFFFFFF
    qx = (x & mask32) ^ (seed & mask32)
    qy = (y & mask32) ^ ((seed * 0x9E3779B9) & mask32)
    h = ((qx * 0x9E3779B9) ^ (qy * 0x85EBCA6B)) & mask32
    h ^= h >> 16
    h = (h * 0x7FEB352D) & mask32
    h ^= h >> 15
    h = (h * 0x846CA68B) & mask32
    h ^= h >> 16
    return (h & 0x00FFFFFF) / float(0x01000000)


def encode_high_alpha(alpha: float, p: float) -> float:
    pivot = 0.5
    if alpha <= pivot or p <= 1.0001:
        return alpha
    t = (alpha - pivot) / (1.0 - pivot)
    return pivot + (1.0 - pivot) * (t ** p)


def decode_high_alpha(beta: float, p: float) -> float:
    pivot = 0.5
    if beta <= pivot or p <= 1.0001:
        return beta
    t = (beta - pivot) / (1.0 - pivot)
    return pivot + (1.0 - pivot) * (t ** (1.0 / p))


def estimate(front_alpha: float, back_alpha: float, p: float, front_seed: int, back_seed: int, threshold_fn):
    front_beta = encode_high_alpha(front_alpha, p)
    back_beta = encode_high_alpha(back_alpha, p)
    fl, fu, bl, bu = 0.0, 1.0, 0.0, 1.0
    front_samples = back_samples = background_samples = 0
    for y in range(8):
        for x in range(8):
            tf = threshold_fn(x, y, front_seed)
            tb = threshold_fn(x, y, back_seed)
            if front_beta > tf:
                front_samples += 1
                fl = max(fl, tf)
            elif back_beta > tb:
                back_samples += 1
                fu = min(fu, tf)
                bl = max(bl, tb)
            else:
                background_samples += 1
                fu = min(fu, tf)
                bu = min(bu, tb)
    front_hat = decode_high_alpha(0.5 * (fl + fu), p)
    back_hat = decode_high_alpha(0.5 * (bl + bu), p)
    return front_hat, back_hat, front_samples, back_samples, background_samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("results/publication/synthetic_visibility.csv"))
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pattern, fn in [("bayer8", bayer_threshold), ("hash", hash_threshold)]:
        for p in [1.0, 1.25, 1.5, 2.0, 2.5]:
            for front_alpha in [0.5, 0.625, 0.75, 0.875]:
                for back_alpha in [0.25, 0.5, 0.75, 0.875]:
                    for front_seed in range(1, 9):
                        for back_seed in range(17, 25):
                            fh, bh, fs, bs, ns = estimate(front_alpha, back_alpha, p, front_seed, back_seed, fn)
                            rows.append({
                                "pattern": pattern,
                                "exponent": p,
                                "front_alpha": front_alpha,
                                "back_alpha": back_alpha,
                                "front_seed": front_seed,
                                "back_seed": back_seed,
                                "front_alpha_hat": fh,
                                "back_alpha_hat": bh,
                                "front_abs_error": abs(fh - front_alpha),
                                "back_abs_error": abs(bh - back_alpha),
                                "front_samples": fs,
                                "back_samples": bs,
                                "background_samples": ns,
                            })

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def avg(pattern: str, p: float, key: str) -> float:
        vals = [r[key] for r in rows if r["pattern"] == pattern and abs(r["exponent"] - p) < 1e-9]
        return sum(vals) / len(vals)

    summary = args.output.with_name("synthetic_visibility_summary.txt")
    text = ["Synthetic censored-visibility study"]
    for p in [1.0, 1.25, 1.5, 2.0, 2.5]:
        text.append(
            f"p={p:.2f}: back MAE Bayer8={avg('bayer8', p, 'back_abs_error'):.5f}, "
            f"hash={avg('hash', p, 'back_abs_error'):.5f}, "
            f"hash back samples={avg('hash', p, 'back_samples'):.2f}/64"
        )
    summary.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
