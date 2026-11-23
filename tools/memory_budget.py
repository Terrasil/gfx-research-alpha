#!/usr/bin/env python3
"""Analytical transparency-storage budget for the formats used by the harness.

The report separates common deferred-renderer storage from method-specific
transparency storage. A common final HDR output is deliberately excluded from
all methods so the comparison does not charge one method for a shared target.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

RESOLUTIONS = ((1920, 1080), (2560, 1440), (3840, 2160))

# Bytes per pixel for the exact formats used in src/main.cpp and target classes.
RGBA16F = 8
R16F = 2
R32F = 4
RGBA8 = 4
D32F = 4

COMMON_DEFERRED_BPP = 3 * RGBA16F + D32F + RGBA16F  # opaque G-buffer + lit HDR

METHODS = {
    "forward-sorted-alpha": {
        "extra_bpp": 0,
        "note": "no additional per-pixel transparency buffer; ordering/geometry work is not a byte-per-pixel texture cost",
    },
    "raw-dither-shared-gbuffer": {
        "extra_bpp": RGBA16F,
        "note": "selected-sample control attachment added to the shared deferred G-buffer",
    },
    "rodd-production-minimum": {
        "extra_bpp": RGBA16F,
        "note": "same selected-sample control attachment; reconstruction meta/debug target excluded",
    },
    "rodd-harness-with-debug-meta": {
        "extra_bpp": RGBA16F + RGBA16F,
        "note": "selected-sample control + RGBA16F reconstruction metadata; final HDR output remains common/excluded",
    },
    "wboit": {
        "extra_bpp": RGBA16F + R16F,
        "note": "RGBA16F accumulation + R16F revealage",
    },
    "object-sorted-forward": {
        "extra_bpp": 0,
        "note": "object sorting requires no additional per-pixel transparency attachment",
    },
    "deferred-sparse-layer-2-edge-aware": {
        "extra_bpp": 4 * RGBA16F + D32F + 2 * (5 * RGBA16F),
        "note": "mixed G-buffer plus two reconstructed G-buffers and lighting targets",
    },
    "deferred-sparse-layer-2-edge-decoupled": {
        "extra_bpp": 4 * RGBA16F + D32F + 2 * (5 * RGBA16F),
        "note": "same storage as the coupled reconstruction",
    },
    "deferred-sparse-layer-2-edge-packed": {
        "extra_bpp": 4 * RGBA16F + D32F + 2 * (3 * RGBA16F + RGBA8 + 4),
        "note": "packed RGBA8 albedo/roughness and RGB10_A2 normal/validity in each reconstructed layer",
    },
    "layered-wboit-8": {
        "extra_bpp": 8 * (RGBA16F + R16F) + R16F + R32F + RGBA8 + D32F,
        "note": "8x accumulation/revealage + global revealage + R32F/RGBA8/D32F support target used by this harness",
    },
}


def mib(bytes_count: int) -> float:
    return bytes_count / (1024.0 * 1024.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    for width, height in RESOLUTIONS:
        pixels = width * height
        for method, spec in METHODS.items():
            extra_bpp = int(spec["extra_bpp"])
            rows.append({
                "method": method,
                "width": width,
                "height": height,
                "common_deferred_bpp": COMMON_DEFERRED_BPP,
                "method_specific_bpp": extra_bpp,
                "method_specific_mib": mib(pixels * extra_bpp),
                "common_plus_method_mib": mib(pixels * (COMMON_DEFERRED_BPP + extra_bpp)),
                "note": spec["note"],
            })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Common deferred storage: {COMMON_DEFERRED_BPP} B/pixel (3xRGBA16F + D32F + RGBA16F lit target)")
    for row in rows:
        if row["width"] == 3840 and row["height"] == 2160:
            print(f"{row['method']:<30} extra={row['method_specific_bpp']:>3} B/px  extra@4K={row['method_specific_mib']:8.2f} MiB")
    if args.output:
        print(f"CSV -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
