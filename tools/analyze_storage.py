#!/usr/bin/env python3
"""Compare matched transparency quality, raster timing and format footprints."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from memory_budget import METHODS as FORMATS

METHODS = {name: FORMATS[name]["extra_bpp"] for name in (
    "wboit","layered-wboit-8","object-sorted-forward","deferred-sparse-layer-2-edge-aware",
    "deferred-sparse-layer-2-edge-decoupled","deferred-sparse-layer-2-edge-packed")}
PACKED = "deferred-sparse-layer-2-edge-packed"
UNPACKED = "deferred-sparse-layer-2-edge-decoupled"
MATCH = ("width","height","lights","alpha_mode","dither","coverage_encoding","coverage_bias",
         "kernel_radius","geometry","layers","alpha","depth_sigma","albedo_sigma")


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    manifest = dict(line.split("=", 1) for line in (root/"manifest.txt").read_text().splitlines() if "=" in line)
    if manifest.get("completed") != "1" or manifest.get("storage_study") not in ("smoke","quick","paper"):
        raise ValueError("A completed packed storage study is required")
    raw, summary, quality = (read(root/name) for name in ("timing_raw.csv","timing_summary.csv","quality.csv"))
    quality_groups, time_groups = defaultdict(dict), defaultdict(dict)
    for r in quality:
        if r["method"] in quality_groups[r["pair_id"]]:
            raise ValueError("Duplicate quality measurement")
        quality_groups[r["pair_id"]][r["method"]] = r
    sample_counts = {"smoke":3,"quick":8,"paper":32}
    repeats = {"smoke":1,"quick":2,"paper":5}[manifest["storage_study"]]
    expected = sample_counts[manifest["storage_study"]]
    planned = int(manifest["planned_cases"])
    timing = defaultdict(dict)
    for r in raw:
        key = r["pair_id"], int(r["repeat"]), r["method"]
        if r["sample"] in timing[key] or not 0 < float(r["gpu_ms"]) < math.inf:
            raise ValueError(f"Invalid timing in {key}")
        timing[key][r["sample"]] = float(r["gpu_ms"])
    if len(timing) != planned or len(summary) != planned:
        raise ValueError("Timing matrix is incomplete")
    for r in summary:
        key = r["pair_id"], int(r["repeat"]), r["method"]
        values = timing[key]
        med = float(r["median_ms"])
        if set(values) != {str(n) for n in range(expected)} or not math.isclose(med,median(values.values()),rel_tol=1e-6):
            raise ValueError(f"Timing summary does not match raw samples in {key}")
        block = time_groups[key[:2]]
        if r["method"] in block:
            raise ValueError(f"Duplicate timing group {key}")
        block[r["method"]] = med
    signed = {}
    for r in read(root/"attribute_quality.csv"):
        signed[(r["pair_id"],r["method"],int(r["layer"]))] = r
    rows = []
    for case, arms in sorted(quality_groups.items()):
        if set(arms) != set(METHODS) or len({tuple(r[k] for k in MATCH) for r in arms.values()}) != 1:
            raise ValueError(f"Unmatched method settings for {case}")
        blocks = [time_groups[case,i] for i in range(repeats)]
        if any(set(block) != set(arms) for block in blocks):
            raise ValueError(f"Incomplete blocks for {case}")
        for arm, r in arms.items():
            color, edge = float(r["color_rmse"]), float(r["gradient_rmse"])
            if not math.isfinite(color) or not math.isfinite(edge):
                raise ValueError(f"Nonfinite quality for {case} / {arm}")
            front = signed.get((case,arm,0),{})
            back = signed.get((case,arm,1),{})
            rows.append(dict(case=case,method=arm,width=r["width"],height=r["height"],geometry=r["geometry"],
                             layers=r["layers"],alpha=r["alpha"],color_rmse=color,gradient_rmse=edge,
                             front_alpha_signed_mean=front.get("alpha_signed_mean",""),
                             back_alpha_signed_mean=back.get("alpha_signed_mean",""),
                             back_reference_samples=back.get("reference_samples",""),
                             back_matched_samples=back.get("matched_samples",""),
                             transparency_bpp=METHODS[arm],
                             transparency_mib=int(r["width"])*int(r["height"])*METHODS[arm]/2**20,
                             gpu_median_ms=median(b[arm] for b in blocks),
                             delta_ms_vs_unpacked=median(b[arm]-b[UNPACKED] for b in blocks),
                             rgb_delta_vs_unpacked=color-float(arms[UNPACKED]["color_rmse"])))
    if len(rows)*repeats != planned:
        raise ValueError("Quality matrix is incomplete")
    write(root/"storage_comparison.csv",rows)
    print(f"{len(quality_groups)} paired cases. Packed layer storage: {METHODS[UNPACKED]} to {METHODS[PACKED]} B per pixel.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    analyze(p.parse_args().results)
