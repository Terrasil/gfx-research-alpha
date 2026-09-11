#!/usr/bin/env python3
"""Publication go/no-go evaluator for the deferred transparency study.

The thresholds in this script are deliberately conservative engineering gates, not
claims about conference acceptance. They answer a simpler question: does the new
method demonstrate enough quality/performance signal to justify writing a paper?
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROPOSED = "deferred-sparse-layer-2-edge-aware"
ATTR1 = "deferred-attribute-reconstruction"
LITSPACE = "rodd-experimental"
WBOIT = "wboit"
LAYERED = "layered-wboit-8"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value
    except (TypeError, ValueError):
        return default


def median(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.median(values) if values else math.nan


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.fmean(values) if values else math.nan


def ratio(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-12:
        return math.nan
    return a / b


def paired_ratios(rows: list[dict[str, str]], sweep: str, method_a: str, method_b: str, metric: str,
                  extra=lambda r: True) -> list[float]:
    groups: dict[tuple, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        if r.get("sweep") != sweep or not extra(r):
            continue
        key = (
            r.get("pair_id"), r.get("width"), r.get("height"), r.get("lights"),
            r.get("alpha_mode"), r.get("dither"), r.get("coverage_encoding"),
            r.get("coverage_bias"), r.get("kernel_radius"), r.get("geometry"),
            r.get("layers"), r.get("alpha"),
        )
        groups[key][r.get("method", "")] = r
    out = []
    for methods in groups.values():
        if method_a not in methods or method_b not in methods:
            continue
        ra = f(methods[method_a], metric)
        rb = f(methods[method_b], metric)
        q = ratio(ra, rb)
        if math.isfinite(q):
            out.append(q)
    return out


def pair_by_config(rows: list[dict[str, str]], sweep: str, method: str, encoding: str | None = None):
    return [
        r for r in rows
        if r.get("sweep") == sweep and r.get("method") == method
        and (encoding is None or r.get("coverage_encoding") == encoding)
    ]


def publication_evaluation(results: Path) -> tuple[str, list[str]]:
    quality = read_csv(results / "quality.csv")
    timing = read_csv(results / "timing_summary.csv")
    overdraw = read_csv(results / "overdraw.csv")
    manifest = (results / "manifest.txt").read_text(encoding="utf-8", errors="replace") if (results / "manifest.txt").exists() else ""

    lines: list[str] = []
    if "completed=1" not in manifest:
        lines.append("ERROR: benchmark manifest does not contain completed=1.")
        return "INCOMPLETE", lines
    if not quality or not timing:
        lines.append("ERROR: quality.csv or timing_summary.csv is missing/empty.")
        return "INCOMPLETE", lines

    # Gate 1: attribute-space reconstruction should improve on the old lit-space decoder.
    single_rmse = paired_ratios(
        quality, "publication-single-quality", PROPOSED, LITSPACE, "color_rmse",
        extra=lambda r: r.get("coverage_encoding") == "identity"
    )
    single_grad = paired_ratios(
        quality, "publication-single-quality", PROPOSED, LITSPACE, "gradient_rmse",
        extra=lambda r: r.get("coverage_encoding") == "identity"
    )
    single_rmse_med = median(single_rmse)
    single_grad_med = median(single_grad)
    gate_attribute = (
        math.isfinite(single_rmse_med) and math.isfinite(single_grad_med)
        and single_rmse_med <= 0.92 and single_grad_med <= 0.95
    )
    lines.append(
        f"G1 attribute-space vs lit-space: median RMSE ratio={single_rmse_med:.3f}, "
        f"gradient ratio={single_grad_med:.3f} -> {'PASS' if gate_attribute else 'FAIL'}"
    )

    # Gate 2: two-layer overlap quality should be competitive with WBOIT in the primary hash cases.
    overlap_rmse = paired_ratios(
        quality, "publication-overlap-quality", PROPOSED, WBOIT, "color_rmse",
        extra=lambda r: r.get("dither") == "hash" and r.get("coverage_encoding") == "identity"
    )
    overlap_grad = paired_ratios(
        quality, "publication-overlap-quality", PROPOSED, WBOIT, "gradient_rmse",
        extra=lambda r: r.get("dither") == "hash" and r.get("coverage_encoding") == "identity"
    )
    overlap_rmse_med = median(overlap_rmse)
    overlap_grad_med = median(overlap_grad)
    gate_overlap = (
        math.isfinite(overlap_rmse_med) and math.isfinite(overlap_grad_med)
        and overlap_rmse_med <= 1.05 and overlap_grad_med <= 1.05
    )
    lines.append(
        f"G2 two-layer overlap vs WBOIT: median RMSE ratio={overlap_rmse_med:.3f}, "
        f"gradient ratio={overlap_grad_med:.3f} -> {'PASS' if gate_overlap else 'FAIL'}"
    )

    # Gate 3: high-alpha reveal must provide a measurable improvement in high-alpha overlap.
    identity_rows = {
        (r.get("geometry"), r.get("alpha")): r
        for r in quality
        if r.get("sweep") == "publication-overlap-quality"
        and r.get("method") == PROPOSED
        and r.get("dither") == "hash"
        and r.get("coverage_encoding") == "identity"
        and f(r, "alpha") >= 0.75
    }
    power_groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for r in quality:
        if r.get("sweep") != "publication-power-reveal" or r.get("method") != PROPOSED:
            continue
        if f(r, "alpha") < 0.75:
            continue
        power_groups[(r.get("geometry"), r.get("alpha"))].append(r)

    power_improvements: list[float] = []
    best_exponents: list[float] = []
    for key, baseline in identity_rows.items():
        candidates = power_groups.get(key, [])
        if not candidates:
            continue
        base_rmse = f(baseline, "color_rmse")
        best = min(candidates, key=lambda r: f(r, "color_rmse", math.inf))
        best_rmse = f(best, "color_rmse")
        if math.isfinite(base_rmse) and base_rmse > 0 and math.isfinite(best_rmse):
            power_improvements.append(1.0 - best_rmse / base_rmse)
            best_exponents.append(f(best, "coverage_bias"))
    power_gain = median(power_improvements)
    gate_power = math.isfinite(power_gain) and power_gain >= 0.05
    lines.append(
        f"G3 high-alpha reveal: median best RMSE improvement={power_gain * 100.0:.1f}% "
        f"(median best p={median(best_exponents):.2f}) -> {'PASS' if gate_power else 'FAIL'}"
    )

    # Gate 4: kernel sensitivity should not imply a single cherry-picked radius.
    kernel_rows = [r for r in quality if r.get("sweep") == "publication-kernel-ablation" and r.get("method") == PROPOSED]
    kernel_groups: dict[tuple, list[float]] = defaultdict(list)
    for r in kernel_rows:
        key = (r.get("geometry"), r.get("alpha"))
        kernel_groups[key].append(f(r, "color_rmse"))
    spreads = []
    for values in kernel_groups.values():
        values = [v for v in values if math.isfinite(v) and v > 0]
        if len(values) >= 2:
            spreads.append(max(values) / min(values))
    kernel_spread = median(spreads)
    gate_kernel = math.isfinite(kernel_spread) and kernel_spread <= 1.35
    lines.append(
        f"G4 kernel robustness: median worst/best RMSE={kernel_spread:.3f} -> {'PASS' if gate_kernel else 'FAIL'}"
    )

    # Gate 5: performance crossover at 256 lights. This is the main performance claim.
    perf = [r for r in timing if r.get("sweep") == "publication-performance" and int(float(r.get("lights", "0"))) == 256]
    perf_by = defaultdict(lambda: defaultdict(list))
    for r in perf:
        perf_by[r.get("geometry")][r.get("method")].append(f(r, "median_ms"))
    perf_ratios = []
    for geometry, methods in perf_by.items():
        a = median(methods.get(PROPOSED, []))
        b = median(methods.get(WBOIT, []))
        q = ratio(a, b)
        if math.isfinite(q):
            perf_ratios.append(q)
            lines.append(f"  256-light timing {geometry}: proposed/WBOIT={q:.3f}")
    perf_ratio_med = median(perf_ratios)
    gate_perf = math.isfinite(perf_ratio_med) and perf_ratio_med <= 0.95
    lines.append(
        f"G5 high-light performance: median proposed/WBOIT={perf_ratio_med:.3f} -> {'PASS' if gate_perf else 'FAIL'}"
    )

    # Informational: Bayer-vs-hash overlap ratio for the proposed method.
    proposed_overlap = [
        r for r in quality
        if r.get("sweep") == "publication-overlap-quality" and r.get("method") == PROPOSED
        and r.get("coverage_encoding") == "identity"
    ]
    by_key = defaultdict(dict)
    for r in proposed_overlap:
        by_key[(r.get("geometry"), r.get("alpha"))][r.get("dither")] = r
    hash_vs_bayer = []
    for methods in by_key.values():
        if "hash" in methods and "bayer8" in methods:
            q = ratio(f(methods["hash"], "color_rmse"), f(methods["bayer8"], "color_rmse"))
            if math.isfinite(q):
                hash_vs_bayer.append(q)
    hb = median(hash_vs_bayer)
    lines.append(f"INFO surface-decorrelated hash/Bayer8 overlap RMSE ratio={hb:.3f}")

    # Informational: out-of-scope degradation with >2 layers.
    stress = [r for r in quality if r.get("sweep") == "publication-layer-stack-stress" and r.get("method") == PROPOSED]
    stress_by_layers = defaultdict(list)
    for r in stress:
        stress_by_layers[int(float(r.get("layers", "0")))].append(f(r, "color_rmse"))
    if stress_by_layers:
        desc = ", ".join(f"{k}L={mean(v):.4g}" for k, v in sorted(stress_by_layers.items()))
        lines.append("INFO out-of-scope stack RMSE: " + desc)

    gates = [gate_attribute, gate_overlap, gate_power, gate_kernel, gate_perf]
    passed = sum(gates)
    critical = gate_attribute and (gate_overlap or gate_perf)
    if critical and passed >= 4:
        verdict = "PROMISING"
    elif critical and passed >= 3:
        verdict = "BORDERLINE"
    else:
        verdict = "STOP"

    lines.append("")
    lines.append(f"VERDICT: {verdict} ({passed}/5 engineering gates passed)")
    lines.append("These gates are a go/no-go heuristic for continuing the research, not a venue acceptance prediction.")
    return verdict, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="results/publication", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    verdict, lines = publication_evaluation(args.results)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = args.output or (args.results / "publication_verdict.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0 if verdict in {"PROMISING", "BORDERLINE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
