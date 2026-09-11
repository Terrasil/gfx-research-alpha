#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUALITY_REFERENCE = "depth-peeling-reference-24"
TIMING_REFERENCE = "classic-forward-fixed-order"


def bootstrap_median_ci(values: np.ndarray, seed: int = 20260904, iterations: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def paired_timing(data: pd.DataFrame, sweep: str) -> pd.DataFrame:
    timing = data[data["sweep"] == sweep].copy()
    if timing.empty:
        return pd.DataFrame()

    keys = ["width", "height", "lights", "alpha", "layers", "repeat"]
    pivot = timing.pivot_table(index=keys, columns="method", values="median_ms", aggfunc="first").reset_index()
    if TIMING_REFERENCE not in pivot.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    methods = [c for c in pivot.columns if c not in keys and c != TIMING_REFERENCE]
    group_keys = ["width", "height", "lights", "alpha", "layers"]
    for group_values, group in pivot.groupby(group_keys, dropna=False):
        width, height, lights, alpha, layers = group_values
        for method in methods:
            valid = group[[method, TIMING_REFERENCE]].dropna()
            if valid.empty:
                continue
            method_ms = valid[method].to_numpy(dtype=float)
            reference_ms = valid[TIMING_REFERENCE].to_numpy(dtype=float)
            delta_ms = method_ms - reference_ms
            delta_pct = 100.0 * delta_ms / reference_ms
            lo_ms, hi_ms = bootstrap_median_ci(delta_ms)
            lo_pct, hi_pct = bootstrap_median_ci(delta_pct)
            rows.append({
                "sweep": sweep,
                "width": int(width),
                "height": int(height),
                "lights": int(lights),
                "alpha": float(alpha),
                "layers": int(layers),
                "method": method,
                "repeats": int(len(valid)),
                "method_median_ms": float(np.median(method_ms)),
                "reference_median_ms": float(np.median(reference_ms)),
                "paired_delta_ms": float(np.median(delta_ms)),
                "paired_delta_ms_ci_low": lo_ms,
                "paired_delta_ms_ci_high": hi_ms,
                "paired_delta_percent": float(np.median(delta_pct)),
                "paired_delta_percent_ci_low": lo_pct,
                "paired_delta_percent_ci_high": hi_pct,
            })
    return pd.DataFrame(rows)


def timing_analysis(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = input_dir / "timing_summary.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(path)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    single = paired_timing(data, "timing-single")
    stack = paired_timing(data, "oit-stack-timing")
    if not single.empty:
        single.to_csv(output_dir / "paired_timing_single.csv", index=False)
    if not stack.empty:
        stack.to_csv(output_dir / "paired_timing_stack.csv", index=False)
    combined = pd.concat([x for x in (single, stack) if not x.empty], ignore_index=True) if (not single.empty or not stack.empty) else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(output_dir / "paired_timing.csv", index=False)
    return single, stack


def quality_analysis(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    path = input_dir / "quality.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    if data.empty:
        return pd.DataFrame()

    columns = [
        "color_rmse", "color_relative_rmse", "color_psnr", "color_max_abs", "global_ssim",
        "gradient_rmse", "gradient_relative_rmse", "gradient_transfer", "ordered_phase_residual_ratio",
        "coverage_rmse", "coverage_mae", "coverage_max_abs",
        "alpha_rmse", "alpha_mae", "alpha_max_abs",
    ]
    columns = [column for column in columns if column in data.columns]
    group_keys = [
        "sweep", "method", "rodd_variant", "width", "height", "lights", "alpha_mode", "dither",
        "coverage_encoding", "coverage_bias", "kernel_radius", "geometry", "layers", "alpha", "wboit_depth_exponent", "layer_span",
    ]
    group_keys = [column for column in group_keys if column in data.columns]
    result = data.groupby(group_keys, dropna=False)[columns].mean().reset_index()
    result.to_csv(output_dir / "quality_summary.csv", index=False)
    return result


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def make_single_timing_plot(timing: pd.DataFrame, output_dir: Path) -> None:
    if timing.empty:
        return
    max_lights = timing["lights"].max()
    subset = timing[(timing["lights"] == max_lights) & np.isclose(timing["alpha"], 0.5)]
    if subset.empty:
        subset = timing
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for method, group in subset.groupby("method"):
        group = group.sort_values("height")
        ax.errorbar(
            group["height"], group["paired_delta_percent"],
            yerr=[
                group["paired_delta_percent"] - group["paired_delta_percent_ci_low"],
                group["paired_delta_percent_ci_high"] - group["paired_delta_percent"],
            ],
            marker="o", capsize=3, label=method,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Output height [pixels]")
    ax.set_ylabel("Method - object-sorted forward timing baseline [%]")
    ax.set_title(f"Single-layer transparency-stage timing ({int(max_lights)} lights, alpha=0.5)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_plot(fig, output_dir / "timing_single_delta.png")


def make_quality_plots(quality: pd.DataFrame, output_dir: Path) -> None:
    if quality.empty:
        return
    subset = quality[(quality["sweep"] == "quality") & (quality["alpha_mode"] == "constant") & (quality["dither"] == "bayer4") & (quality["method"] != QUALITY_REFERENCE)]
    if subset.empty:
        subset = quality[(quality["sweep"] == "quality") & (quality["method"] != QUALITY_REFERENCE)]
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method, group in subset.groupby("method"):
            group = group.sort_values("alpha")
            ax.plot(group["alpha"], group["color_relative_rmse"], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Reference opacity")
        ax.set_ylabel("Linear-HDR relative RMSE")
        ax.set_title("Single-layer image error to full-rate source-over")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "quality_single_rel_rmse.png")

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method, group in subset.groupby("method"):
            group = group.sort_values("alpha")
            ax.plot(group["alpha"], group["alpha_rmse"], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Reference opacity")
        ax.set_ylabel("Opacity RMSE")
        ax.set_title("Single-layer reconstructed opacity error")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "alpha_rmse.png")

    edge = quality[(quality["sweep"] == "depth-edge") & (quality["method"] != QUALITY_REFERENCE)]
    if not edge.empty:
        summary = edge.groupby("method", as_index=False)["color_relative_rmse"].mean().sort_values("color_relative_rmse")
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.bar(summary["method"], summary["color_relative_rmse"])
        ax.set_yscale("log")
        ax.set_ylabel("Mean linear-HDR relative RMSE")
        ax.set_title("Depth-discontinuity reconstruction")
        ax.tick_params(axis="x", labelrotation=22)
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, output_dir / "depth_edge_rel_rmse.png")

    texture = quality[(quality["sweep"] == "texture-protection") & (quality["method"] != QUALITY_REFERENCE)]
    if not texture.empty and "gradient_relative_rmse" in texture.columns:
        summary = texture.groupby(["method", "alpha"], as_index=False)[["color_relative_rmse", "gradient_relative_rmse"]].mean()
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method, group in summary.groupby("method"):
            group = group.sort_values("alpha")
            ax.plot(group["alpha"], group["gradient_relative_rmse"], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Reference opacity")
        ax.set_ylabel("Gradient relative RMSE")
        ax.set_title("Texture-edge preservation on the textured transparent panel")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "texture_edge_gradient_rmse.png")

    kernel = quality[(quality["sweep"] == "kernel") & (quality["method"] != QUALITY_REFERENCE)]
    if not kernel.empty:
        summary = kernel.groupby(["kernel_radius"], as_index=False)[["color_relative_rmse", "alpha_rmse"]].mean()
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(summary["kernel_radius"], summary["color_relative_rmse"], marker="o", label="HDR relRMSE")
        ax.plot(summary["kernel_radius"], summary["alpha_rmse"], marker="s", label="alpha RMSE")
        ax.set_yscale("log")
        ax.set_xlabel("Kernel radius [pixels]")
        ax.set_ylabel("Mean error")
        ax.set_title("Kernel-radius sensitivity")
        ax.grid(True, alpha=0.25)
        ax.legend()
        save_plot(fig, output_dir / "kernel_radius.png")



def make_pattern_comparison_plots(quality: pd.DataFrame, output_dir: Path) -> None:
    if quality.empty:
        return
    ordered = quality[
        (quality["method"] == "rodd-experimental")
        & (quality["dither"].isin(["bayer4", "bayer8", "bayer16", "rodd4-candidate"]))
        & (quality["alpha_mode"] == "constant")
        & (quality["sweep"].isin(["quality", "mixed-quality", "angled-quality", "depth-edge"]))
    ].copy()
    if ordered.empty:
        return

    metrics = [column for column in [
        "color_relative_rmse", "gradient_relative_rmse", "alpha_rmse"
    ] if column in ordered.columns]
    if metrics:
        summary = ordered.groupby(["sweep", "dither", "alpha"], as_index=False)[metrics].mean()
        summary.to_csv(output_dir / "rodd_pattern_comparison.csv", index=False)

    base = ordered[ordered["sweep"] == "quality"]
    if base.empty:
        base = ordered

    if "color_relative_rmse" in base.columns:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for dither, group in base.groupby("dither"):
            group = group.groupby("alpha", as_index=False)["color_relative_rmse"].mean().sort_values("alpha")
            ax.plot(group["alpha"], group["color_relative_rmse"], marker="o", label=dither)
        ax.set_yscale("log")
        ax.set_xlabel("Reference opacity")
        ax.set_ylabel("Linear-HDR relative RMSE")
        ax.set_title("RODD: Bayer vs reconstruction-aware pattern candidate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "rodd_pattern_color_rmse.png")

    if "alpha_rmse" in base.columns:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for dither, group in base.groupby("dither"):
            group = group.groupby("alpha", as_index=False)["alpha_rmse"].mean().sort_values("alpha")
            ax.plot(group["alpha"], group["alpha_rmse"], marker="o", label=dither)
        ax.set_yscale("log")
        ax.set_xlabel("Reference opacity")
        ax.set_ylabel("Opacity RMSE")
        ax.set_title("RODD alpha inversion: ordered-pattern comparison")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "rodd_pattern_alpha_rmse.png")

def make_measurement_design_plots(quality: pd.DataFrame, output_dir: Path) -> None:
    if quality.empty or "coverage_encoding" not in quality.columns:
        return
    subset = quality[
        (quality["sweep"] == "measurement-design")
        & (quality["method"] == "rodd-experimental")
    ].copy()
    if subset.empty:
        return

    metrics = [column for column in [
        "color_relative_rmse", "alpha_rmse", "gradient_relative_rmse",
        "ordered_phase_residual_ratio", "coverage_rmse"
    ] if column in subset.columns]
    summary = subset.groupby(
        ["dither", "coverage_encoding", "coverage_bias", "alpha"],
        as_index=False,
        dropna=False,
    )[metrics].mean()
    summary.to_csv(output_dir / "measurement_design_summary.csv", index=False)

    focus = summary[np.isclose(summary["alpha"], 0.5)].copy()
    if focus.empty:
        focus = summary.copy()
    focus["label"] = (
        focus["dither"].astype(str) + "/"
        + focus["coverage_encoding"].astype(str) + "/b="
        + focus["coverage_bias"].map(lambda value: f"{value:.2g}")
    )

    if "color_relative_rmse" in focus.columns:
        ordered = focus.sort_values("color_relative_rmse")
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        ax.bar(ordered["label"], ordered["color_relative_rmse"])
        ax.set_yscale("log")
        ax.set_ylabel("Linear-HDR relative RMSE")
        ax.set_title("Measurement design: pattern size and invertible alpha companding")
        ax.tick_params(axis="x", labelrotation=34)
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, output_dir / "measurement_design_color_rmse.png")

    if "alpha_rmse" in focus.columns:
        ordered = focus.sort_values("alpha_rmse")
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        ax.bar(ordered["label"], ordered["alpha_rmse"])
        ax.set_yscale("log")
        ax.set_ylabel("Continuous alpha RMSE")
        ax.set_title("Measurement design: recovered physical alpha")
        ax.tick_params(axis="x", labelrotation=34)
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, output_dir / "measurement_design_alpha_rmse.png")

def make_measurement_sampling_plots(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "overdraw.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    required = {"sweep", "method", "dither", "coverage_encoding", "coverage_bias", "alpha", "coverage_survival_ratio"}
    if data.empty or not required.issubset(data.columns):
        return

    subset = data[
        (data["sweep"] == "measurement-design")
        & (data["method"] == "rodd-experimental")
    ].copy()
    if subset.empty:
        return

    subset["foreground_sample_ratio"] = subset["coverage_survival_ratio"]
    subset["exposed_background_ratio"] = 1.0 - subset["coverage_survival_ratio"]
    subset["minority_sample_ratio"] = np.minimum(
        subset["foreground_sample_ratio"], subset["exposed_background_ratio"]
    )
    summary = subset.groupby(
        ["dither", "coverage_encoding", "coverage_bias", "alpha"],
        as_index=False,
        dropna=False,
    )[["foreground_sample_ratio", "exposed_background_ratio", "minority_sample_ratio"]].mean()
    summary.to_csv(output_dir / "measurement_design_sampling_summary.csv", index=False)

    focus = summary[
        (summary["coverage_encoding"] == "identity")
        | (
            (summary["coverage_encoding"] == "complementary-codesign")
            & np.isclose(summary["coverage_bias"], 0.65)
        )
        | (
            (summary["coverage_encoding"] == "balanced-logit-companding")
            & np.isclose(summary["coverage_bias"], 0.35)
        )
    ].copy()
    if focus.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for (dither, encoding), group in focus.groupby(["dither", "coverage_encoding"]):
        group = group.sort_values("alpha")
        ax.plot(
            group["alpha"],
            group["minority_sample_ratio"],
            marker="o",
            label=f"{dither}/{encoding}",
        )
    ax.set_xlabel("Physical opacity")
    ax.set_ylabel("min(foreground samples, exposed-background samples)")
    ax.set_title("Measurement preconditioning: availability of both F and B samples")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_plot(fig, output_dir / "measurement_design_sample_balance.png")


def make_ablation_plots(quality: pd.DataFrame, output_dir: Path) -> None:
    if quality.empty or "rodd_variant" not in quality.columns:
        return
    subset = quality[quality["sweep"] == "rodd-ablation"].copy()
    if subset.empty:
        return

    metrics = [column for column in [
        "color_relative_rmse", "alpha_rmse", "coverage_rmse",
        "gradient_transfer", "ordered_phase_residual_ratio", "global_ssim"
    ] if column in subset.columns]
    if metrics:
        summary = subset.groupby(["rodd_variant", "dither", "geometry", "alpha"], as_index=False)[metrics].mean()
        summary.to_csv(output_dir / "rodd_ablation_summary.csv", index=False)

    focus = subset[np.isclose(subset["alpha"], 0.5)]
    if focus.empty:
        focus = subset

    if "color_relative_rmse" in focus.columns:
        summary = focus.groupby(["rodd_variant", "dither"], as_index=False)["color_relative_rmse"].mean().sort_values("color_relative_rmse")
        summary["label"] = summary["rodd_variant"].astype(str) + "/" + summary["dither"].astype(str)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(summary["label"], summary["color_relative_rmse"])
        ax.set_yscale("log")
        ax.set_ylabel("Mean linear-HDR relative RMSE")
        ax.set_title("RODD component ablation, alpha=0.5")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, output_dir / "rodd_ablation_color_rmse.png")

    if "alpha_rmse" in focus.columns:
        summary = focus.groupby(["rodd_variant", "dither"], as_index=False)["alpha_rmse"].mean().sort_values("alpha_rmse")
        summary["label"] = summary["rodd_variant"].astype(str) + "/" + summary["dither"].astype(str)
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(summary["label"], summary["alpha_rmse"])
        ax.set_yscale("log")
        ax.set_ylabel("Continuous alpha RMSE")
        ax.set_title("Ordinal-alpha inversion ablation, alpha=0.5")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, output_dir / "rodd_ablation_alpha_rmse.png")

    if "ordered_phase_residual_ratio" in focus.columns:
        ordered = focus[np.isfinite(focus["ordered_phase_residual_ratio"])]
        if not ordered.empty:
            summary = ordered.groupby(["rodd_variant", "dither"], as_index=False)["ordered_phase_residual_ratio"].mean().sort_values("ordered_phase_residual_ratio")
            summary["label"] = summary["rodd_variant"].astype(str) + "/" + summary["dither"].astype(str)
            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            ax.bar(summary["label"], summary["ordered_phase_residual_ratio"])
            ax.set_ylabel("Fraction of error locked to 4x4 phase")
            ax.set_title("Residual ordered-pattern energy proxy")
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(True, axis="y", alpha=0.25)
            save_plot(fig, output_dir / "rodd_ablation_pattern_residual.png")


def make_oit_plots(stack_timing: pd.DataFrame, quality: pd.DataFrame, output_dir: Path) -> None:
    stack_quality = quality[(quality["sweep"] == "oit-stack-quality") & (quality["method"] != QUALITY_REFERENCE)] if not quality.empty else pd.DataFrame()
    if not stack_quality.empty:
        summary = stack_quality.groupby(["method", "layers"], as_index=False)["color_relative_rmse"].mean()
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method, group in summary.groupby("method"):
            group = group.sort_values("layers")
            ax.plot(group["layers"], group["color_relative_rmse"], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Transparent layers")
        ax.set_ylabel("Linear-HDR relative RMSE")
        ax.set_title("Controlled multi-layer stack: quality to sorted source-over")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "oit_stack_quality.png")

    if not stack_timing.empty:
        subset = stack_timing[(stack_timing["width"] == 1920) & (stack_timing["height"] == 1080)]
        if subset.empty:
            subset = stack_timing
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method, group in subset.groupby("method"):
            group = group.sort_values("layers")
            ax.errorbar(
                group["layers"], group["paired_delta_percent"],
                yerr=[
                    group["paired_delta_percent"] - group["paired_delta_percent_ci_low"],
                    group["paired_delta_percent_ci_high"] - group["paired_delta_percent"],
                ],
                marker="o", capsize=3, label=method,
            )
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Transparent layers")
        ax.set_ylabel("Method - sorted source-over [%]")
        ax.set_title("Controlled multi-layer stack: transparency-stage timing")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / "oit_stack_timing.png")

    weights = quality[quality["sweep"] == "wboit-weight"] if not quality.empty else pd.DataFrame()
    if not weights.empty:
        weights = weights.sort_values("wboit_depth_exponent")
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.plot(weights["wboit_depth_exponent"], weights["color_relative_rmse"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("WBOIT depth exponent")
        ax.set_ylabel("Linear-HDR relative RMSE")
        ax.set_title("WBOIT depth-weight sensitivity (4-layer stack)")
        ax.grid(True, alpha=0.25)
        save_plot(fig, output_dir / "wboit_weight_sensitivity.png")



def make_light_scaling_plots(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "timing_summary.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    subset = data[data["sweep"] == "light-scaling"].copy()
    if subset.empty:
        return

    summary = subset.groupby(["method", "lights"], as_index=False).agg(
        total_median_ms=("median_ms", "median"),
        total_min_ms=("median_ms", "min"),
        total_max_ms=("median_ms", "max"),
    )
    if "deferred_lighting_median_ms" in subset.columns:
        lighting = subset.groupby(["method", "lights"], as_index=False).agg(
            deferred_lighting_median_ms=("deferred_lighting_median_ms", "median")
        )
        summary = summary.merge(lighting, on=["method", "lights"], how="left")
    summary.to_csv(output_dir / "light_scaling_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for method, group in summary.groupby("method"):
        group = group.sort_values("lights")
        ax.plot(group["lights"], group["total_median_ms"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Point lights")
    ax.set_ylabel("Transparency GPU time [ms]")
    ax.set_title("Light-count scaling at 1080p, alpha=0.5")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_plot(fig, output_dir / "light_scaling_total.png")

    if "deferred_lighting_median_ms" in summary.columns:
        lit = summary[np.isfinite(summary["deferred_lighting_median_ms"])].copy()
        if not lit.empty:
            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            for method, group in lit.groupby("method"):
                group = group.sort_values("lights")
                ax.plot(group["lights"], group["deferred_lighting_median_ms"], marker="o", label=method)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Point lights")
            ax.set_ylabel("Deferred-lighting pass [ms]")
            ax.set_title("Lighting stage: proposed shared-G-buffer path vs full-rate control")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            save_plot(fig, output_dir / "light_scaling_deferred_only.png")


def make_reconstruction_order_plot(quality: pd.DataFrame, output_dir: Path) -> None:
    if quality.empty:
        return
    subset = quality[(quality["sweep"] == "reconstruction-order") & (quality["method"] != QUALITY_REFERENCE)]
    if subset.empty:
        return
    for geometry, geo in subset.groupby("geometry"):
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for method, group in geo.groupby("method"):
            group = group.sort_values("alpha")
            ax.plot(group["alpha"], group["color_relative_rmse"], marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Reference opacity")
        ax.set_ylabel("Linear-HDR relative RMSE")
        ax.set_title(f"Sparse-vs-full deferred lighting quality ablation: {geometry}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, output_dir / f"reconstruction_order_{geometry}.png")

def write_summary(single_timing: pd.DataFrame, stack_timing: pd.DataFrame, quality: pd.DataFrame, output_dir: Path) -> None:
    lines: list[str] = []
    if not single_timing.empty:
        lines.append("SINGLE-LAYER TIMING")
        for method, group in single_timing.groupby("method"):
            lines.append(
                f"{method}: paired median delta range {group.paired_delta_percent.min():+.3f}% to {group.paired_delta_percent.max():+.3f}%"
            )
        lines.append("")
    if not quality.empty:
        lines.append("QUALITY")
        for sweep, sweep_group in quality.groupby("sweep"):
            lines.append(f"[{sweep}]")
            quality_group_keys = ["method"] + (["rodd_variant"] if "rodd_variant" in sweep_group.columns else [])
            for group_name, group in sweep_group.groupby(quality_group_keys):
                label = "/".join(group_name) if isinstance(group_name, tuple) else str(group_name)
                lines.append(
                    f"{label}: mean color relRMSE={group.color_relative_rmse.mean():.6g}, "
                    + (f"mean gradient relRMSE={group.gradient_relative_rmse.mean():.6g}, " if "gradient_relative_rmse" in group.columns else "")
                    + f"mean alpha RMSE={group.alpha_rmse.mean():.6g}"
                )
        lines.append("")
    if not stack_timing.empty:
        lines.append("MULTI-LAYER TIMING")
        for method, group in stack_timing.groupby("method"):
            lines.append(
                f"{method}: paired median delta range {group.paired_delta_percent.min():+.3f}% to {group.paired_delta_percent.max():+.3f}%"
            )
    if not lines:
        lines.append("No benchmark CSV files were found or they contained no analyzable rows.")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze gfx-research-alpha benchmark-suite output")
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output or args.results / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    single_timing, stack_timing = timing_analysis(args.results, output_dir)
    quality = quality_analysis(args.results, output_dir)
    make_single_timing_plot(single_timing, output_dir)
    make_quality_plots(quality, output_dir)
    make_pattern_comparison_plots(quality, output_dir)
    make_measurement_design_plots(quality, output_dir)
    make_measurement_sampling_plots(args.results, output_dir)
    make_ablation_plots(quality, output_dir)
    make_oit_plots(stack_timing, quality, output_dir)
    make_light_scaling_plots(args.results, output_dir)
    make_reconstruction_order_plot(quality, output_dir)
    write_summary(single_timing, stack_timing, quality, output_dir)
    print(f"Analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
