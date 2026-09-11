#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path

ATTRS = [
    "albedo_rmse",
    "normal_mean_deg",
    "roughness_rmse",
    "position_rmse",
    "alpha_rmse",
    "match_recall",
    "false_positive_ratio",
]
TARGETS = ["color_relative_rmse", "gradient_relative_rmse"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return math.nan
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return math.nan if den <= 1e-20 else sum(x*y for x, y in zip(dx, dy)) / den


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/v32-edge-decision")
    attr_path = root / "attribute_quality.csv"
    quality_path = root / "quality.csv"
    if not attr_path.exists() or not quality_path.exists():
        print("attribute_quality.csv or quality.csv missing", file=sys.stderr)
        return 1

    attrs = read_rows(attr_path)
    quality = read_rows(quality_path)
    qmap = {(r.get("pair_id"), r.get("method")): r for r in quality}

    out = []
    for layer in ("0", "1"):
        layer_rows = [r for r in attrs if r.get("layer") == layer]
        for attr in ATTRS:
            for target in TARGETS:
                xs, ys = [], []
                for r in layer_rows:
                    q = qmap.get((r.get("pair_id"), r.get("method")))
                    if not q:
                        continue
                    x = num(r.get(attr))
                    y = num(q.get(target))
                    if x is None or y is None:
                        continue
                    xs.append(x)
                    ys.append(y)
                out.append({
                    "layer": layer,
                    "attribute_metric": attr,
                    "image_metric": target,
                    "samples": len(xs),
                    "pearson_r": pearson(xs, ys),
                })

    output = root / "attribute_correlation.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "attribute_metric", "image_metric", "samples", "pearson_r"])
        writer.writeheader()
        writer.writerows(out)

    ranked = [r for r in out if r["samples"] >= 3 and math.isfinite(r["pearson_r"])]
    ranked.sort(key=lambda r: abs(r["pearson_r"]), reverse=True)
    print("Strongest absolute attribute/image correlations:")
    for r in ranked[:10]:
        print(f"  layer {r['layer']} {r['attribute_metric']} vs {r['image_metric']}: r={r['pearson_r']:.3f} (n={r['samples']})")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
