# gfx-research-alpha

Minimal research harness for bounded sparse-layer deferred transparency.

## Current method

The proposed path reconstructs up to two transparent G-buffer layers from deterministic coverage observations, then performs deferred lighting and front-to-back composition. Depth peeling (24 layers) is the image-quality reference. Baselines include classic fixed-order forward alpha, object-sorted forward alpha, WBOIT and layered WBOIT.

The formal method scope is bounded (`K <= 2`). Four- and eight-layer stacks are stress tests, not a claim of arbitrary-depth OIT.

## Build

```powershell
cmake -S . -B cmake-build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target gfx-research-alpha -j 30
```

## Publication benchmark

No PowerShell runner scripts are required. From the build directory run:

```powershell
.\gfx-research-alpha.exe --benchmark-paper --benchmark-output=results\publication
```

The benchmark writes:

```text
results/publication/quality.csv
results/publication/attribute_quality.csv
results/publication/timing_raw.csv
results/publication/timing_summary.csv
results/publication/overdraw.csv
results/publication/manifest.txt
results/publication/captures/
```

### Key paper captures

The publication suite now saves the same key cases for:

```text
classic-forward-fixed-order
object-sorted-forward
wboit
layered-wboit-8
deferred-sparse-layer-2-edge-aware
depth-peeling-reference-24
```

Forward captures are generated for the important overlap comparisons at `alpha=0.75`, for the four-layer `alpha=0.75` stress case, and the 256-light comparison. This allows figures to compare the proposed path directly against ordinary forward alpha rather than only OIT approximations.

## Analysis tools

The `tools/` directory contains the publication analysis utilities. Typical commands from the project root are:

```powershell
python .\tools\synthetic_visibility_study.py --output .\results\publication\synthetic_visibility.csv
python .\tools\evaluate_publication.py .\cmake-build-release\results\publication
```

Other utilities provide edge, spectrum, temporal-stability, memory and attribute-correlation analysis.

## Source layout

```text
cmake/       CPM bootstrap
include/     public/local C++ headers
src/         renderer and benchmark harness
shaders/     shaders used by the current executable
scripts/     none; direct CLI invocation is used
tests/       CPU reference-math tests
tools/       result-analysis utilities
```

## Notes on novelty

The project does not claim dithering, forward alpha blending, WBOIT, reconstruction-before-lighting, bilateral filtering, binary threshold inversion or blue noise as individually novel. The research question is the bounded sparse-layer formulation combining known-threshold coverage observations, censored deeper-layer evidence, attribute-space reconstruction and deferred lighting.
