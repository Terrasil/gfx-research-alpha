# gfx-research-alpha

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/20)
[![OpenGL 4.6](https://img.shields.io/badge/OpenGL-4.6-5586A4.svg)](https://www.opengl.org/)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C.svg)](https://cmake.org/)

A deferred transparency testbed for reconstructing two censored layers from sparse alpha-tested samples.

[Method](#method) · [Build](#build) · [Controls](#controls) · [Experiments](#experiments) · [References](#references)

## Method

A fragment for surface $k$ is eligible after its known threshold $\tau_k(x)$ is satisfied. Nearer eligible fragments hide evidence for layers behind them. The reconstruction estimates $\widehat\alpha_f$ and $\widehat\alpha_b$ from the surviving local samples and the known thresholds.

```math
\begin{aligned}
I_k(x)&=\mathbf 1[\tau_k(x)<\alpha_k(x)],\\
C&=\widehat\alpha_f C_f+(1-\widehat\alpha_f)
       \left[\widehat\alpha_b C_b+(1-\widehat\alpha_b)C_0\right],\\
A&=1-(1-\widehat\alpha_f)(1-\widehat\alpha_b).
\end{aligned}
```

The coupled decoder multiplies estimated opacity by a geometry-confidence factor. The uncoupled decoder writes the opacity estimate directly once the layer passes the existence checks. The packed variant uses the uncoupled decoder with `RGBA8` albedo/roughness and `RGB10_A2` normals with binary validity. Position and control retain `RGBA16F`.

| Method | Additional production render-target footprint |
|---|---:|
| WBOIT | 10 B/px |
| Eight-bin Layered WBOIT in this implementation | 94 B/px |
| Two reconstructed layers | 116 B/px |
| Two reconstructed layers with packed materials | 100 B/px |

The table covers formats used by each path. A shared opaque G-buffer and lighting target, output buffers and benchmark-only reference captures are outside that accounting.

## Build

Requirements are CMake 3.25+, a C++20 compiler, Ninja, Python 3 with Jinja2, and a desktop OpenGL 4.6 Core driver. The initial CMake configuration downloads the pinned gfx-research-base revision and its dependencies. A local checkout can be supplied with `-DGFX_RESEARCH_BASE_LOCAL_PATH=../gfx-research-base`.

```powershell
python -m pip install jinja2
cmake -S . -B out/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out/build --parallel
ctest --test-dir out/build --output-on-failure
```

On Linux, run the same build commands and omit `.exe` when launching an executable.

## Controls

Launch `gfx-research-alpha.exe` to edit the geometry, coverage, alpha, light count and camera.

**Tests** starts the packed-storage smoke, quick and publication profiles, plus the existing transparency suites. Each packed-storage run writes to a new subdirectory and displays its path.

## Experiments

```powershell
.\out\build\gfx-research-alpha.exe --benchmark-storage-study=smoke --benchmark-output=results/storage-smoke
python tools/analyze_storage.py results/storage-smoke
.\out\build\gfx-research-alpha.exe --benchmark-storage-study=paper --benchmark-output=results/storage-paper
python tools/analyze_storage.py results/storage-paper
```

The paper profile uses 1280 × 720, two light counts, six methods and five timed blocks. The geometry families include thin patterns, crossing surfaces and stacks of 2, 4 and 8 layers. Arms in a case share the same alpha, kernel and stable hash. A 24-layer depth peel supplies linear HDR reference images. The per-layer report measures signed opacity bias and attribute error on matched surface IDs, with counts for interpreting sparse back-layer support.

| Output | Contents |
|---|---|
| `storage_comparison.csv` | Quality, measured timing and analytical format storage for each arm |
| `attribute_quality.csv` | Front/back matched-pixel counts, signed alpha bias, normal, position and material errors |
| `timing_raw.csv` / `captures/` | Individual GPU samples and representative linear HDR captures |


## Opacity evidence ablation (additional experiment)

Two **new** modes use the same two-layer geometry, surface selection, material reconstruction, and compositing as the uncoupled variant. They differ only in the decoded opacity:

- `deferred-sparse-layer-2-occupancy-raw`: observed positive occupancy for both layers; negative front-layer censorship is not corrected. This is an intentionally biased control.
- `deferred-sparse-layer-2-occupancy-conditional`: front occupancy and back occupancy divided by local non-front coverage. This estimates conditional occupancy **without using the known-threshold bounds**.

The pre-existing coupled and uncoupled variants remain unchanged. The new variants have stable IDs **19 and 20**, and their paired results must be evaluated separately from the archived six-method storage study.

```powershell
.\out\build\gfx-research-alpha.exe --benchmark-storage-study=opacity-smoke --benchmark-output=results/opacity-smoke
.\out\build\gfx-research-alpha.exe --benchmark-storage-study=opacity-paper --benchmark-output=results/opacity-paper
python tools/analyze_opacity_ablation.py results/opacity-paper
```

The paper profile sweeps the same geometry/alpha/light settings as the storage study, with paired GPU timing and front/back attribute-quality CSV. It can test whether deterministic threshold inversion improves upon both occupancy controls **on the evaluated surfaces**; it cannot establish physical opacity for fully unobserved layers. Do not update any manuscript claims before verifying the generated run.

## References

1. Morgan McGuire and Louis Bavoil. [Weighted Blended Order-Independent Transparency](https://jcgt.org/published/0002/02/09/). JCGT, 2013.
2. Thomas Porter and Tom Duff. [Compositing Digital Images](https://doi.org/10.1145/800031.808606). SIGGRAPH, 1984.

## Exposure-corrected ordinal blend (experimental)

Method 21, `deferred-sparse-layer-2-exposure-ordinal`, changes only the back-layer occupancy prior in the ordinal blend. It retains the front rule, the threshold-derived evidence and validity rules, and decoupled opacity. The motivation is the observed residual bias of blending conditional threshold estimates with raw back occupancy. This is a hypothesis, not a measured improvement. The earlier four-arm opacity profiles remain unchanged. New `hybrid-smoke` and `hybrid-paper` profiles include all four controls plus this method, with 5 / 1350 timing cases respectively. They write the existing CSVs and matched depth-step captures. Use the Tests tab or the CLI from the project root:

```powershell
.\cmake-build-release\gfx-research-alpha.exe --benchmark-storage-study=hybrid-smoke --benchmark-output=results/hybrid-smoke
.\cmake-build-release\gfx-research-alpha.exe --benchmark-storage-study=hybrid-paper --benchmark-output=results/hybrid-paper
python tools/analyze_opacity_ablation.py results/hybrid-paper
```
