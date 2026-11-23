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

## References

1. Morgan McGuire and Louis Bavoil. [Weighted Blended Order-Independent Transparency](https://jcgt.org/published/0002/02/09/). JCGT, 2013.
2. Thomas Porter and Tom Duff. [Compositing Digital Images](https://doi.org/10.1145/800031.808606). SIGGRAPH, 1984.
