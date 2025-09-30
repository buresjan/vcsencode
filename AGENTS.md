# AGENTS.md — Operating Manual for Codex

## Mission
Implement a robust, reversible Vessel Coordinate System (VCS) encoder/decoder as a Python package **vcsencode** that:
1) takes a capped tubular STL -> computes centerline (VMTK), RMF frame, and fits splines to produce a fixed-length numpy encoding vector **a** plus metadata, following Romero et al., Applied Mathematics and Computation 487 (2025), Eqs. (1)–(10). Target Figure style = their Fig. 3 (left 3D + right τ–θ map).  
2) runs in reverse: encoding -> triangulated wall (+ caps) -> STL.  
3) includes a CLI and unit tests; visualizations must work **from the encoding only**.  
Paper is in `docs/vcs_romero.pdf` and `docs/vcs_romero.md` (equations & figures referenced below).

## Canonical references
- Definitions & equations: Romero et al. (VCS), Eqs. (1)–(4) for coordinates; (5)–(9) for model & encoding vector; (10) for residuals; Fig. 3 for plots.  
- Centerlines: use **VMTK** (Python) rather than custom A*.  
- Frames: **Rotation Minimizing Frame** (parallel transport) via **double reflection**; avoid Frenet.  
- Fitting: `scipy.interpolate.BSpline` (centerline), `RectBivariateSpline` (ρ_w on τ–θ grid) with explicit θ periodic seam handling.

## Package architecture (create this exact layout)
```
vcsencode/
  __init__.py
  io/                     # mesh I/O + hygiene + cap detection
  centerline/             # VMTK wrapper + spline fit + prolongation
  geom/                   # frames (RMF), projections (τ,θ,ρ), rays
  encoding/               # forward (fit), inverse (mesh), metrics
  visualize/              # Fig.3-style and QC figures
  cli/                    # Typer-based CLI
  tests/                  # pytest unit/integration tests
  resources/              # colormaps, style sheets
docs/
  README.figures.md
  vcs_romero.pdf
  vcs_romero.md
pyproject.toml
README.md
LICENSE
environment.yml
.pre-commit-config.yaml
```

## Data models (Python dataclasses)
- `Mesh3D`: vertices (n,3), faces (m,3), units="mm".
- `CenterlineBSpline`: `degree=3`; uniform `knots` over [0,1]; `coeffs` `(L,3)`; methods: `eval(t), tangent(t), resample(n), length()`.
- `RMF`: callables t(tau), v1(tau), v2(tau) computed with double-reflection; deterministic initial v1(0) rule.
- `RadiusSurfaceBSpline`: degrees (3,3); uniform knots in τ,θ; coeffs (K,R); θ is periodic.
- `VCSModel`: `centerline`, `frame` (derived), `radius` surface; `meta`: {units, L,K,R, knot vectors, degrees, θ_periodic, τ_param="arc-length", init_frame_rule, software_versions}.  
  `pack() -> np.ndarray` equals Eq. (9) ordering; `unpack(a, meta) -> VCSModel`.

## Algorithmic requirements (must implement)
1) **Mesh hygiene** (Trimesh): watertightness, normals, self-intersection check; unit normalization to mm; component check; optional repairs.
2) **Cap detection**: find two nearly planar end patches; fit planes; area-weighted centers; seeds offset inward along normals.
3) **Centerline**: use `vmtkcenterlines` with seeds; trim to segment; **prolong** to the exact cap centers if needed.
4) **Centerline BSpline**: cubic, uniform knots (L default 9); reparameterize to constant speed τ∈[0,1].
5) **Frame**: Rotation-minimizing frame (double-reflection); stable initial v1(0): set so that wall centroid lies in plane {t(0), v1(0)}.
6) **Projection**: closest-point τ by solving t(τ)·(c(τ)-x)=0 with robust bracketing+Newton; then ρ=||x-c(τ)||, θ = atan2( (x-c)·v2, (x-c)·v1 ).
7) **Radius sampling**: For a grid of τ_i (aligned to radius-surface knots K default 19) and dense θ_j, ray-cast from c(τ_i) along v1 cosθ + v2 sinθ; require single positive hit (star-convex check). Use Trimesh with Embree acceleration.
8) **Fit ρ_w(τ,θ)**: cubic `RectBivariateSpline` with θ-periodic seam (tile or tie coefficients). Defaults: R=15 angular knots. Save coeffs+knots.
9) **Encoding vector**: Eq. (9): a = (c0,..,c_n, b_00,.., b_{K R}). Save alongside `meta.json`.
10) **Inverse**: evaluate x(τ,θ) per Eq. (8) on grid; triangulate, θ seam stitch; add planar caps at τ=0,1; export STL.
11) **Metrics & QC**: residual r(p)=||p - x(τ(p),θ(p))|| (Eq. 10) at mesh vertices; summary (mean, Q75) and color maps.
12) **Figures**: Reproduce Fig. 3 from *encoding only*:  
    - Left: 3D surface + centerline (orange) + one local frame glyph (red), two sample points/lines (yellow).  
    - Right: τ–θ rectangular map colored by ρ_w (mm), with two yellow lines (fixed θ and fixed τ).  
    Export PNG (300/600 dpi) and vector (SVG/EPS).

## CLI (Typer)
- `vcsencode encode IN.stl --out-npy model.npy --out-meta meta.json [--fig fig3.png] [--L 9 --K 19 --R 15]`
- `vcsencode decode model.npy meta.json --out-stl OUT.stl [--nt 200 --nθ 256]`
- `vcsencode qc IN.stl model.npy meta.json --fig residuals.png --report qc.txt`

## Checkpoints (must exist)
A) `encode` runs end-to-end on `vessel_segment.stl` and writes `model.npy`, `meta.json`; prints residual mean and Q75.  
B) `decode` reconstructs `reconstructed.stl` (watertight).  
C) `figure` reproduces paper-like Fig. 3 from encoding only.  
D) Unit tests pass on synthetic cylinders (round-trip Hausdorff small; RMF orthonormal; θ-seam continuity).

## Coding standards & tooling
- Python ≥3.11, type hints, dataclasses; `black`, `ruff`, `pytest`.  
- Logging (`logging`): INFO summarizing steps; DEBUG for counts/knots/residuals.  
- Determinism: seed RNG; log library versions in meta.

## Non-negotiables / pitfalls
- Use RMF (double-reflection), **never** Frenet.  
- τ must be constant-speed.  
- θ periodic seam must be continuous.  
- Store every non-derivable convention in `meta` to guarantee reversibility.

## Definition of Done (DoD)
- All checkpoints A–D pass on `vessel_segment.stl`.  
- Enc/dec round-trip residual < mesh resolution (target ~≤1 mm typical per paper); publish Fig. 3 clone.  
- CLI help, README, and docstring examples complete.
