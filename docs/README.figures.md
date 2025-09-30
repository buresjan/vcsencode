# README.figures.md

## Purpose

This document explains how to generate **publication‑ready figures** strictly **from the VCS encoding** produced by `vcsencode`. It reproduces the two‑panel layout of the target figure:

- **Left panel**: 3D vessel with centerline and one local frame glyph, plus two reference curves.
- **Right panel**: \((\tau,\theta)\) rectangle colored by \(\rho_w(\tau,\theta)\) (mm), with the same reference curves.

The figure generator accepts only `model.npy` (encoding vector) and `meta.json` (metadata).

---

## Inputs

- `model.npy` – fixed‑length encoding vector \( \mathbf{a} \).
- `meta.json` – degrees, knot vectors, units, periodicity, frame initialization rule, and versions.

These are created by:

```bash
vcsencode encode vessel_segment.stl --out-npy model.npy --out-meta meta.json
```

---

## One‑command figure (CLI)

```bash
# Composite figure saved as PNG (left: 3D, right: τ–θ map)
vcsencode encode vessel_segment.stl \
  --out-npy model.npy \
  --out-meta meta.json \
  --fig fig3.png
```

**Outputs**

- `fig3.png` — composite image (300–600 dpi ready; see below for DPI control).
- `model.npy`, `meta.json` — persisted for reuse.

---

## Python API (from encoding only)

```python
import json
import numpy as np
from vcsencode.encoding.forward import VCSModel
from vcsencode.visualize.figures import figure_vcs_overview

# Load encoding and metadata
a = np.load("model.npy")
meta = json.load(open("meta.json"))

# Rebuild the model (centerline BSpline, RMF, and ρ surface)
model = VCSModel.unpack(a, meta)

# Generate the two-panel figure and save
figset = figure_vcs_overview(model, show_frames=True, style="paper")

# Raster (PNG) at journal quality
figset.save("fig3.png", dpi=600)

# Vector export (SVG or EPS)
figset.save("fig3.svg")
# figset.save("fig3.eps")
```

---

## What the function draws

- **Left panel (3D)**: reconstructed surface \(\mathbf{x}(\tau,\theta)\), centerline \(\mathbf{c}(\tau)\) (orange), one local RMF glyph \(\{\mathbf{t},\mathbf{v}_1,\mathbf{v}_2\}\) (red), and two yellow reference curves:
  - a constant‑\(\theta\) line (default \(\theta=5\pi/4\)),
  - a constant‑\(\tau\) line (default \(\tau=1/5\)).
- **Right panel (2D)**: \([0,1]\times[0,2\pi]\) heatmap of \(\rho_w(\tau,\theta)\) with the same two yellow curves and a colorbar labeled “Rho (mm)”.

---

## Reproducibility & style

- **Units**: millimeters; colorbar labeled “Rho (mm)”.
- **Camera**: deterministic azimuth/elevation and window size.
- **Lighting**: distant key + mild ambient; shadows off.
- **Colormap**: sequential, print‑friendly.
- **Lines/markers**: centerline **orange**, local frame **red**, reference curves **yellow**.
- **Fonts**: sans‑serif; sizes in points to be DPI‑independent.
- All non‑derivable conventions (e.g., initial frame rule) are stored in `meta.json`.

---

## DPI and vector export

- **PNG**: set DPI at save time:
  ```python
  figset.save("fig3.png", dpi=600)  # use 300–600 for journals
  ```
- **Vector**: `figset.save("fig3.svg")` (or `.eps`) writes a vector composite.  
  If your toolchain prefers separate panels, export each and combine in your DTP software.

---

## Residual quality‑control figure (optional)

```bash
# Computes per-vertex residual r(p) and writes a QC report/figure
vcsencode qc vessel_segment.stl model.npy meta.json --fig residuals.png --report qc.txt
```

---

## Troubleshooting

- **Jagged PNG**: increase `dpi` (e.g., 600) when saving.
- **Large SVG**: prefer EPS, or rasterize text layers if your publisher requests it.
- **Seam artifacts**: ensure the θ‑periodic option was used during fitting (handled by the encoder by default).

---

## Minimal end‑to‑end example

```bash
# 1) Encode once
vcsencode encode vessel_segment.stl --out-npy model.npy --out-meta meta.json

# 2) Make the figure from the encoding only
python - <<'PY'
import json, numpy as np
from vcsencode.encoding.forward import VCSModel
from vcsencode.visualize.figures import figure_vcs_overview
a = np.load("model.npy"); meta = json.load(open("meta.json"))
model = VCSModel.unpack(a, meta)
figure_vcs_overview(model).save("fig3.png", dpi=600)
PY
```

---
