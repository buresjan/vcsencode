import numpy as np
from vcsencode.io import load_stl, clean_mesh
from vcsencode.encoding.forward import build_model
from vcsencode.encoding.inverse import export_stl
from vcsencode.encoding.metrics import residuals

# Tune pad_ends_mm; if None => automatic (≈ 3× median radius, clamped to [30,200] mm)
PAD_LEN = None

mesh = clean_mesh(load_stl("vessel_segment.stl"), repair=True)
model = build_model(mesh, params=dict(
    unit_scale=10.0,       # adjust if needed
    pad_ends_mm=PAD_LEN,   # None (auto) or a number, e.g. 60.0
    L=9, K=19, R=15,
    resampling=0.25,
    rays_thetas=256, rays_tau_samples=160
))
export_stl(model, "reconstructed_padded_cropped.stl", n_tau=320, n_theta=420)
print("Wrote reconstructed_padded_cropped.stl")
print(residuals(mesh, model, max_vertices=20000).summary)
