from vcsencode.io import load_stl, clean_mesh
from vcsencode.encoding.forward import build_model
from vcsencode.encoding.inverse import export_stl
from vcsencode.encoding.metrics import residuals

mesh = clean_mesh(load_stl("vessel_segment.stl"), repair=True)
model = build_model(mesh, params=dict(
    unit_scale=10.0,
    pad_ends_mm=60.0,      # or None for auto
    L=9, K=19, R=15,
    resampling=0.25,
    rays_thetas=256, rays_tau_samples=160
))
export_stl(model, "reconstructed_autocap.stl", n_tau=320, n_theta=420, cap_mode="autoslice")
print("Wrote reconstructed_autocap.stl")
print(residuals(mesh, model, max_vertices=20000).summary)
