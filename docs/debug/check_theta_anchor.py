from vcsencode.io import load_stl, clean_mesh
from vcsencode.encoding.forward import build_model
from vcsencode.encoding.inverse import export_stl

mesh = clean_mesh(load_stl("vessel_segment.stl"), repair=True)
model = build_model(mesh, params=dict(
    unit_scale=10.0,
    pad_ends_mm=60.0,
    L=15, K=31, R=25,
    resampling=0.20,
    rays_thetas=512, rays_tau_samples=240,
    theta_anchor="rho_argmax"
))
print("theta_offset (rad):", model.meta.get("theta_offset"))
export_stl(model, "reconstructed_theta_anchor.stl", n_tau=360, n_theta=540, cap_mode="autoslice")
print("Wrote reconstructed_theta_anchor.stl")
