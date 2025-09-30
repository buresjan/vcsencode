import os
import numpy as np
import pytest

from vcsencode.io import load_stl, clean_mesh, detect_caps, cap_centers, inward_seed_points

STL_PATH = "vessel_segment.stl"

vmtk_available = True
try:
    from vmtk import vmtkscripts  # noqa: F401
except Exception:
    vmtk_available = False


@pytest.mark.skipif(not os.path.exists(STL_PATH), reason="vessel_segment.stl not found at repo root")
@pytest.mark.skipif(not vmtk_available, reason="VMTK not importable in this environment")
def test_centerline_bspline_pipeline():
    from vcsencode.centerline import compute_centerline_vmtk, extend_to_cap_centers, fit_centerline_bspline

    mesh = clean_mesh(load_stl(STL_PATH), repair=True)
    caps = detect_caps(mesh)
    centers = cap_centers(caps)
    p0, p1 = inward_seed_points(mesh, caps, offset_mm=2.0)

    pl = compute_centerline_vmtk(mesh, p0, p1, resampling=0.5)
    pl = extend_to_cap_centers(pl, centers)
    cl = fit_centerline_bspline(pl, L=9, constant_speed=True)

    # Endpoints should be near cap centers
    c_start = cl.eval(0.0)
    c_end = cl.eval(1.0)
    assert np.linalg.norm(c_start - centers[0]) < 5.0  # mm tolerance (mesh-scale dependent)
    assert np.linalg.norm(c_end - centers[1]) < 5.0

    # Tangent is unit length
    T = cl.tangent(np.linspace(0, 1, 11))
    n = np.linalg.norm(T, axis=1)
    assert np.all(np.isfinite(n)) and np.allclose(n, 1.0, atol=1e-6)

    # Length positive
    assert cl.length() > 0.0
