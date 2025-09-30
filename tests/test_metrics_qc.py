import numpy as np
import pytest

from vcsencode.models import Mesh3D, VCSModel
from vcsencode.centerline import Polyline3D, fit_centerline_bspline
from vcsencode.geom.frames import compute_rmf
from vcsencode.encoding.metrics import residuals


def _synthetic_cylinder_mesh(radius=10.0, height=120.0, sections=240):
    trimesh = pytest.importorskip("trimesh")
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    tm.apply_translation([0.0, 0.0, height / 2.0])
    return Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units="mm")


def _straight_centerline(L=120.0, npts=401, Lc=9):
    z = np.linspace(0.0, L, npts)
    P = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    pl = Polyline3D(points=P)
    return fit_centerline_bspline(pl, L=Lc, constant_speed=True)


def test_residuals_small_on_self_fit():
    pytest.importorskip("trimesh")
    from vcsencode.encoding.forward import fit_radius_surface
    R = 10.0
    L = 120.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=300)
    cl = _straight_centerline(L=L, npts=401, Lc=9)
    rmf = compute_rmf(cl, step_mm=5.0)
    rs, _, _ = fit_radius_surface(mesh, cl, rmf, K=11, R=9, thetas=120, tau_samples=80)

    meta = {
        "units": "mm",
        "k_tau": rs.k_tau,
        "k_theta": rs.k_theta,
        "knots_tau": rs.knots_tau.tolist(),
        "knots_theta": rs.knots_theta.tolist(),
        "theta_periodic": rs.theta_periodic,
        "cl_degree": cl.degree,
        "cl_knots": cl.knots.tolist(),
    }
    model = VCSModel(centerline=cl, radius=rs, meta=meta)

    res = residuals(mesh, model, max_vertices=5000, chunk=2000)
    s = res.summary
    assert s["mean"] < 0.2
    assert s["p95"] < 0.8
