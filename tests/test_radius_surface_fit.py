import numpy as np
import pytest

from vcsencode.models import Mesh3D
from vcsencode.centerline import Polyline3D, fit_centerline_bspline
from vcsencode.geom.frames import compute_rmf


def _synthetic_cylinder_mesh(radius=10.0, height=120.0, sections=180):
    trimesh = pytest.importorskip("trimesh")
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    tm.apply_translation([0.0, 0.0, height / 2.0])  # shift to z in [0, height]
    return Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units="mm")


def _straight_centerline_bspl(L=120.0, npts=241, Lc=9):
    z = np.linspace(0.0, L, npts)
    P = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    pl = Polyline3D(points=P)
    return fit_centerline_bspline(pl, L=Lc, constant_speed=True)


def test_radius_surface_matches_cylinder():
    pytest.importorskip("trimesh")
    from vcsencode.encoding.forward import fit_radius_surface

    R = 12.5
    L = 150.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=240)
    cl = _straight_centerline_bspl(L=L, npts=401, Lc=9)
    rmf = compute_rmf(cl, step_mm=5.0)

    rs, _, _ = fit_radius_surface(mesh, cl, rmf, K=13, R=11, thetas=128, tau_samples=80)

    # Sample random (tau,theta) and compare
    rng = np.random.default_rng(123)
    taus = rng.uniform(0.0, 1.0, 100)
    thetas = rng.uniform(0.0, 2*np.pi, 100)

    errs = []
    for t, th in zip(taus, thetas):
        rho_hat = rs.rho(t, th)
        errs.append(abs(rho_hat - R))
    errs = np.array(errs)

    assert np.nanmean(errs) < 0.05
    assert np.nanmax(errs) < 0.25


def test_theta_seam_continuity():
    pytest.importorskip("trimesh")
    from vcsencode.encoding.forward import fit_radius_surface

    R = 10.0
    L = 100.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=180)
    cl = _straight_centerline_bspl(L=L, npts=201, Lc=9)
    rmf = compute_rmf(cl, step_mm=5.0)

    rs, _, _ = fit_radius_surface(mesh, cl, rmf, K=11, R=9, thetas=90, tau_samples=60)

    # Evaluate at seam θ=0 vs θ=2π for several τ; values should match closely
    taus = np.linspace(0.05, 0.95, 10)
    d = []
    for t in taus:
        d.append(abs(rs.rho(t, 0.0) - rs.rho(t, 2*np.pi)))
    d = np.array(d)
    assert np.nanmax(d) < 5e-3
