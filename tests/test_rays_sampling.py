import numpy as np
import pytest
import trimesh

from vcsencode.models import Mesh3D
from vcsencode.centerline import fit_centerline_bspline, Polyline3D
from vcsencode.geom.frames import compute_rmf
from vcsencode.geom.rays import cast_radius


def _synthetic_cylinder_mesh(radius=10.0, height=120.0, sections=128):
    # Create an open cylinder aligned with z, centered at origin [-h/2, +h/2]
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # Shift to [0, height] along z to match a 0..L centerline if desired
    tm.apply_translation([0.0, 0.0, height / 2.0])
    return Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units="mm")


def _straight_centerline_spline(L=120.0, n=201):
    z = np.linspace(0.0, L, n)
    P = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    pl = Polyline3D(points=P)
    return fit_centerline_bspline(pl, L=9, constant_speed=True)


def test_cast_radius_matches_cylinder_radius():
    R = 10.0
    L = 120.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=180)
    cl = _straight_centerline_spline(L=L, n=241)
    rmf = compute_rmf(cl, step_mm=5.0)

    # Sample radii at multiple taus, avoid the very ends
    taus = [0.05, 0.5, 0.95]
    thetas = np.linspace(0.0, 2*np.pi, 361)[:-1]  # 0..2π exclusive
    for t in taus:
        d = cast_radius(mesh, cl, rmf, t, thetas)
        assert np.all(np.isfinite(d)), f"NaNs present at tau={t}"
        # Mean and spread should match radius closely
        assert abs(np.nanmean(d) - R) < 1e-2
        assert np.nanmax(np.abs(d - R)) < 5e-2  # allow small discretization error


def test_cast_radius_handles_missing_hits_gracefully():
    # Extremely coarse mesh may miss some directions; function should not crash.
    R = 10.0
    L = 50.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=16)
    cl = _straight_centerline_spline(L=L, n=51)
    rmf = compute_rmf(cl, step_mm=10.0)
    thetas = np.linspace(0.0, 2*np.pi, 33)[:-1]
    d = cast_radius(mesh, cl, rmf, 0.5, thetas)
    # Should return a vector of length len(thetas) with finite or NaN values
    assert d.shape == (len(thetas),)
    assert np.any(np.isfinite(d))
