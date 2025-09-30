from __future__ import annotations

from math import tau as TWO_PI

import numpy as np
import pytest

from vcsencode.centerline import Polyline3D, fit_centerline_bspline
from vcsencode.encoding.forward import fit_radius_surface
from vcsencode.geom.frames import compute_rmf
from vcsencode.models import Mesh3D


def _synthetic_cylinder_mesh(radius: float = 10.0, height: float = 120.0, sections: int = 180) -> Mesh3D:
    trimesh = pytest.importorskip("trimesh")
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    tm.apply_translation([0.0, 0.0, height / 2.0])
    return Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units="mm")


def _straight_centerline_bspline(length: float = 120.0, samples: int = 241, control_points: int = 9):
    z = np.linspace(0.0, length, samples, dtype=float)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    pl = Polyline3D(points=pts)
    return fit_centerline_bspline(pl, L=control_points, constant_speed=True)


def test_eval_grid_theta_periodic_matches_seam():
    pytest.importorskip("trimesh")
    pytest.importorskip("rtree")

    mesh = _synthetic_cylinder_mesh(radius=12.0, height=100.0, sections=200)
    cl = _straight_centerline_bspline(length=100.0, samples=321, control_points=11)
    rmf = compute_rmf(cl, step_mm=5.0)

    rs, _, _ = fit_radius_surface(
        mesh,
        cl,
        rmf,
        K=15,
        R=12,
        thetas=144,
        tau_samples=90,
        theta_periodic=True,
    )

    taus = np.linspace(0.0, 1.0, 41)
    thetas = np.linspace(0.0, TWO_PI, 129, endpoint=True)
    grid = rs.eval_grid(taus, thetas)
    seam_diff = np.abs(grid[:, 0] - grid[:, -1])
    assert float(np.max(seam_diff)) < 1e-6
