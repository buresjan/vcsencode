import numpy as np
import pytest

from vcsencode.models import Mesh3D
from vcsencode.centerline import Polyline3D, fit_centerline_bspline
from vcsencode.geom.frames import compute_rmf

def _synthetic_cylinder_mesh(radius=10.0, height=120.0, sections=240):
    trimesh = pytest.importorskip("trimesh")
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    tm.apply_translation([0.0, 0.0, height / 2.0])  # z in [0, height]
    return Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units="mm")

def _straight_centerline(L=120.0, npts=401, Lc=9):
    z = np.linspace(0.0, L, npts)
    P = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    pl = Polyline3D(points=P)
    return fit_centerline_bspline(pl, L=Lc, constant_speed=True)

def test_inverse_surface_matches_cylinder_wall():
    pytest.importorskip("trimesh")
    from vcsencode.encoding.forward import fit_radius_surface
    from vcsencode.encoding.inverse import surface_mesh

    R = 15.0
    L = 200.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=360)
    cl = _straight_centerline(L=L, npts=401, Lc=9)
    rmf = compute_rmf(cl, step_mm=5.0)
    rs, _, _ = fit_radius_surface(mesh, cl, rmf, K=13, R=11, thetas=180, tau_samples=120)

    from vcsencode.models import VCSModel
    model = VCSModel(centerline=cl, radius=rs, meta={"units": "mm"})

    wall = surface_mesh(model, n_tau=240, n_theta=360)
    V = wall.vertices
    # Radial distance from axis (ignore caps)
    r = np.sqrt(V[:, 0]**2 + V[:, 1]**2)
    err = np.abs(r - R)
    assert float(np.nanmean(err)) < 0.2
    assert float(np.nanmax(err)) < 0.8

def test_export_stl_watertight(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from vcsencode.encoding.forward import fit_radius_surface
    from vcsencode.encoding.inverse import export_stl

    R = 8.0
    L = 80.0
    mesh = _synthetic_cylinder_mesh(radius=R, height=L, sections=180)
    cl = _straight_centerline(L=L, npts=241, Lc=9)
    rmf = compute_rmf(cl, step_mm=5.0)
    rs, _, _ = fit_radius_surface(mesh, cl, rmf, K=11, R=9, thetas=120, tau_samples=80)

    from vcsencode.models import VCSModel
    model = VCSModel(centerline=cl, radius=rs, meta={"units": "mm"})

    out_path = tmp_path / "reconstructed.stl"
    export_stl(model, str(out_path), n_tau=160, n_theta=256)
    assert out_path.exists() and out_path.stat().st_size > 0

    tm = trimesh.load_mesh(str(out_path))
    assert isinstance(tm, trimesh.Trimesh)
    assert tm.is_watertight
