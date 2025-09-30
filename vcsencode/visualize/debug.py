from __future__ import annotations
import numpy as np
import pyvista as pv

from ..models import Mesh3D, Polyline3D, CenterlineBSpline


def _mesh3d_to_pv(mesh: Mesh3D) -> pv.PolyData:
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int32)
    # PyVista expects a flattened faces array with a leading count per face
    n = F.shape[0]
    faces = np.hstack([np.full((n, 1), 3, dtype=np.int32), F]).ravel()
    return pv.PolyData(V, faces)


def _line_from_points(P: np.ndarray) -> pv.PolyData:
    """
    Build a PolyData polyline from a sequence of points.
    Compatible with modern PyVista (no .lines_from_points).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    # PolyData expects a connectivity array: [n, 0,1,2,...,n-1]
    cells = np.hstack([[n], np.arange(n, dtype=np.int32)])
    return pv.PolyData(P, lines=cells)


def show_centerline_overlay(
    mesh: Mesh3D,
    polyline: Polyline3D,
    centerline: CenterlineBSpline | None = None,
    cap_centers: np.ndarray | None = None,
    seeds: tuple[np.ndarray, np.ndarray] | None = None,
    screenshot: str | None = "centerline_debug.png",
    window_size: tuple[int, int] = (1400, 900),
    theme: str = "document",
) -> pv.Plotter:
    """
    Render the vessel surface with the extracted polyline centerline and (optionally)
    the fitted spline centerline and diagnostic markers. If `screenshot` is provided,
    save the image and close the window (off-screen render).
    """
    pv.set_plot_theme(theme)
    pl = pv.Plotter(off_screen=screenshot is not None, window_size=window_size)

    # Mesh
    surf = _mesh3d_to_pv(mesh)
    pl.add_mesh(surf, color="#d9d9d9", smooth_shading=True, ambient=0.25, specular=0.2)

    # Polyline from VMTK — render as a thin tube so it is always visible
    line_pl = _line_from_points(polyline.points)
    # tube radius ~ 1% of bbox diagonal, clamped into [0.2, 1.5] mm for visibility
    diag = float(np.linalg.norm(surf.bounds[1] - surf.bounds[0]))
    r_tube = np.clip(0.01 * diag, 0.2, 1.5)
    try:
        line_tube = line_pl.tube(radius=r_tube, n_sides=16)
        pl.add_mesh(line_tube, color="#ff8c00", specular=0.3, name="centerline_polyline_tube")
    except Exception:
        pl.add_mesh(line_pl, color="#ff8c00", line_width=6, name="centerline_polyline")

    # Optional: spline resample for smooth curve
    if centerline is not None:
        ts = np.linspace(0, 1, 400)
        Cs = centerline.eval(ts)
        line_spl = _line_from_points(Cs)
        try:
            line_spl_tube = line_spl.tube(radius=0.8 * r_tube, n_sides=16)
            pl.add_mesh(line_spl_tube, color="#00a3ff", specular=0.3, name="centerline_spline_tube")
        except Exception:
            pl.add_mesh(line_spl, color="#00a3ff", line_width=4, name="centerline_spline")

    # Markers: cap centers & seeds
    if cap_centers is not None:
        for c in cap_centers:
            pl.add_mesh(pv.Sphere(radius=0.5, center=c), color="#ffd700", name="cap_center")
    if seeds is not None:
        p0, p1 = seeds
        pl.add_mesh(pv.Sphere(radius=0.7, center=p0), color="#ff3333", name="seed0")
        pl.add_mesh(pv.Sphere(radius=0.7, center=p1), color="#33ff66", name="seed1")

    pl.add_axes(line_width=2)
    # Ensure perspective projection
    try:
        pl.enable_parallel_projection()
        pl.camera.parallel_projection = False
    except Exception:
        try:
            pl.camera.parallel_projection = False
        except Exception:
            pass
    pl.camera.zoom(1.2)

    if screenshot:
        pl.show(screenshot=screenshot, auto_close=True)
    else:
        pl.show()

    return pl
