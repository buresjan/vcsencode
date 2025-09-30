from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..models import Mesh3D, Polyline3D, CenterlineBSpline

# VTK / VMTK imports are local to functions to keep import time light.


# ----------------------------- VTK helpers -----------------------------

def _mesh3d_to_vtk_polydata(mesh: Mesh3D):
    """Convert Mesh3D -> vtkPolyData with triangle polys."""
    import vtk
    from vtk.util import numpy_support as vtk_np

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Mesh3D expects vertices (N,3) and faces (M,3).")

    # Points
    pts = vtk.vtkPoints()
    pts.SetData(vtk_np.numpy_to_vtk(v, deep=True))

    # Triangles as a connectivity array: [3, i, j, k, 3, ...]
    ntri = f.shape[0]
    cells = np.empty((ntri, 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = f
    cells_id = vtk_np.numpy_to_vtkIdTypeArray(cells.ravel(), deep=True)

    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(ntri, cells_id)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cell_array)
    return poly


def _vtk_centerline_to_longest_polyline_points(centerlines_poly):
    """
    Extract the longest polyline from a vtkPolyData with polyline cells.
    Returns (N,3) float64 array of points.
    """
    import vtk
    from vtk.util import numpy_support as vtk_np

    pts_vtk = centerlines_poly.GetPoints()
    if pts_vtk is None or pts_vtk.GetNumberOfPoints() == 0:
        raise RuntimeError("Centerlines output has no points.")
    pts = np.array([pts_vtk.GetPoint(i) for i in range(pts_vtk.GetNumberOfPoints())], dtype=np.float64)

    lines = centerlines_poly.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        # Sometimes lines are in Polys
        lines = centerlines_poly.GetPolys()
    if lines is None or lines.GetNumberOfCells() == 0:
        # Fallback: treat all points as a single polyline (unlikely)
        return pts

    data = vtk_np.vtk_to_numpy(lines.GetData())
    # Format: [n, id0, id1, ..., n, id0, ...]
    polylines = []
    i = 0
    ndata = data.shape[0]
    while i < ndata:
        cnt = int(data[i]); i += 1
        ids = data[i : i + cnt]; i += cnt
        poly_pts = pts[np.asarray(ids, dtype=np.int64)]
        polylines.append(poly_pts)

    # Choose the polyline with the largest arclength
    def _length(P):
        if P.shape[0] < 2:
            return 0.0
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        return float(d.sum())

    lengths = [ _length(P) for P in polylines ]
    idx = int(np.argmax(lengths))
    return polylines[idx].astype(np.float64)


def _snap_to_surface_vertex(mesh: Mesh3D, p: np.ndarray) -> np.ndarray:
    """Snap a 3D point to the nearest mesh vertex; returns (3,) float64."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    di = np.argmin(np.sum((v - p[None, :])**2, axis=1))
    return v[di].astype(np.float64)


# --------------------------- Public API funcs ---------------------------

def compute_centerline_vmtk(
    mesh: Mesh3D,
    p_source: np.ndarray,
    p_target: np.ndarray,
    resampling: float | None = 0.5,
) -> Polyline3D:
    """
    Compute a centerline polyline using VMTK with source/target seed points.

    Args:
        mesh: watertight, capped Mesh3D
        p_source, p_target: seeds near inlet/outlet; will be snapped to surface vertices
        resampling: step length in mm for VMTK resampling (default 0.5 mm).
                    If None, VMTK is asked not to resample (not recommended).

    Returns:
        Polyline3D with points ordered from source to target.

    Raises:
        ImportError if VMTK is not available.
        RuntimeError on VMTK processing errors.
    """
    try:
        from vmtk import vmtkscripts
    except Exception as e:
        raise ImportError("VMTK is required. Please ensure 'vmtk' is installed in the active environment.") from e

    poly = _mesh3d_to_vtk_polydata(mesh)

    # Use the **interior** seeds as provided (do NOT snap to surface).
    src = np.asarray(p_source, dtype=np.float64)
    tgt = np.asarray(p_target, dtype=np.float64)

    # Configure vmtkCenterlines
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = poly
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(map(float, src.tolist()))   # inlet/on-surface
    cl.TargetPoints = list(map(float, tgt.tolist()))   # outlet/on-surface
    cl.AppendEndPoints = 1
    cl.CapDisplacement = 0.0
    # These options improve robustness on simple tubular segments
    cl.SimplifyVoronoi = 1
    cl.StopFastMarchingOnReachingTarget = 1
    if resampling is None:
        cl.Resampling = 0
    else:
        cl.Resampling = 1
        cl.ResamplingStepLength = float(resampling)

    print(f"[vmtkcenterlines] seeds:\n  src={src}\n  tgt={tgt}\n  resampling={resampling}")
    cl.Execute()
    centerlines_poly = cl.Centerlines
    if centerlines_poly is None:
        raise RuntimeError("VMTK returned no centerlines.")

    # Diagnostics about output
    n_pts = centerlines_poly.GetNumberOfPoints() if centerlines_poly is not None else 0
    n_cells = centerlines_poly.GetNumberOfCells() if centerlines_poly is not None else 0
    print(f"[vmtkcenterlines] output: points={n_pts}, cells={n_cells}")
    if n_pts == 0:
        raise RuntimeError("VMTK returned an empty centerline (0 points). Check seed locations and surface.")

    P = _vtk_centerline_to_longest_polyline_points(centerlines_poly)

    # Ensure orientation: start near src, end near tgt
    if np.linalg.norm(P[0] - tgt) < np.linalg.norm(P[0] - src):
        P = P[::-1].copy()

    return Polyline3D(points=P)


def extend_to_cap_centers(polyline: Polyline3D, cap_centers: np.ndarray, tol: float = 1e-6) -> Polyline3D:
    """
    Prolong the polyline so that its endpoints coincide with the two cap centers.

    cap_centers: (2,3) [min_end, max_end] or any order — we match by proximity.
    """
    P = np.asarray(polyline.points, dtype=np.float64)
    c0, c1 = np.asarray(cap_centers[0], dtype=np.float64), np.asarray(cap_centers[1], dtype=np.float64)

    # Match starts/ends by nearest cap
    d00 = np.linalg.norm(P[0]  - c0); d01 = np.linalg.norm(P[0]  - c1)
    d10 = np.linalg.norm(P[-1] - c0); d11 = np.linalg.norm(P[-1] - c1)

    # Determine which cap is start
    if (d00 + d11) <= (d01 + d10):
        start_cap, end_cap = c0, c1
    else:
        start_cap, end_cap = c1, c0
        P = P[::-1].copy()

    if np.linalg.norm(P[0] - start_cap) > tol:
        P = np.vstack([start_cap[None, :], P])
    if np.linalg.norm(P[-1] - end_cap) > tol:
        P = np.vstack([P, end_cap[None, :]])

    return Polyline3D(points=P)


def fit_centerline_bspline(polyline: Polyline3D, L: int = 9, constant_speed: bool = True) -> CenterlineBSpline:
    """
    Fit a cubic B-spline to the centerline polyline.
    If constant_speed=True, parameter τ is defined as normalized arc length in [0,1].

    Returns:
        CenterlineBSpline with degree=3, uniform interior knots on [0,1], and L control points.
    """
    from scipy.interpolate import make_lsq_spline

    P = np.asarray(polyline.points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        raise ValueError("Polyline3D must contain at least 2 points of shape (N,3).")

    # Deduplicate extremely close points
    d = np.zeros(P.shape[0], dtype=bool)
    d[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) < 1e-9
    if d.any():
        P = P[~d]

    # Arc-length parameterization τ∈[0,1]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    length = float(seg.sum())
    if length <= 0:
        raise ValueError("Polyline has zero length.")
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tau = (s / s[-1]).astype(np.float64)

    # Ensure monotonic unique τ
    keep = np.r_[True, np.diff(tau) > 1e-12]
    tau = tau[keep]
    P = P[keep]

    # Spline degree
    k = 3
    # Number of control points (at least k+1, at most number of data)
    n_coeff = int(np.clip(L, k + 1, max(k + 1, P.shape[0])))
    # Interior knots count
    n_int = max(n_coeff - k - 1, 0)
    if n_int > 0:
        # Uniform interior knots in (0,1)
        t_int = np.linspace(0.0, 1.0, n_int + 2, dtype=np.float64)[1:-1]
    else:
        t_int = np.array([], dtype=np.float64)

    # Build the **full open-clamped** knot vector: [0,...,0, t_int..., 1,...,1]
    knots = np.r_[np.zeros(k + 1, dtype=np.float64), t_int, np.ones(k + 1, dtype=np.float64)]

    # LSQ fit per coordinate using the SAME knot vector for x,y,z
    splx = make_lsq_spline(tau, P[:, 0], knots, k=k)
    sply = make_lsq_spline(tau, P[:, 1], knots, k=k)
    splz = make_lsq_spline(tau, P[:, 2], knots, k=k)
    coeffs = np.stack([splx.c, sply.c, splz.c], axis=1)  # shape (n_coeff, 3)

    return CenterlineBSpline(
        degree=k,
        knots=np.asarray(knots, dtype=np.float64),
        coeffs=np.asarray(coeffs, dtype=np.float64),
    )
