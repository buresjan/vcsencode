from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import trimesh
from trimesh import Trimesh

from ..models import Mesh3D


# ---------- Internal utilities ----------

def _to_trimesh(mesh: Mesh3D) -> Trimesh:
    """Convert Mesh3D -> trimesh.Trimesh without additional processing."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return tm


def _from_trimesh(tm: Trimesh, units: str = "mm") -> Mesh3D:
    """Convert trimesh.Trimesh -> Mesh3D."""
    tm.remove_unreferenced_vertices()
    return Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units=units)


def _principal_axis(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute centroid and principal (lengthwise) axis using PCA.
    Returns (centroid, axis_unit).
    """
    c = vertices.mean(axis=0)
    X = vertices - c
    # SVD of covariance surrogate
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    axis = Vt[0]
    n = np.linalg.norm(axis)
    if n == 0.0:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / n
    return c, axis


# ---------- Public API ----------

def load_stl(path: str) -> Mesh3D:
    """
    Load an STL file and return Mesh3D in millimeters.
    Notes:
      - If the file is a Scene, geometries will be concatenated.
      - No unit scaling is performed; we assume mm.
    """
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        if len(obj.geometry) == 0:
            raise ValueError(f"No geometry found in scene: {path}")
        tm = trimesh.util.concatenate(tuple(obj.geometry.values()))
    else:
        tm = obj
    if not isinstance(tm, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}")
    tm.remove_unreferenced_vertices()
    return _from_trimesh(tm, units="mm")


def save_stl(mesh: Mesh3D, path: str) -> None:
    """
    Save Mesh3D as STL (binary).
    """
    tm = _to_trimesh(mesh)
    tm.export(path)


def clean_mesh(mesh: Mesh3D, repair: bool = True) -> Mesh3D:
    """
    Basic hygiene using trimesh:
      - remove duplicate/degenerate faces
      - merge/weld vertices
      - fix normals and winding
      - optional hole fill to improve watertightness
    Returns a new Mesh3D.
    """
    tm = _to_trimesh(mesh)

    # Remove obvious defects
    try:
        tm.remove_duplicate_faces()
    except Exception:
        pass
    try:
        tm.remove_degenerate_faces()
    except Exception:
        pass
    tm.remove_unreferenced_vertices()
    try:
        tm.merge_vertices()  # weld close vertices
    except Exception:
        pass

    # Normals & orientation
    try:
        import trimesh.repair as repair_mod
        repair_mod.fix_normals(tm, multibody=True)
        repair_mod.fix_winding(tm)
        if repair and not tm.is_watertight:
            repair_mod.fill_holes(tm)
            tm.remove_unreferenced_vertices()
            tm.merge_vertices()
            repair_mod.fix_normals(tm, multibody=True)
    except Exception:
        # If repair utilities not available, proceed best-effort
        pass

    return _from_trimesh(tm, units=mesh.units)


@dataclass
class CapInfo:
    """
    Descriptor for a detected planar end-cap.

    Attributes:
        face_indices: indices of cap faces in the original mesh
        plane_point:  one point on the fitted plane (centroid of cap vertices)
        plane_normal: unit normal of the plane, oriented so that:
                      - for the min-end cap: dot(n, axis) < 0
                      - for the max-end cap: dot(n, axis) > 0
        area:         total cap area (sum of triangle areas)
        center:       area-weighted centroid of the cap triangles
        side:         "min" for inlet end, "max" for outlet end (by s = (x-c)·axis)
    """
    face_indices: np.ndarray
    plane_point: np.ndarray
    plane_normal: np.ndarray
    area: float
    center: np.ndarray
    side: str  # "min" or "max"


def detect_caps(mesh: Mesh3D, end_fraction: float = 0.08, normal_dot_thresh: float = 0.85) -> List[CapInfo]:
    """
    Detect two nearly planar caps at the extremes of the vessel along its principal axis.

    Strategy:
      1) Compute principal axis (PCA).
      2) Select candidate faces whose vertex projections lie within end windows
         near the min and max along the axis.
      3) Filter faces whose normals align with +/- axis beyond a threshold.
      4) Fit a plane to each cap's vertices; compute area-weighted center.

    Returns:
        [CapInfo(min_end), CapInfo(max_end)]

    Raises:
        ValueError if not exactly two caps are detected.
    """
    tm = _to_trimesh(mesh)
    v = tm.vertices
    f = tm.faces
    face_normals = tm.face_normals

    c, axis = _principal_axis(v)
    s = (v - c) @ axis
    smin, smax = float(s.min()), float(s.max())
    length = max(smax - smin, 1e-9)
    w = end_fraction * length

    # Precompute per-face s (centroid along axis)
    s_face = ((v[f].mean(axis=1) - c) @ axis)

    # Candidate faces by axial window
    in_low = s_face <= (smin + w)
    in_high = s_face >= (smax - w)

    # Normal alignment filter
    dot_axis = face_normals @ axis
    low_mask = in_low & (np.abs(dot_axis) >= normal_dot_thresh)
    high_mask = in_high & (np.abs(dot_axis) >= normal_dot_thresh)

    # Fallback if too strict
    if not low_mask.any():
        low_mask = in_low
    if not high_mask.any():
        high_mask = in_high

    caps: List[CapInfo] = []
    for side, mask in (("min", low_mask), ("max", high_mask)):
        fi = np.nonzero(mask)[0]
        if fi.size == 0:
            continue

        # Unique vertex set
        vids = np.unique(f[fi].ravel())
        pts = v[vids]
        if pts.shape[0] < 3:
            continue

        # Fit plane by PCA of points
        pc = pts.mean(axis=0)
        U, S, Vt = np.linalg.svd(pts - pc, full_matrices=False)
        n = Vt[-1]  # normal is the smallest variance direction
        n = n / (np.linalg.norm(n) + 1e-12)

        # Orient the plane normal consistently wrt axis
        # For the "min" end we want dot(n, axis) < 0; for "max" end dot(n, axis) > 0
        d = float(n @ axis)
        if side == "min" and d > 0:
            n = -n
        if side == "max" and d < 0:
            n = -n

        # Area-weighted centroid of the selected triangles
        areas = tm.area_faces[fi]
        centers = tm.triangles_center[fi]
        A = float(areas.sum())
        center = (centers * areas[:, None]).sum(axis=0) / max(A, 1e-12)

        caps.append(CapInfo(face_indices=fi, plane_point=pc, plane_normal=n, area=A, center=center, side=side))

    # Must return exactly two, ordered by axial position of centers
    if len(caps) != 2:
        raise ValueError(f"Failed to detect two caps (found {len(caps)}).")

    caps.sort(key=lambda cap: (cap.center - c) @ axis)  # increasing s: min first
    return caps


def cap_centers(caps: List[CapInfo]) -> np.ndarray:
    """
    Return (2,3) array of cap centers in axial order: [min_end, max_end].
    """
    if len(caps) != 2:
        raise ValueError("cap_centers expects exactly two caps.")
    centers = np.vstack((caps[0].center, caps[1].center)).astype(np.float64)
    return centers


def inward_seed_points(mesh: Mesh3D, caps: List[CapInfo], offset_mm: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce two seed points **inside** the vessel by stepping *against* each cap's
    plane normal (normals are outward per detect_caps). We then verify the points are
    inside the *closed* mesh volume using trimesh.contains and automatically nudge
    deeper inside if needed.
    """
    tm = _to_trimesh(mesh)
    if not tm.is_watertight:
        # Try a light repair; seeds still work if caps are present
        try:
            import trimesh.repair as repair_mod

            repair_mod.fill_holes(tm)
            tm.remove_unreferenced_vertices()
        except Exception:
            pass

    def _nudge(center: np.ndarray, normal: np.ndarray, base_offset: float) -> np.ndarray:
        # progressively try larger offsets until point is inside or we give up
        for mul in [1.0, 2.0, 4.0, 8.0]:
            p = center - normal * (base_offset * mul)
            try:
                inside = bool(tm.contains([p])[0])
            except Exception:
                # contains() may fail if mesh is still not strictly watertight.
                # Fall back to returning the point anyway.
                inside = True
            if inside:
                return p
        # Last resort: move to center minus 10% of bbox diag along normal
        diag = float(np.linalg.norm(tm.bounds[1] - tm.bounds[0]))
        return center - normal * (0.1 * diag)

    # Caps are ordered [min_end, max_end]; normals are outward → step inward with '-n'
    n0 = np.asarray(caps[0].plane_normal, dtype=np.float64)
    n1 = np.asarray(caps[1].plane_normal, dtype=np.float64)
    c0 = np.asarray(caps[0].center, dtype=np.float64)
    c1 = np.asarray(caps[1].center, dtype=np.float64)
    p0 = _nudge(c0, n0, float(offset_mm))
    p1 = _nudge(c1, n1, float(offset_mm))
    return p0.astype(np.float64), p1.astype(np.float64)


def cap_surface_seed_points(mesh: Mesh3D, caps: List[CapInfo]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return two *surface* seed points suitable for VMTK's 'pointlist' selector,
    by taking the nearest mesh vertex to each cap center **restricted to the cap faces**.
    """
    tm = _to_trimesh(mesh)
    v = tm.vertices
    f = tm.faces
    seeds = []
    for cap in caps:
        vids = np.unique(f[np.asarray(cap.face_indices, dtype=np.int64)].ravel())
        Pv = v[vids]
        c = np.asarray(cap.center, dtype=np.float64)
        j = int(np.argmin(np.sum((Pv - c[None, :]) ** 2, axis=1)))
        seeds.append(Pv[j].astype(np.float64))
    return seeds[0], seeds[1]
