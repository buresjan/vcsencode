"""
Vectorized ray casting from the centerline toward the vessel wall.

Given:
  - centerline c(tau) and RMF {t, v1, v2}
  - a mesh surface (triangles)

For a fixed tau and an array of thetas, we cast rays:
   origin  = c(tau) + eps * d(theta)              # small offset to avoid self-hit
   direction d(theta) = v1(tau)*cos(theta) + v2(tau)*sin(theta)

We return the first positive intersection distance along each ray. If no hit is found
for a given theta, we return np.nan at that position.

Acceleration:
- Uses trimesh.ray interfaces; if pyembree is available, trimesh auto-selects it.
- Falls back to pure Python triangle intersector otherwise.

Notes:
- At tau very near segment ends, caps may be present. Since directions are in the
  normal plane (orthogonal to t), rays are parallel to cap planes and should not hit caps.
- Star-convexity: in typical tubes, each ray has a single positive hit. We don't enforce
  this strictly here; NaNs mark missing hits.
"""
from __future__ import annotations
import numpy as np
import trimesh

from ..models import Mesh3D, CenterlineBSpline, RMF


def _to_trimesh(mesh: Mesh3D) -> trimesh.Trimesh:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    tm.remove_unreferenced_vertices()
    return tm


def _bbox_scale(tm: trimesh.Trimesh) -> float:
    b = tm.bounds  # (2,3)
    diag = float(np.linalg.norm(b[1] - b[0]))
    if not np.isfinite(diag) or diag <= 0.0:
        return 1.0
    return diag


def cast_radius(
    mesh: Mesh3D,
    centerline: CenterlineBSpline,
    frame: RMF,
    tau: float,
    theta_array: np.ndarray,
    origin_offset: float | None = None,
) -> np.ndarray:
    """
    Cast rays from c(tau) in directions d(theta) = v1*cos(theta) + v2*sin(theta).

    Args:
        mesh: triangle mesh (mm)
        centerline: CenterlineBSpline with eval() available
        frame: RMF with v1(), v2(), t() callables
        tau: scalar in [0,1]
        theta_array: 1D array of angles in radians

    Returns:
        distances: 1D array same length as theta_array.
                   For each theta, the first positive intersection distance (mm) or np.nan.
    """
    tm = _to_trimesh(mesh)
    tau = float(np.clip(tau, 0.0, 1.0))
    thetas = np.asarray(theta_array, dtype=np.float64).ravel()
    n = thetas.size

    # Origin and directions
    c = centerline.eval(tau).reshape(1, 3)
    v1 = frame.v1(tau).reshape(1, 3)
    v2 = frame.v2(tau).reshape(1, 3)

    ct = np.cos(thetas)[:, None]
    st = np.sin(thetas)[:, None]
    D = (v1 * ct) + (v2 * st)                  # (n,3)
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)

    O = np.repeat(c, n, axis=0)                # (n,3)

    # Small offset to avoid self-hits (origin lying exactly on surface/vertex)
    eps = (1e-6 * _bbox_scale(tm)) if origin_offset is None else float(origin_offset)
    O = O + eps * D

    # Compute explicit intersection locations and take nearest positive distance
    try:
        loc, idx_ray, _ = tm.ray.intersects_location(O, D)
        distances = np.full(n, np.nan, dtype=float)
        if len(idx_ray) > 0:
            di = np.linalg.norm(loc - O[idx_ray], axis=1)
            # keep smallest positive distance per ray using np.minimum.at
            per_ray_min = np.full(n, np.inf, dtype=float)
            mpos = di > eps
            if np.any(mpos):
                np.minimum.at(per_ray_min, idx_ray[mpos], di[mpos])
            distances = per_ray_min
            distances[~np.isfinite(distances)] = np.nan
            distances[distances == np.inf] = np.nan
    except Exception:
        distances = np.full(n, np.nan, dtype=float)

    # Convert distance from origin to distance from centerline: rho = eps + d
    distances = np.asarray(distances, dtype=float)
    rho = distances + eps
    # Remove tiny/negative (numerical) values
    rho[~np.isfinite(rho)] = np.nan
    rho[rho <= 0.0] = np.nan
    return rho
