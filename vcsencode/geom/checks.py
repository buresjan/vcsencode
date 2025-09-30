from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from ..models import CenterlineBSpline, Mesh3D
from .projection import closest_point_tau


def _bspl(centerline: CenterlineBSpline):
    k = centerline.degree
    t = centerline.knots
    Cx = BSpline(t, centerline.coeffs[:, 0], k)
    Cy = BSpline(t, centerline.coeffs[:, 1], k)
    Cz = BSpline(t, centerline.coeffs[:, 2], k)
    return (Cx, Cy, Cz), (Cx.derivative(), Cy.derivative(), Cz.derivative()), (Cx.derivative(2), Cy.derivative(2), Cz.derivative(2))


def _radius_of_curvature(centerline: CenterlineBSpline, tau: np.ndarray) -> np.ndarray:
    (Cx, Cy, Cz), (Cx1, Cy1, Cz1), (Cx2, Cy2, Cz2) = _bspl(centerline)
    t = np.atleast_1d(tau).astype(float)
    r1 = np.stack([Cx1(t), Cy1(t), Cz1(t)], axis=1)
    r2 = np.stack([Cx2(t), Cy2(t), Cz2(t)], axis=1)
    num = np.linalg.norm(np.cross(r1, r2), axis=1)
    den = np.linalg.norm(r1, axis=1) ** 3 + 1e-12
    kappa = num / den
    R = np.where(kappa > 1e-10, 1.0 / kappa, np.inf)
    return R


def check_tau_uniqueness(centerline: CenterlineBSpline, mesh: Mesh3D, max_vertices: int | None = 10000) -> dict:
    """
    Paper Section 4: τ is unique inside Ω = {x : dist(x,c) < R_curv(τ(x))}.
    Returns summary dict with violation fraction and maxima.
    """
    V = np.asarray(mesh.vertices, float)
    if max_vertices is not None and V.shape[0] > max_vertices:
        idx = np.linspace(0, V.shape[0] - 1, max_vertices, dtype=int)
        V = V[idx]
    taus = np.empty(V.shape[0], float)
    rhos = np.empty(V.shape[0], float)
    for i, p in enumerate(V):
        tau = closest_point_tau(centerline, p, tol=1e-9, maxit=60)
        c = centerline.eval(tau).reshape(3)
        taus[i] = tau
        rhos[i] = float(np.linalg.norm(p - c))
    Rcurv = _radius_of_curvature(centerline, taus)
    ratio = rhos / (Rcurv + 1e-12)
    vio = ratio > 1.0 + 1e-6
    return {
        "n": int(V.shape[0]),
        "violations": int(np.count_nonzero(vio)),
        "violations_frac": float(np.count_nonzero(vio) / float(V.shape[0])),
        "max_ratio": float(np.nanmax(ratio)),
    }
