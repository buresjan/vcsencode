"""
Rotation-minimizing (parallel-transport) frame along a 3D centerline.

We implement the **double reflection method** (Wang et al., 2008), which updates the
frame between two consecutive samples (x_i, t_i) -> (x_{i+1}, t_{i+1}) via two reflections:

Table I (Algorithm—Double Reflection):
  1) v1 := x_{i+1} - x_i
  2) c1 := v1·v1
  3) r'_i := r_i - 2 (v1·r_i)/c1 * v1        # R1 applied to r_i
  4) t'_i := t_i - 2 (v1·t_i)/c1 * v1        # R1 applied to t_i
  5) v2 := t_{i+1} - t'_i
  6) c2 := v2·v2
  7) r_{i+1} := r'_i - 2 (v2·r'_i)/c2 * v2   # R2 applied to r'_i
  8) s_{i+1} := t_{i+1} × r_{i+1}
  9) U_{i+1} := (r_{i+1}, s_{i+1}, t_{i+1})

We discretize the spline by uniform *arc-length* samples in τ∈[0,1] using ~1 mm step.
Returned callables (t, v1, v2) interpolate precomputed frames piecewise linearly with
re-orthonormalization using the exact centerline tangent at the query τ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from ..models import RMF, CenterlineBSpline


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _choose_initial_normal(t0: np.ndarray, rule: str = "centroid-plane") -> np.ndarray:
    """
    Deterministic initial v1(0) orthogonal to t0.
    'centroid-plane' rule: pick the world axis most orthogonal to t0 and project it.
    This ensures v1(0) ⟂ t0 and defines the plane {t0, v1(0)}.
    """
    t0 = _normalize(np.asarray(t0, dtype=np.float64)).reshape(3)
    axes = np.eye(3, dtype=np.float64)
    k = int(np.argmin(np.abs(axes @ t0)))  # most orthogonal axis
    a = axes[k]
    v1 = a - (a @ t0) * t0
    v1 = _normalize(v1)
    return v1.reshape(3)


def _interp_frame(centerline: CenterlineBSpline, taus: np.ndarray, R: np.ndarray, S: np.ndarray):
    """
    Build interpolation functions for v1(τ) and v2(τ) from discrete samples.
    We linearly interpolate v1 and then Gram–Schmidt with the exact tangent t(τ).
    """
    taus = np.asarray(taus, dtype=np.float64)
    n = len(taus) - 1

    def _interp_v1(tauq: np.ndarray) -> np.ndarray:
        tq = np.asarray(tauq, dtype=np.float64)
        tq = np.clip(tq, 0.0, 1.0)
        # indices of left segment
        idx = np.searchsorted(taus, tq, side="right") - 1
        idx = np.clip(idx, 0, n - 1)
        w = (tq - taus[idx]) / np.maximum(taus[idx + 1] - taus[idx], 1e-12)
        v = (1.0 - w)[:, None] * R[idx] + w[:, None] * R[idx + 1]
        # re-orthonormalize vs. true tangent
        T = centerline.tangent(tq)
        T = _normalize(T)
        V2 = np.cross(T, v)
        V2 = _normalize(V2)
        V1 = np.cross(V2, T)
        V1 = _normalize(V1)
        return V1

    def v1_func(tauq):
        v = _interp_v1(np.atleast_1d(tauq))
        return v[0] if np.isscalar(tauq) else v

    def v2_func(tauq):
        tq = np.atleast_1d(tauq)
        T = centerline.tangent(tq)
        T = _normalize(T)
        V1 = _interp_v1(tq)
        V2 = np.cross(T, V1)
        V2 = _normalize(V2)
        return V2[0] if np.isscalar(tauq) else V2

    def t_func(tauq):
        return centerline.tangent(tauq)

    return t_func, v1_func, v2_func


def compute_rmf(
    centerline: CenterlineBSpline,
    init_rule: str = "centroid-plane",
    method: str = "double_reflection",
    step_mm: float = 1.0,
    init_v1: np.ndarray | None = None,
) -> RMF:
    """
    Compute a Rotation-Minimizing Frame (RMF) along the centerline using the
    **double reflection** method (stable even for nearly-collinear segments).

    Args:
        centerline: cubic B-spline curve with τ∈[0,1] constant-speed parameterization.
        init_rule: how to pick v1(0). Default 'centroid-plane' picks the world axis most
                   orthogonal to t(0) and projects it to the normal plane.
        method: only 'double_reflection' is implemented.
        step_mm: target arc-length step for sampling.
        init_v1: optional user-specified initial v1(0); projected to normal plane.

    Returns:
        RMF with callables t(τ), v1(τ), v2(τ).
    """
    assert method == "double_reflection", "Only double_reflection RMF is implemented."
    # Sample τ by approximate arc-length step
    L = max(centerline.length(), 1e-6)
    nseg = int(np.clip(np.ceil(L / float(step_mm)), 4, 5000))
    taus = np.linspace(0.0, 1.0, nseg + 1, dtype=np.float64)
    X = centerline.eval(taus)           # (n+1,3)
    T = centerline.tangent(taus)        # (n+1,3)
    T = _normalize(T)

    # Initial frame U0 = (r0, s0, t0)
    t0 = T[0]
    if init_v1 is not None:
        r0 = np.asarray(init_v1, dtype=np.float64).reshape(3)
        r0 = r0 - (r0 @ t0) * t0
        if np.linalg.norm(r0) < 1e-12:
            r0 = _choose_initial_normal(t0, rule=init_rule)
        else:
            r0 = _normalize(r0)
    else:
        r0 = _choose_initial_normal(t0, rule=init_rule)
    s0 = np.cross(t0, r0); s0 = _normalize(s0)
    r0 = np.cross(s0, t0); r0 = _normalize(r0)

    R = np.empty_like(T); S = np.empty_like(T)
    R[0] = r0; S[0] = s0

    eps = 1e-20
    # Apply double reflection along samples
    for i in range(nseg):
        xi, xi1 = X[i], X[i + 1]
        ti, ti1 = T[i], T[i + 1]
        ri = R[i]

        v1 = xi1 - xi
        c1 = float(v1 @ v1)
        if c1 > eps:
            ri_p = ri - 2.0 * (ri @ v1) / c1 * v1
            ti_p = ti - 2.0 * (ti @ v1) / c1 * v1
        else:
            ri_p = ri
            ti_p = ti

        v2 = ti1 - ti_p
        c2 = float(v2 @ v2)
        if c2 > eps:
            ri1 = ri_p - 2.0 * (ri_p @ v2) / c2 * v2
        else:
            ri1 = ri_p

        si1 = np.cross(ti1, ri1)

        # Orthonormalize to guard against drift
        ti1 = _normalize(ti1)
        si1 = si1 - (si1 @ ti1) * ti1
        si1 = _normalize(si1)
        ri1 = np.cross(si1, ti1); ri1 = _normalize(ri1)

        R[i + 1] = ri1
        S[i + 1] = si1

    # Build callables with interpolation + re-orthonormalization
    t_func, v1_func, v2_func = _interp_frame(centerline, taus, R, S)
    return RMF(t=t_func, v1=v1_func, v2=v2_func)
