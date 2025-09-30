"""
Closest-point projection and VCS coordinate evaluation.

Given a centerline spline c(τ), we find τ(x) by minimizing d(τ) = ||c(τ) - x||^2.
The stationary condition is:
    g(τ) = c'(τ) · (c(τ) - x) = 0
with derivative:
    g'(τ) = c''(τ) · (c(τ) - x) + c'(τ) · c'(τ).

We use a coarse scan to get an initial τ, then Newton with clamping to [0,1].
θ(x) and ρ(x) are then computed in the normal plane spanned by the RMF {v1, v2}.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline

from ..models import CenterlineBSpline, RMF


def _bsplines(centerline: CenterlineBSpline):
    """Return coordinate-wise BSplines and their derivatives for c, c', c''."""
    k = centerline.degree
    t = centerline.knots
    Cx = BSpline(t, centerline.coeffs[:, 0], k)
    Cy = BSpline(t, centerline.coeffs[:, 1], k)
    Cz = BSpline(t, centerline.coeffs[:, 2], k)
    Cx1, Cy1, Cz1 = Cx.derivative(), Cy.derivative(), Cz.derivative()
    Cx2, Cy2, Cz2 = Cx1.derivative(), Cy1.derivative(), Cz1.derivative()
    return (Cx, Cy, Cz), (Cx1, Cy1, Cz1), (Cx2, Cy2, Cz2)


def closest_point_tau(centerline: CenterlineBSpline, x: np.ndarray, tol: float = 1e-8, maxit: int = 30) -> float:
    """
    Find τ ∈ [0,1] minimizing ||c(τ) - x|| via Newton on g(τ)=c'(τ)·(c(τ)-x)=0, with bracketing init.
    Robustness:
      - coarse grid (N=400) to find a good initial τ
      - clamp iterates to [0,1]
      - fallback to coarse minimum if Newton stagnates
    """
    x = np.asarray(x, dtype=np.float64).reshape(3)
    (Cx, Cy, Cz), (Cx1, Cy1, Cz1), (Cx2, Cy2, Cz2) = _bsplines(centerline)

    # Coarse scan
    ts = np.linspace(0.0, 1.0, 401, dtype=np.float64)
    cx = np.stack([Cx(ts), Cy(ts), Cz(ts)], axis=1)
    d2 = np.sum((cx - x[None, :])**2, axis=1)
    j0 = int(np.argmin(d2))
    tau = ts[j0]

    def g_and_gp(t):
        c = np.array([Cx(t), Cy(t), Cz(t)], dtype=np.float64)
        c1 = np.array([Cx1(t), Cy1(t), Cz1(t)], dtype=np.float64)
        c2 = np.array([Cx2(t), Cy2(t), Cz2(t)], dtype=np.float64)
        v = c - x
        g = float(c1 @ v)
        gp = float(c2 @ v + c1 @ c1)
        return g, gp

    # Newton with clamped updates
    last = tau
    for _ in range(maxit):
        g, gp = g_and_gp(tau)
        if abs(gp) < 1e-14:
            break
        step = -g / gp
        tau_new = float(np.clip(tau + step, 0.0, 1.0))
        if abs(tau_new - tau) < tol:
            tau = tau_new
            break
        last, tau = tau, tau_new

    # If Newton didn't improve much, fall back to local refine around coarse min
    if abs(tau - ts[j0]) > 0.1:
        tau = ts[j0]
    return float(tau)


def theta(centerline: CenterlineBSpline, frame: RMF, x: np.ndarray, tau: float) -> float:
    """
    Angle θ ∈ [0,2π) in the plane spanned by {v1(τ), v2(τ)}, measured from v1.
    """
    x = np.asarray(x, dtype=np.float64).reshape(3)
    c = centerline.eval(float(tau)).reshape(3)
    v = x - c
    v1 = frame.v1(float(tau)).reshape(3)
    v2 = frame.v2(float(tau)).reshape(3)
    a = float(np.arctan2(v @ v2, v @ v1))
    if a < 0.0:
        a += 2.0 * np.pi
    return a


def rho(centerline: CenterlineBSpline, x: np.ndarray, tau: float) -> float:
    """
    Radial distance ρ = ||x - c(τ)||.
    """
    x = np.asarray(x, dtype=np.float64).reshape(3)
    c = centerline.eval(float(tau)).reshape(3)
    return float(np.linalg.norm(x - c))
