"""
Residuals and QC metrics.

Given:
  - Mesh3D with vertices P_i
  - VCSModel (centerline spline, RMF, and radius surface)

We compute, for each vertex p:
  τ = argmin_t ||c(t) - p||
  θ = atan2( (p-c)·v2(τ), (p-c)·v1(τ) )
  ρ̂ = ρ_w(τ, θ)
  p̂ = c(τ) + ρ̂ [ v1(τ) cosθ + v2(τ) sinθ ]
  r(p) = ||p - p̂||

Returns summary stats and (optionally) arrays for further analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from ..models import Mesh3D, VCSModel
from ..geom.frames import compute_rmf
from ..geom.projection import closest_point_tau, theta as theta_from_frame


@dataclass
class ResidualResult:
    summary: Dict[str, float]
    tau: np.ndarray
    theta: np.ndarray
    rho_hat: np.ndarray
    residuals: np.ndarray


def _predict_point(model: VCSModel, tau: float, theta: float, rmf) -> np.ndarray:
    """Compute x̂(τ,θ) = c(τ) + ρ_w(τ,θ)[v1 cosθ + v2 sinθ]."""
    c = model.centerline.eval(tau)
    v1 = rmf.v1(tau)
    v2 = rmf.v2(tau)
    rho_hat = model.radius.rho(tau, theta)
    d = np.cos(theta) * v1 + np.sin(theta) * v2
    return c + rho_hat * d


def residuals(
    mesh: Mesh3D,
    model: VCSModel,
    *,
    max_vertices: Optional[int] = None,
    chunk: int = 5000,
) -> ResidualResult:
    """
    Compute residuals at mesh vertices (optionally subsample with max_vertices).

    Parameters
    ----------
    mesh : Mesh3D
    model : VCSModel
    max_vertices : int | None
        If provided and vertex count exceeds this, uniform subsampling is used.
    chunk : int
        Number of vertices to process per batch.

    Returns
    -------
    ResidualResult with summary and per-vertex arrays (tau, theta, rho_hat, residuals).
    """
    V = np.asarray(mesh.vertices, dtype=float)
    # Ensure vertices are in the same working units as the model (mm by default)
    scale = float(model.meta.get("unit_scale", 1.0))
    if scale != 1.0:
        V = V * scale
    N = V.shape[0]
    if max_vertices is not None and N > max_vertices:
        idx = np.linspace(0, N - 1, max_vertices, dtype=int)
        V = V[idx]
        N = V.shape[0]

    # RMF along the curve with ~1 mm sampling (derived from curve length)
    step_mm = float(model.meta.get("rmf_step_mm", max(model.centerline.length() / 1000.0, 1e-3)))
    v1_0 = model.meta.get("rmf_v1_0", None)
    init_v1 = None if v1_0 is None else np.asarray(v1_0, float)
    rmf = compute_rmf(model.centerline, step_mm=step_mm, init_v1=init_v1)

    tau_arr = np.empty(N, dtype=float)
    th_arr = np.empty(N, dtype=float)
    rhoh_arr = np.empty(N, dtype=float)
    res_arr = np.empty(N, dtype=float)

    # Process in chunks for memory safety
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        P = V[start:end]

        # Project each vertex
        for i, p in enumerate(P):
            tau = closest_point_tau(model.centerline, p, tol=1e-9, maxit=60)
            th = theta_from_frame(model.centerline, rmf, p, tau)
            xhat = _predict_point(model, tau, th, rmf)
            r = float(np.linalg.norm(p - xhat))
            tau_arr[start + i] = tau
            th_arr[start + i] = th
            rhoh_arr[start + i] = model.radius.rho(tau, th)
            res_arr[start + i] = r

    # Summary stats
    finite = np.isfinite(res_arr)
    vals = res_arr[finite] if finite.any() else np.array([np.nan])
    summary = {
        "count": float(vals.size),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p90": float(np.nanpercentile(vals, 90)),
        "p95": float(np.nanpercentile(vals, 95)),
        "max": float(np.nanmax(vals)),
    }

    return ResidualResult(summary=summary, tau=tau_arr, theta=th_arr, rho_hat=rhoh_arr, residuals=res_arr)
