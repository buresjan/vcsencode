from __future__ import annotations
import numpy as np
from scipy.interpolate import LSQBivariateSpline

from ..models import VCSModel, RadiusSurfaceBSpline
from ..centerline import Polyline3D, fit_centerline_bspline


def crop_and_refit(model: VCSModel, tau0: float, tau1: float, *,
                   L: int | None = None, K: int | None = None, R: int | None = None,
                   tau_samples: int = 120, theta_samples: int = 180) -> VCSModel:
    """
    Crop a VCSModel to the sub-interval [tau0, tau1] (in the model's current τ)
    by *refitting* both the centerline spline and the radius surface.

    - Centerline: sample densely on [tau0,tau1], refit cubic constant-speed B-spline with L control points.
    - Radius: sample Z = rho(t,θ) on a grid in [0,1]x[0,2π), where t in [0,1] maps to τ=tau0+(tau1-tau0)*t,
              then LSQ-fit a bicubic surface with K (τ) and R (θ) basis counts.
    """
    assert 0.0 <= tau0 < tau1 <= 1.0
    cl_old = model.centerline
    rs_old = model.radius

    # Set defaults
    L = int(L if L is not None else cl_old.coeffs.shape[0])
    K = int(K if K is not None else max(13, 2 * L + 1))
    R = int(R if R is not None else 15)

    # --- Centerline refit ---
    ts = np.linspace(tau0, tau1, max(300, tau_samples * 2))
    P = cl_old.eval(ts)
    cl_new = fit_centerline_bspline(Polyline3D(points=P), L=L, constant_speed=True)

    # --- Radius refit ---
    T_sub = np.linspace(0.0, 1.0, tau_samples)
    TH = np.linspace(0.0, 2.0 * np.pi, theta_samples, endpoint=False)
    Tau = tau0 + (tau1 - tau0) * T_sub
    Z = rs_old.eval_grid(Tau, TH)  # shape (tau_samples, theta_samples)

    kx = 3
    ky = 3
    n_int_tau = max(K - (kx + 1), 0)
    tx_int = np.linspace(0.0, 1.0, n_int_tau + 2)[1:-1] if n_int_tau > 0 else np.array([], float)

    n_int_th = max(R - (ky + 1), 0)
    ty_int = np.linspace(0.0, 2.0 * np.pi, n_int_th + 2)[1:-1] if n_int_th > 0 else np.array([], float)

    # Duplicate θ seam for periodicity
    TH_dup = np.r_[TH, 0.0, 2.0 * np.pi]
    Z_dup = np.concatenate([Z, Z[:, :1], Z[:, :1]], axis=1)

    TT = np.repeat(T_sub, TH_dup.size)
    THH = np.tile(TH_dup, T_sub.size)
    ZZ = Z_dup.reshape(-1)

    spl = LSQBivariateSpline(TT, THH, ZZ, tx_int, ty_int, kx=kx, ky=ky, bbox=[0, 1, 0, 2 * np.pi])
    tx_full, ty_full = spl.get_knots()
    Nx = len(tx_full) - kx - 1
    Ny = len(ty_full) - ky - 1
    coeffs = spl.get_coeffs().reshape((Nx, Ny), order="F")

    rs_new = RadiusSurfaceBSpline(
        k_tau=kx,
        k_theta=ky,
        knots_tau=tx_full.astype(float),
        knots_theta=ty_full.astype(float),
        coeffs=coeffs.astype(float),
        theta_periodic=True,
    )
    setattr(rs_new, "_spl", spl)

    # --- Meta update (preserve planes, units, scale, etc.) ---
    meta = dict(model.meta)
    meta["crop_tau0"] = float(tau0)
    meta["crop_tau1"] = float(tau1)
    meta["knots_tau"] = tx_full.tolist()
    meta["knots_theta"] = ty_full.tolist()
    # Recompute tau_margin consistent with forward fit definition
    meta["tau_margin"] = float(1.0 / (float(tau_samples) * 10.0))

    return VCSModel(centerline=cl_new, radius=rs_new, meta=meta)
