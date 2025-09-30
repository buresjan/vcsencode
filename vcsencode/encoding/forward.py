"""
Forward VCS encoding:
- Sample wall radii by ray casting from the centerline normal planes
- Fit a bicubic LSQ bivariate spline ρ_w(τ,θ) with θ-seam handling
- Build a VCSModel (centerline + radius surface + metadata)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import warnings
import numpy as np
from scipy.interpolate import LSQBivariateSpline

from ..models import Mesh3D, CenterlineBSpline, RMF, RadiusSurfaceBSpline, VCSModel
from ..geom.rays import cast_radius
from ..geom.frames import compute_rmf
from ..io import clean_mesh, detect_caps, cap_centers, inward_seed_points, cap_surface_seed_points
from ..geom.padding import pad_mesh_ends
from ..geom.projection import closest_point_tau
from .crop import crop_and_refit
from ..centerline import compute_centerline_vmtk, extend_to_cap_centers, fit_centerline_bspline


def _fill_nan_circular(y: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a 1D circular sequence using linear interpolation with wrap-around.
    """
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n == 0:
        return y
    if np.all(~np.isfinite(y)):
        raise ValueError("All NaNs in a θ-scan; cannot fit radius surface.")

    x = np.arange(n)
    good = np.isfinite(y)
    if good.sum() == n:
        return y

    # For circular wrap, extend by one period on each side using the first/last valid samples
    xg = x[good]
    yg = y[good]
    # prepend/append sentinel points to allow interpolation across the wrap
    x_ext = np.r_[xg[0] - n, xg, xg[-1] + n]
    y_ext = np.r_[yg[-1], yg, yg[0]]

    y_interp = np.interp(x, x_ext, y_ext)
    y[~good] = y_interp[~good]
    return y


def fit_radius_surface(
    mesh: Mesh3D,
    centerline: CenterlineBSpline,
    frame: RMF,
    K: int = 19,
    R: int = 15,
    thetas: Optional[int] = None,
    tau_samples: Optional[int] = None,
    *,
    theta_anchor: str = "rho_argmax",
) -> Tuple[RadiusSurfaceBSpline, float, float]:
    """
    Fit bicubic ρ_w(τ,θ) with θ seam handling using LSQBivariateSpline.

    Parameters
    ----------
    mesh : Mesh3D
    centerline : CenterlineBSpline
    frame : RMF
    K, R : int
        Target numbers of basis functions (longitudinal, angular).
        (Interior knot counts = K-(kx+1), R-(ky+1) with kx=ky=3)
    thetas : int | None
        Number of θ samples for ray casting (default = R*8, >= R*4).
    tau_samples : int | None
        Number of τ samples for ray casting (default = max(5*K, 80)).

    Returns
    -------
    (RadiusSurfaceBSpline, tau_margin, theta_offset)
        theta_offset is the absolute rotation (radians) applied so that θ=0
        corresponds to the selected anchor direction.
    """
    kx = 3
    ky = 3
    if thetas is None:
        thetas = max(R * 8, 120)
    if tau_samples is None:
        tau_samples = max(5 * K, 80)

    # Sampling grids (avoid exact endpoints to reduce cap-related pathologies)
    tau_margin = 1.0 / (tau_samples * 10.0)
    taus = np.linspace(tau_margin, 1.0 - tau_margin, tau_samples, dtype=float)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, thetas, endpoint=False, dtype=float)

    # Ray-cast distances ρ(τ_i, θ_j) with robustness:
    # - try increasing origin offsets if a θ-scan returns all-NaN
    # - later, impute any remaining all-NaN rows from neighbors
    Z = np.empty((tau_samples, thetas), dtype=float)
    Z[:] = np.nan
    # Offsets to try (in mm): small -> larger
    bbox_diag = float(np.linalg.norm(np.array(mesh.vertices).ptp(axis=0)))
    trial_offsets = [1e-6 * bbox_diag, 1e-4 * bbox_diag, 1e-3 * bbox_diag, 1e-2 * bbox_diag]
    for i, tau in enumerate(taus):
        d = None
        for off in trial_offsets:
            d_try = cast_radius(mesh, centerline, frame, float(tau), theta_grid, origin_offset=off)
            if np.any(np.isfinite(d_try)):
                d = d_try
                break
        if d is None or not np.any(np.isfinite(d)):
            # keep NaNs for now; filled below by imputation
            continue
        Z[i, :] = _fill_nan_circular(d)

    # Impute any all-NaN τ-rows by copying or averaging nearest valid rows
    row_valid = np.isfinite(Z).any(axis=1)
    if not row_valid.any():
        raise ValueError("Radius sampling failed at all τ positions; cannot fit radius surface.")
    for i in range(tau_samples):
        if row_valid[i]:
            continue
        # find nearest previous and next valid rows
        prev_idx = None
        for j in range(i - 1, -1, -1):
            if row_valid[j]:
                prev_idx = j
                break
        next_idx = None
        for j in range(i + 1, tau_samples):
            if row_valid[j]:
                next_idx = j
                break
        if prev_idx is not None and next_idx is not None:
            # linear interpolate between neighbors proportionally to distance in τ
            t0, t1 = taus[prev_idx], taus[next_idx]
            w = 0.0 if abs(t1 - t0) < 1e-12 else (taus[i] - t0) / (t1 - t0)
            Z[i, :] = (1 - w) * Z[prev_idx, :] + w * Z[next_idx, :]
        elif prev_idx is not None:
            Z[i, :] = Z[prev_idx, :]
        elif next_idx is not None:
            Z[i, :] = Z[next_idx, :]
        else:
            # Should not happen because we checked row_valid.any()
            pass

    # ---- Robust clipping of outliers ---------------------------------------
    # Guard against occasional far intersections that produce giant spikes.
    zv = Z[np.isfinite(Z)]
    if zv.size == 0:
        raise ValueError("No finite samples after imputation; cannot fit radius surface.")
    med = float(np.median(zv))
    p995 = float(np.percentile(zv, 99.5))
    hi = max(3.0 * med, p995)        # generous upper guard
    Z = np.clip(Z, 0.0, hi)

    # Deterministic θ anchoring (optional)
    theta_offset = 0.0
    anchor_mode = (theta_anchor or "rho_argmax").strip().lower()
    if anchor_mode not in {"rho_argmax", "none"}:
        anchor_mode = "rho_argmax"
    if anchor_mode == "rho_argmax":
        # Use the first sampled τ row (closest to inlet margin)
        row_idx = 0
        row = Z[row_idx]
        if np.all(~np.isfinite(row)):
            # fallback to first finite row if initial row is invalid post-clipping
            finite_rows = np.where(np.isfinite(Z).any(axis=1))[0]
            if finite_rows.size > 0:
                row_idx = int(finite_rows[0])
                row = Z[row_idx]
        if np.any(np.isfinite(row)):
            j_star = int(np.nanargmax(row))
            theta_offset = float(theta_grid[j_star])
            if j_star != 0:
                Z = np.roll(Z, -j_star, axis=1)
                theta_grid = np.roll(theta_grid, -j_star)
            theta_grid = (theta_grid - theta_offset) % (2.0 * np.pi)

    # Ensure theta_grid remains sorted ascending after possible wrap
    order = np.argsort(theta_grid)
    if not np.all(order == np.arange(theta_grid.size)):
        theta_grid = theta_grid[order]
        Z = Z[:, order]

    # Quick diagnostics
    zv = Z[np.isfinite(Z)]
    print(f"[fit_radius_surface] rho stats mm: min={zv.min():.3f}  median={np.median(zv):.3f}  max={zv.max():.3f}  (after clip)")

    # Create **virtual boundary rows** to anchor τ=0 and τ=1 using the nearest sampled rows
    # This greatly stabilizes the ends without sampling through the caps.
    Z0 = Z[0:1, :]
    Z1 = Z[-1:, :]
    taus_aug = np.r_[0.0, taus, 1.0]
    Z_aug = np.vstack([Z0, Z, Z1])

    # Duplicate the θ seam for periodicity (θ=0 and θ=2π)
    theta_dup = np.r_[theta_grid, 0.0, 2.0 * np.pi]
    Z_dup = np.concatenate([Z_aug, Z_aug[:, :1], Z_aug[:, :1]], axis=1)

    # Flatten to scattered points for LSQ fit
    TT = np.repeat(taus_aug, theta_dup.size)
    TH = np.tile(theta_dup, taus_aug.size)
    ZZ = Z_dup.reshape(-1)

    # Remove any residual NaNs just in case
    m = np.isfinite(ZZ)
    TT, TH, ZZ = TT[m], TH[m], ZZ[m]

    # Interior knots (strictly inside)
    n_int_tau = max(K - (kx + 1), 0)
    tx_int = np.linspace(0.0, 1.0, n_int_tau + 2, dtype=float)[1:-1] if n_int_tau > 0 else np.array([], float)

    n_int_th = max(R - (ky + 1), 0)
    ty_int = np.linspace(0.0, 2.0 * np.pi, n_int_th + 2, dtype=float)[1:-1] if n_int_th > 0 else np.array([], float)

    # Fit LSQ bicubic with domain bbox fixed to [0,1]×[0,2π]
    spl = LSQBivariateSpline(TT, TH, ZZ, tx_int, ty_int, kx=kx, ky=ky, bbox=[0.0, 1.0, 0.0, 2.0 * np.pi])

    # Extract full knot vectors and coefficients
    tx_full, ty_full = spl.get_knots()          # full open-clamped knots incl. boundaries
    Nx = len(tx_full) - kx - 1                  # number of basis functions along τ
    Ny = len(ty_full) - ky - 1                  # number of basis functions along θ
    coeffs_flat = spl.get_coeffs()
    coeffs = coeffs_flat.reshape((Nx, Ny), order="F")  # FITPACK uses Fortran order

    rs = RadiusSurfaceBSpline(
        k_tau=kx,
        k_theta=ky,
        knots_tau=tx_full.astype(float),
        knots_theta=ty_full.astype(float),
        coeffs=coeffs.astype(float),
        theta_periodic=True,
    )
    # Attach the SciPy object for efficient evaluation
    setattr(rs, "_spl", spl)
    return rs, tau_margin, float(theta_offset)


def build_model(mesh: Mesh3D, params: Optional[Dict[str, Any]] = None) -> VCSModel:
    """
    Orchestrate STL -> VCSModel:
      clean -> caps -> seeds -> VMTK centerline -> prolong -> cubic BSpline (τ arc-length)
      -> RMF -> radius surface fit -> assemble model and meta.

    params:
      L (int): centerline control points (default 9)
      K (int): longitudinal basis count for ρ (default 19)
      R (int): angular basis count for ρ (default 15)
      resampling (float|None): VMTK resampling step (mm), default 0.5
      seed_offset_mm (float): how far inside the caps to place seeds, default 2.0
      rays_thetas (int|None): θ samples for ray casting (default R*8)
      rays_tau_samples (int|None): τ samples for ray casting (default max(5*K, 80))
    """
    if params is None:
        params = {}
    L = int(params.get("L", 9))
    K = int(params.get("K", 19))
    R = int(params.get("R", 15))
    resampling = params.get("resampling", 0.5)
    seed_offset_raw = params.get("seed_offset_mm", 2.0)
    try:
        seed_offset = float(seed_offset_raw)
    except (TypeError, ValueError):
        seed_offset = 2.0
    if "seed_offset_mm" in params and seed_offset_raw is not None and seed_offset != 2.0:
        warnings.warn(
            "seed_offset_mm is deprecated and ignored. VMTK 'pointlist' expects ON-surface seeds; "
            "we use cap-surface seeds and prolong to cap centers. This parameter no longer impacts centerline.",
            DeprecationWarning,
            stacklevel=2,
        )
    rays_thetas = params.get("rays_thetas", None)
    rays_tau_samples = params.get("rays_tau_samples", None)
    unit_scale = float(params.get("unit_scale", 1.0))
    pad_len = params.get("pad_ends_mm", None)
    theta_anchor = str(params.get("theta_anchor", "rho_argmax")).strip().lower()

    # Optional unit scaling (e.g., cm->mm => 10.0)
    if unit_scale != 1.0:
        V = np.asarray(mesh.vertices, dtype=float) * unit_scale
        F = np.asarray(mesh.faces, dtype=np.int32)
        mesh = Mesh3D(vertices=V, faces=F, units="mm")

    # Hygiene + caps + seeds
    mesh_c = clean_mesh(mesh, repair=True)
    caps = detect_caps(mesh_c)
    centers = cap_centers(caps)

    mesh_for_encoding = mesh_c
    orig_caps = caps
    orig_centers = centers
    if pad_len is None or float(pad_len) > 0.0:
        mesh_for_encoding = pad_mesh_ends(mesh_c, caps, None if pad_len is None else float(pad_len))
        caps = detect_caps(mesh_for_encoding)
        centers = cap_centers(caps)
    s0, s1 = cap_surface_seed_points(mesh_for_encoding, caps)

    # Centerline via VMTK -> prolong -> cubic BSpline (τ∈[0,1] constant speed)
    pl = compute_centerline_vmtk(mesh_for_encoding, s0, s1, resampling=resampling)
    pl = extend_to_cap_centers(pl, centers)
    cl = fit_centerline_bspline(pl, L=L, constant_speed=True)

    # RMF
    rmf = compute_rmf(cl, step_mm=1.0)

    # ρ_w(τ,θ) fit
    anchor_mode = "none" if theta_anchor in {"none", "off", "false"} else "rho_argmax"
    rs, tau_margin, theta_offset = fit_radius_surface(
        mesh_for_encoding,
        cl,
        rmf,
        K=K,
        R=R,
        thetas=rays_thetas,
        tau_samples=rays_tau_samples,
        theta_anchor=anchor_mode,
    )

    # Metadata for reproducibility and reversibility
    meta = {
        "units": mesh.units,
        "unit_scale": unit_scale,
        "k_tau": 3,
        "k_theta": 3,
        "L": L,
        "K": K,
        "R": R,
        "knots_tau": rs.knots_tau.tolist(),
        "knots_theta": rs.knots_theta.tolist(),
        "theta_periodic": True,
        "tau_param": "arc-length",
        "tau_margin": float(tau_margin),
        "pad_ends_mm": None if (pad_len is None or float(pad_len) == 0.0) else float(pad_len),
        "theta_anchor": anchor_mode,
        "theta_offset": float(theta_offset),
        "seed_offset_mm_deprecated": seed_offset,
        "cap_planes": [
            {
                "center": np.asarray(orig_caps[0].center, dtype=float).tolist(),
                "normal": np.asarray(orig_caps[0].plane_normal, dtype=float).tolist(),
            },
            {
                "center": np.asarray(orig_caps[1].center, dtype=float).tolist(),
                "normal": np.asarray(orig_caps[1].plane_normal, dtype=float).tolist(),
            },
        ],
        "cap_centers": orig_centers.tolist(),
        "software_versions": {},  # can be filled later
    }

    model_padded = VCSModel(centerline=cl, radius=rs, meta=meta)

    if mesh_for_encoding is not mesh_c:
        t0 = float(closest_point_tau(cl, np.asarray(orig_centers[0], dtype=float)))
        t1 = float(closest_point_tau(cl, np.asarray(orig_centers[1], dtype=float)))
        t0, t1 = (t0, t1) if t0 < t1 else (t1, t0)
        model_cropped = crop_and_refit(
            model_padded, t0, t1, L=L, K=K, R=R,
            tau_samples=rays_tau_samples or max(5 * K, 80),
            theta_samples=rays_thetas or max(8 * R, 120),
        )
        return model_cropped
    else:
        return model_padded
