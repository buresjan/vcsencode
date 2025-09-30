from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple
import numpy as np
from scipy.interpolate import BSpline


@dataclass
class Mesh3D:
    """
    Triangle mesh in millimeters.
    - vertices: (N,3) float64 array
    - faces:    (M,3) int32 array indexing into vertices
    """
    vertices: np.ndarray
    faces: np.ndarray
    units: str = "mm"


@dataclass
class Polyline3D:
    """
    Piecewise-linear 3D polyline.
    - points: (P,3) float64 array, ordered along the line.
    """
    points: np.ndarray


@dataclass
class CenterlineBSpline:
    """
    Cubic B-spline representation of the vessel centerline.

    Parameterization:
        tau ∈ [0,1] is *constant-speed* (proportional to arc length).
    Accessors (to be implemented later):
        eval(tau):      R^1 -> R^3, centerline point c(tau)
        tangent(tau):   unit tangent vector t(tau) = c'(tau)/||c'(tau)||
        length():       arc length of the curve
    """
    degree: int
    knots: np.ndarray        # shape (L + degree + 1,)
    coeffs: np.ndarray       # shape (L, 3) control points (x,y,z)

    def eval(self, tau: float) -> np.ndarray:
        """
        Evaluate the centerline at parameter tau ∈ [0,1].
        Accepts scalar or array input; returns matching shape with last dimension 3.
        """
        t = np.asarray(tau, dtype=np.float64)
        t = np.clip(t, 0.0, 1.0)

        spline_x = BSpline(self.knots, self.coeffs[:, 0], self.degree)
        spline_y = BSpline(self.knots, self.coeffs[:, 1], self.degree)
        spline_z = BSpline(self.knots, self.coeffs[:, 2], self.degree)
        pts = np.stack([spline_x(t), spline_y(t), spline_z(t)], axis=-1)

        if np.isscalar(tau):
            return pts.astype(np.float64)
        return pts

    def tangent(self, tau: float) -> np.ndarray:
        """
        Evaluate the unit tangent vector along the spline at tau.
        """
        t = np.asarray(tau, dtype=np.float64)
        t = np.clip(t, 0.0, 1.0)

        deriv_x = BSpline(self.knots, self.coeffs[:, 0], self.degree).derivative()
        deriv_y = BSpline(self.knots, self.coeffs[:, 1], self.degree).derivative()
        deriv_z = BSpline(self.knots, self.coeffs[:, 2], self.degree).derivative()
        vecs = np.stack([deriv_x(t), deriv_y(t), deriv_z(t)], axis=-1)
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = vecs / norms

        if np.isscalar(tau):
            return tangents.astype(np.float64)
        return tangents

    def length(self) -> float:
        """
        Approximate the spline arc length by sampling uniformly in tau.
        """
        ts = np.linspace(0.0, 1.0, 2001, dtype=np.float64)
        pts = self.eval(ts)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(np.sum(seg))


@dataclass
class RMF:
    """
    Rotation-Minimizing (parallel-transport) Frame along the centerline.

    Returns orthonormal basis at any tau:
        t(tau)  -> unit tangent
        v1(tau) -> first normal
        v2(tau) -> second normal
    Deterministic initial orientation v1(0) will be defined later.
    """
    t: Callable[[float], np.ndarray]   = field(default=lambda tau: np.zeros(3))
    v1: Callable[[float], np.ndarray]  = field(default=lambda tau: np.zeros(3))
    v2: Callable[[float], np.ndarray]  = field(default=lambda tau: np.zeros(3))


@dataclass
class RadiusSurfaceBSpline:
    """
    Cubic bivariate spline for wall distance rho_w(tau, theta).

    Domain:
        tau   ∈ [0,1]   (longitudinal)
        theta ∈ [0,2π)  (angular), periodic in theta.

    Accessors (to be implemented later):
        rho(tau, theta)        -> float
        eval_grid(T, Theta)    -> (len(T), len(Theta)) array of rho values
    """
    k_tau: int
    k_theta: int
    knots_tau: np.ndarray     # 1D knot vector
    knots_theta: np.ndarray   # 1D knot vector (periodic)
    coeffs: np.ndarray        # shape (K, R) spline coefficients
    theta_periodic: bool = True

    def _basis_values(self, u: float, degree: int, knots: np.ndarray, cache_attr: str) -> np.ndarray:
        """Evaluate all basis functions at parameter ``u`` using cached splines."""

        splines = getattr(self, cache_attr, None)
        if splines is None:
            n_basis = len(knots) - degree - 1
            splines = []
            for idx in range(n_basis):
                coeff = np.zeros(n_basis, dtype=float)
                coeff[idx] = 1.0
                splines.append(BSpline(knots, coeff, degree, extrapolate=False))
            setattr(self, cache_attr, splines)

        u = float(u)
        values = np.empty(len(splines), dtype=float)
        for idx, spline in enumerate(splines):
            values[idx] = float(spline(u))
        return values

    def rho(self, tau: float, theta: float) -> float:
        """Evaluate ρ_w(tau, theta) using either LSQ spline or basis expansion."""
        t = float(np.clip(tau, 0.0, 1.0))
        th = float(theta)
        if self.theta_periodic:
            twopi = 2.0 * np.pi
            th = (th % twopi + twopi) % twopi
        spl = getattr(self, "_spl", None)
        if spl is not None:
            return float(spl.ev(t, th))
        Bt = self._basis_values(t, self.k_tau, self.knots_tau, "_basis_tau")
        Bth = self._basis_values(th, self.k_theta, self.knots_theta, "_basis_theta")
        return float(Bt @ self.coeffs @ Bth)

    def eval_grid(self, T: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        """Evaluate ρ_w on tensor grid T × Theta."""
        T = np.asarray(T, dtype=float)
        Theta = np.asarray(Theta, dtype=float)
        if self.theta_periodic:
            twopi = 2.0 * np.pi
            Theta = (Theta % twopi + twopi) % twopi
        spl = getattr(self, "_spl", None)
        if spl is not None:
            TT, TH = np.meshgrid(T, Theta, indexing="ij")
            return spl.ev(TT.ravel(), TH.ravel()).reshape(TT.shape)
        BT = np.stack([self._basis_values(t, self.k_tau, self.knots_tau, "_basis_tau") for t in T], axis=0)
        BTH = np.stack([self._basis_values(th, self.k_theta, self.knots_theta, "_basis_theta") for th in Theta], axis=0)
        return BT @ self.coeffs @ BTH.T


@dataclass
class VCSModel:
    """
    Vessel Coordinate System model bundle.

    Contains:
        - centerline: CenterlineBSpline
        - radius:     RadiusSurfaceBSpline
        - meta:       dict with units, degrees, knot vectors, periodicity, frame init rule, versions

    Pack/Unpack:
        - pack()   -> 1D numpy array 'a' (fixed-length encoding vector)
        - unpack() -> reconstruct VCSModel from 'a' + meta
    """
    centerline: CenterlineBSpline
    radius: RadiusSurfaceBSpline
    meta: Dict

    def pack(self) -> np.ndarray:
        clc = np.asarray(self.centerline.coeffs, dtype=float).reshape(-1)
        rsc = np.asarray(self.radius.coeffs, dtype=float).reshape(-1)
        return np.concatenate([clc, rsc])

    @staticmethod
    def unpack(a: np.ndarray, meta: Dict) -> "VCSModel":
        a = np.asarray(a, dtype=float).ravel()
        k_tau = int(meta.get("k_tau", 3))
        k_theta = int(meta.get("k_theta", 3))
        knots_tau_r = np.asarray(meta["knots_tau"], dtype=float)
        knots_theta_r = np.asarray(meta["knots_theta"], dtype=float)
        Nx = len(knots_tau_r) - k_tau - 1
        Ny = len(knots_theta_r) - k_theta - 1
        n_r = Nx * Ny

        if a.size < n_r:
            raise ValueError("Encoding vector too short for radius surface coefficients.")
        n_cl_flat = a.size - n_r
        if n_cl_flat % 3 != 0:
            raise ValueError("Centerline coefficient segment must be divisible by 3.")
        L = n_cl_flat // 3

        cl_coeffs = a[: 3 * L].reshape(L, 3)
        rs_coeffs = a[3 * L :].reshape(Nx, Ny)

        cl_degree = int(meta.get("cl_degree", 3))
        cl_knots = meta.get("cl_knots")
        if cl_knots is None:
            n_int = max(L - cl_degree - 1, 0)
            if n_int > 0:
                t_int = np.linspace(0.0, 1.0, n_int + 2, dtype=float)[1:-1]
            else:
                t_int = np.array([], dtype=float)
            cl_knots_full = np.r_[np.zeros(cl_degree + 1), t_int, np.ones(cl_degree + 1)]
        else:
            cl_knots_full = np.asarray(cl_knots, dtype=float)

        centerline = CenterlineBSpline(degree=cl_degree, knots=cl_knots_full, coeffs=cl_coeffs)
        radius = RadiusSurfaceBSpline(
            k_tau=k_tau,
            k_theta=k_theta,
            knots_tau=knots_tau_r,
            knots_theta=knots_theta_r,
            coeffs=rs_coeffs,
            theta_periodic=bool(meta.get("theta_periodic", True)),
        )
        return VCSModel(centerline=centerline, radius=radius, meta=meta)
