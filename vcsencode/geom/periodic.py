"""Periodic basis helpers for angular (theta) B-splines."""
from __future__ import annotations

from math import tau as TWO_PI
from typing import Tuple

import numpy as np

try:  # SciPy is optional during import but required for evaluation
    from scipy.interpolate import BSpline
except Exception as exc:  # pragma: no cover - fail lazily at runtime
    BSpline = None
    _IMPORT_ERR = exc
else:  # pragma: no cover
    _IMPORT_ERR = None


def periodic_theta_knots(
    R: int,
    start: float = 0.0,
    end: float = TWO_PI,
    degree: int = 3,
) -> np.ndarray:
    """Return uniform periodic knot vector for ``R`` independent control points.

    The resulting knot vector has length ``R + 2*degree + 1`` and supports a
    periodic spline on ``[start, end)`` with step ``h = (end-start)/R``.
    """
    if degree != 3:
        raise ValueError("Only cubic (degree=3) periodic theta splines are supported.")
    if R < degree + 1:
        raise ValueError("R must satisfy R >= degree + 1 for periodic splines.")

    start = float(start)
    end = float(end)
    h = (end - start) / float(R)
    idx = np.arange(-degree, R + degree + 1, dtype=float)
    return start + h * idx


def periodic_theta_basis_matrix(
    thetas: np.ndarray,
    R: int,
    start: float = 0.0,
    end: float = TWO_PI,
    degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct periodic cubic basis matrix ``Bθ`` (Nθ × R).

    Returns the dense basis matrix evaluated at ``thetas`` and the corresponding
    periodic knot vector. Evaluation wraps angles into ``[start, end)``.
    """
    if _IMPORT_ERR is not None:  # pragma: no cover - deferred failure
        raise RuntimeError(f"SciPy required for periodic θ splines: {_IMPORT_ERR!r}")

    thetas = np.asarray(thetas, dtype=float)
    knots = periodic_theta_knots(R, start=start, end=end, degree=degree)
    k = degree
    h = end - start
    theta_wrapped = (thetas - start) % h + start

    basis = np.empty((theta_wrapped.size, R), dtype=float)
    for j in range(R):
        coeff = np.zeros(R, dtype=float)
        coeff[j] = 1.0
        coeff_ext = np.concatenate([coeff[-k:], coeff], axis=0)
        spline = BSpline(knots, coeff_ext, k, extrapolate=False)
        basis[:, j] = spline(theta_wrapped)
    return basis, knots
