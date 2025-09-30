from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from ..models import CenterlineBSpline, RadiusSurfaceBSpline, VCSModel


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_npz(model: VCSModel, path: str | Path) -> None:
    """Persist a VCSModel to an NPZ file for later decoding/visualization."""
    path = Path(path)
    if path.parent and path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    cl = model.centerline
    rs = model.radius
    meta_serializable = _to_serializable(dict(getattr(model, "meta", {})))
    meta_json = json.dumps(meta_serializable, sort_keys=True)

    payload: Dict[str, Any] = {
        "cl_degree": np.asarray(cl.degree, dtype=np.int32),
        "cl_knots": np.asarray(cl.knots, dtype=np.float64),
        "cl_cx": np.asarray(cl.coeffs[:, 0], dtype=np.float64),
        "cl_cy": np.asarray(cl.coeffs[:, 1], dtype=np.float64),
        "cl_cz": np.asarray(cl.coeffs[:, 2], dtype=np.float64),
        "rs_degree_tau": np.asarray(rs.k_tau, dtype=np.int32),
        "rs_degree_theta": np.asarray(rs.k_theta, dtype=np.int32),
        "rs_knots_tau": np.asarray(rs.knots_tau, dtype=np.float64),
        "rs_knots_theta": np.asarray(rs.knots_theta, dtype=np.float64),
        "rs_coeffs": np.asarray(rs.coeffs, dtype=np.float64),
        "theta_periodic": np.asarray(bool(getattr(rs, "theta_periodic", True))),
        "meta_json": np.asarray(meta_json),
    }

    np.savez(path, **payload)


def load_npz(resource: str | Path) -> VCSModel:
    """Load a VCSModel from a NPZ file produced by :func:`save_npz`."""
    path = Path(resource)
    with np.load(path, allow_pickle=False) as data:
        cl_degree = int(np.asarray(data["cl_degree"]).reshape(-1)[0])
        cl_knots = np.asarray(data["cl_knots"], dtype=np.float64)
        coeff_x = np.asarray(data["cl_cx"], dtype=np.float64)
        coeff_y = np.asarray(data["cl_cy"], dtype=np.float64)
        coeff_z = np.asarray(data["cl_cz"], dtype=np.float64)

        rs_degree_tau = int(np.asarray(data["rs_degree_tau"]).reshape(-1)[0])
        rs_degree_theta = int(np.asarray(data["rs_degree_theta"]).reshape(-1)[0])
        rs_knots_tau = np.asarray(data["rs_knots_tau"], dtype=np.float64)
        rs_knots_theta = np.asarray(data["rs_knots_theta"], dtype=np.float64)
        rs_coeffs = np.asarray(data["rs_coeffs"], dtype=np.float64)
        theta_periodic = bool(np.asarray(data.get("theta_periodic", True)).reshape(-1)[0])

        meta_raw = data.get("meta_json")
        if meta_raw is None:
            meta: Dict[str, Any] = {}
        else:
            meta = json.loads(str(np.asarray(meta_raw).item()))

    cl_coeffs = np.column_stack([coeff_x, coeff_y, coeff_z])
    centerline = CenterlineBSpline(degree=cl_degree, knots=cl_knots, coeffs=cl_coeffs)
    radius = RadiusSurfaceBSpline(
        k_tau=rs_degree_tau,
        k_theta=rs_degree_theta,
        knots_tau=rs_knots_tau,
        knots_theta=rs_knots_theta,
        coeffs=rs_coeffs,
        theta_periodic=theta_periodic,
    )

    return VCSModel(centerline=centerline, radius=radius, meta=meta)


__all__ = ["save_npz", "load_npz"]
