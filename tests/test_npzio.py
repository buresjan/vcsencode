import json
import numpy as np

from vcsencode.encoding.npzio import save_npz, load_npz
from vcsencode.models import CenterlineBSpline, RadiusSurfaceBSpline, VCSModel


def _open_knot_vector(num_basis: int, degree: int, start: float, end: float) -> np.ndarray:
    interior = max(num_basis - degree - 1, 0)
    if interior > 0:
        internal = np.linspace(start, end, interior + 2, dtype=float)[1:-1]
    else:
        internal = np.array([], dtype=float)
    return np.concatenate([
        np.full(degree + 1, start, dtype=float),
        internal,
        np.full(degree + 1, end, dtype=float),
    ])


def _make_dummy_model() -> VCSModel:
    cl_knots = _open_knot_vector(num_basis=4, degree=3, start=0.0, end=1.0)
    cl_coeffs = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.33],
            [0.0, -1.0, 0.66],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    centerline = CenterlineBSpline(degree=3, knots=cl_knots, coeffs=cl_coeffs)

    rs_knots_tau = _open_knot_vector(num_basis=5, degree=3, start=0.0, end=1.0)
    rs_knots_theta = _open_knot_vector(num_basis=6, degree=3, start=0.0, end=2.0 * np.pi)
    coeffs = np.linspace(4.0, 6.0, num=5 * 6, dtype=float).reshape(5, 6)
    radius = RadiusSurfaceBSpline(
        k_tau=3,
        k_theta=3,
        knots_tau=rs_knots_tau,
        knots_theta=rs_knots_theta,
        coeffs=coeffs,
        theta_periodic=True,
    )

    meta = {
        "units": "mm",
        "unit_scale": 1.0,
        "theta_anchor": "rho_argmax",
        "theta_offset": 0.5,
    }
    return VCSModel(centerline=centerline, radius=radius, meta=meta)


def test_npz_round_trip(tmp_path):
    model = _make_dummy_model()
    target = tmp_path / "model.npz"
    save_npz(model, target)

    loaded = load_npz(target)

    assert loaded.centerline.degree == model.centerline.degree
    np.testing.assert_allclose(loaded.centerline.knots, model.centerline.knots)
    np.testing.assert_allclose(loaded.centerline.coeffs, model.centerline.coeffs)

    assert loaded.radius.k_tau == model.radius.k_tau
    assert loaded.radius.k_theta == model.radius.k_theta
    np.testing.assert_allclose(loaded.radius.knots_tau, model.radius.knots_tau)
    np.testing.assert_allclose(loaded.radius.knots_theta, model.radius.knots_theta)
    np.testing.assert_allclose(loaded.radius.coeffs, model.radius.coeffs)

    assert loaded.radius.theta_periodic == model.radius.theta_periodic
    assert loaded.meta == model.meta
