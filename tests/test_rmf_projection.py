import numpy as np
import pytest

from vcsencode.centerline import fit_centerline_bspline, Polyline3D
from vcsencode.geom.frames import compute_rmf
from vcsencode.geom.projection import closest_point_tau, theta, rho

def _straight_polyline(n=101, L=100.0):
    z = np.linspace(0.0, L, n)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return Polyline3D(points=pts)

def test_rmf_on_straight_line_minimal_twist():
    pl = _straight_polyline()
    cl = fit_centerline_bspline(pl, L=9, constant_speed=True)
    rmf = compute_rmf(cl, init_rule="centroid-plane", step_mm=5.0)

    ts = np.linspace(0, 1, 51)
    T = rmf.t(ts); V1 = rmf.v1(ts); V2 = rmf.v2(ts)

    # Orthonormality checks
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.linalg.norm(V1, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.linalg.norm(V2, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.sum(T*V1, axis=1), 0.0, atol=1e-7)
    assert np.allclose(np.sum(T*V2, axis=1), 0.0, atol=1e-7)
    assert np.allclose(np.sum(V1*V2, axis=1), 0.0, atol=1e-7)

    # Minimal twist on straight line: V1 should be (nearly) constant
    dots = np.sum(V1 * V1[0], axis=1)
    assert np.all(dots > 0.999)  # within ~2.5 degrees

def test_projection_on_synthetic_cylinder():
    # Straight centerline from z=0..L, RMF constants => cylinder of radius R
    Lz = 120.0
    R = 10.0
    pl = _straight_polyline(n=121, L=Lz)
    cl = fit_centerline_bspline(pl, L=9, constant_speed=True)
    rmf = compute_rmf(cl, step_mm=5.0)

    # Sample random (tau, theta) points on the cylinder, then project back
    rng = np.random.default_rng(42)
    taus = rng.uniform(0.0, 1.0, size=30)
    thetas = rng.uniform(0.0, 2*np.pi, size=30)

    for t, th in zip(taus, thetas):
        c = cl.eval(t)
        v1 = rmf.v1(t); v2 = rmf.v2(t)
        x = c + R*(np.cos(th)*v1 + np.sin(th)*v2)

        t_hat = closest_point_tau(cl, x, tol=1e-10, maxit=50)
        rho_hat = rho(cl, x, t_hat)
        th_hat = theta(cl, rmf, x, t_hat)

        # τ should match within small tolerance (normalized by length)
        assert abs(t_hat - t) < 5e-3

        # ρ should be near R
        assert abs(rho_hat - R) < 1e-6

        # θ modulo 2π should be close
        d = ( (th_hat - th + np.pi) % (2*np.pi) ) - np.pi
        assert abs(d) < 1e-2
