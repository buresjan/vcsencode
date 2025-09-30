from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)
from scipy.interpolate import BSpline, LSQBivariateSpline


@dataclass(slots=True)
class _CenterlineEval:
    knots: np.ndarray
    coeff_x: np.ndarray
    coeff_y: np.ndarray
    coeff_z: np.ndarray
    degree: int

    def __post_init__(self) -> None:
        self._spl_x = BSpline(self.knots, self.coeff_x, self.degree, extrapolate=False)
        self._spl_y = BSpline(self.knots, self.coeff_y, self.degree, extrapolate=False)
        self._spl_z = BSpline(self.knots, self.coeff_z, self.degree, extrapolate=False)

    def eval(self, tau: np.ndarray | float) -> np.ndarray:
        t = np.asarray(tau, dtype=float)
        x = self._spl_x(t)
        y = self._spl_y(t)
        z = self._spl_z(t)
        stacked = np.column_stack([x, y, z]) if t.ndim else np.array([x, y, z], dtype=float)
        return stacked


@dataclass(slots=True)
class _RadiusEval:
    spline: LSQBivariateSpline

    def eval_grid(self, taus: Iterable[float], thetas: Iterable[float]) -> np.ndarray:
        t = _to_float_array(taus)
        th = _to_float_array(thetas)
        return np.asarray(self.spline(t, th), dtype=float)


def _basis_matrix(knots: np.ndarray, degree: int, sample: np.ndarray) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    basis = np.empty((sample.size, n_basis), dtype=float)
    for idx in range(n_basis):
        coeff = np.zeros(n_basis, dtype=float)
        coeff[idx] = 1.0
        spline = BSpline(knots, coeff, degree, extrapolate=False)
        basis[:, idx] = spline(sample)
    return basis


def _to_float_array(values: Iterable[float] | np.ndarray | float) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    if np.isscalar(values):
        return np.array([values], dtype=float)
    return np.array(list(values), dtype=float)


def _load_numpy_model(npz: Any) -> Tuple[_CenterlineEval, _RadiusEval, Dict[str, float]]:
    if isinstance(npz, (str, os.PathLike)):
        with np.load(npz) as data:
            payload = {key: data[key] for key in data.files}
    else:
        payload = dict(npz)

    degree_cl = int(payload.get("cl_degree", 3))
    knots = np.asarray(payload["cl_knots"], dtype=float)
    cx = np.asarray(payload["cl_cx"], dtype=float)
    cy = np.asarray(payload["cl_cy"], dtype=float)
    cz = np.asarray(payload["cl_cz"], dtype=float)
    cl_eval = _CenterlineEval(knots=knots, coeff_x=cx, coeff_y=cy, coeff_z=cz, degree=degree_cl)

    deg_tau = int(payload.get("rs_degree_tau", 3))
    deg_theta = int(payload.get("rs_degree_theta", 3))
    knots_tau = np.asarray(payload["rs_knots_tau"], dtype=float)
    knots_theta = np.asarray(payload["rs_knots_theta"], dtype=float)
    coeffs = np.asarray(payload["rs_coeffs"], dtype=float, order="F")

    n_tau_basis = len(knots_tau) - deg_tau - 1
    n_theta_basis = len(knots_theta) - deg_theta - 1
    if coeffs.shape != (n_tau_basis, n_theta_basis):
        raise ValueError("Coefficient array shape does not match knot configuration.")

    def greville(knots: np.ndarray, degree: int) -> np.ndarray:
        count = len(knots) - degree - 1
        return np.array([np.mean(knots[i + 1 : i + degree + 1]) for i in range(count)], dtype=float)

    tau_g = greville(knots_tau, deg_tau)
    theta_g = greville(knots_theta, deg_theta)
    btau = _basis_matrix(knots_tau, deg_tau, tau_g)
    btheta = _basis_matrix(knots_theta, deg_theta, theta_g)
    z_samples = btau @ coeffs @ btheta.T

    tau_internal = knots_tau[deg_tau + 1 : -deg_tau - 1] if knots_tau.size > 2 * (deg_tau + 1) else np.empty(0, dtype=float)
    theta_internal = (
        knots_theta[deg_theta + 1 : -deg_theta - 1] if knots_theta.size > 2 * (deg_theta + 1) else np.empty(0, dtype=float)
    )

    tau_grid, theta_grid = np.meshgrid(tau_g, theta_g, indexing="ij")
    spline = LSQBivariateSpline(
        tau_grid.ravel(),
        theta_grid.ravel(),
        z_samples.ravel(),
        tau_internal,
        theta_internal,
        kx=deg_tau,
        ky=deg_theta,
        bbox=[0.0, 1.0, 0.0, 2.0 * math.pi],
    )

    rs_eval = _RadiusEval(spline=spline)
    meta = {"unit_scale": float(payload.get("unit_scale", 1.0))}
    return cl_eval, rs_eval, meta


def _from_vcsmodel(model: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    class _CLEval:
        def __init__(self, cl: Any) -> None:
            self._cl = cl

        def eval(self, tau: np.ndarray | float) -> np.ndarray:
            return self._cl.eval(tau)

    class _RSEval:
        def __init__(self, rs: Any) -> None:
            self._rs = rs

        def eval_grid(self, taus: Iterable[float], thetas: Iterable[float]) -> np.ndarray:
            return self._rs.eval_grid(_to_float_array(taus), _to_float_array(thetas))

    meta: Dict[str, Any] = dict(getattr(model, "meta", {}))
    meta.setdefault("unit_scale", 1.0)
    return _CLEval(model.centerline), _RSEval(model.radius), meta


def _rmf_along_centerline(cl_eval: Any, taus: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = cl_eval.eval(taus)
    derivatives = np.gradient(points, taus, axis=0, edge_order=2)
    tangents = derivatives / (np.linalg.norm(derivatives, axis=1, keepdims=True) + 1e-12)

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    v1 = np.cross(ref, tangents[0])
    if np.linalg.norm(v1) < 1e-8:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        v1 = np.cross(ref, tangents[0])
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = np.cross(tangents[0], v1)

    frames_v1 = [v1]
    frames_v2 = [v2]
    for idx in range(1, taus.size):
        t_prev = tangents[idx - 1]
        t_curr = tangents[idx]
        bridge = t_prev + t_curr
        if np.linalg.norm(bridge) < 1e-12:
            frames_v1.append(frames_v1[-1])
            frames_v2.append(frames_v2[-1])
            continue
        bridge = bridge / np.linalg.norm(bridge)
        v1_tmp = frames_v1[-1] - 2.0 * np.dot(frames_v1[-1], bridge) * bridge
        v2_tmp = frames_v2[-1] - 2.0 * np.dot(frames_v2[-1], bridge) * bridge
        v1_tmp -= np.dot(v1_tmp, t_curr) * t_curr
        v1_tmp = v1_tmp / (np.linalg.norm(v1_tmp) + 1e-12)
        v2_tmp = np.cross(t_curr, v1_tmp)
        frames_v1.append(v1_tmp)
        frames_v2.append(v2_tmp)
    return tangents, np.vstack(frames_v1), np.vstack(frames_v2)


def _grid_vertices_faces(
    centerline_points: np.ndarray,
    frame_v1: np.ndarray,
    frame_v2: np.ndarray,
    radii: np.ndarray,
    thetas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_tau, n_theta = radii.shape
    vertices = np.empty((n_tau * n_theta, 3), dtype=float)
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    for idx_tau in range(n_tau):
        base = idx_tau * n_theta
        origin = centerline_points[idx_tau]
        v1 = frame_v1[idx_tau]
        v2 = frame_v2[idx_tau]
        offsets = radii[idx_tau][:, None] * (cos_theta[:, None] * v1 + sin_theta[:, None] * v2)
        vertices[base : base + n_theta, :] = origin + offsets

    faces = []
    for i in range(n_tau - 1):
        row0 = i * n_theta
        row1 = (i + 1) * n_theta
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            a = row0 + j
            b = row0 + jn
            c = row1 + j
            d = row1 + jn
            faces.append([3, a, b, d])
            faces.append([3, a, d, c])
    return vertices, np.asarray(faces, dtype=np.int32)


def _make_arrow(origin: np.ndarray, direction: np.ndarray, scale: float) -> pv.PolyData:
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        norm = 1.0
    return pv.Arrow(
        start=np.asarray(origin, dtype=float),
        direction=direction / norm,
        scale=scale,
        tip_length=0.25,
        tip_radius=0.08,
        shaft_radius=0.035,
    )


def make_vcs_overview(
    model_or_npz: Any,
    out_path: str,
    *,
    n_tau: int = 240,
    n_theta: int = 360,
    line_tau: float = 0.20,
    line_theta: float = 5.0 * math.pi / 4.0,
    cmap: str = "viridis",
    dpi: int = 400,
    right_panel_elev: float = 35.0,
    right_panel_azim: float = -60.0,
    vessel_camera: Dict[str, Any] | None = None,
) -> None:
    if hasattr(model_or_npz, "centerline") and hasattr(model_or_npz, "radius"):
        cl_eval, rs_eval, meta = _from_vcsmodel(model_or_npz)
    else:
        cl_eval, rs_eval, meta = _load_numpy_model(model_or_npz)

    n_tau = int(n_tau)
    n_theta = int(n_theta)
    taus = np.linspace(0.0, 1.0, n_tau)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    line_tau_clamped = float(min(max(line_tau, 0.0), 1.0))
    remainder = float(math.fmod(line_theta, 2.0 * math.pi))
    if remainder < 0.0:
        remainder += 2.0 * math.pi
    line_theta_wrapped = remainder
    scale = float(meta.get("unit_scale", 1.0))

    radii = rs_eval.eval_grid(taus, thetas) * scale
    rho_theta = rs_eval.eval_grid(taus, [line_theta_wrapped]).reshape(-1) * scale
    rho_tau = rs_eval.eval_grid([line_tau_clamped], thetas).reshape(-1) * scale

    centerline_grid = cl_eval.eval(taus) * scale
    taus_for_frames = np.unique(np.concatenate([taus, np.array([line_tau_clamped], dtype=float)]))
    tangents_all, frames_v1_all, frames_v2_all = _rmf_along_centerline(cl_eval, taus_for_frames)

    def _lookup(target: float) -> int:
        diff = np.abs(taus_for_frames - target)
        idx = int(np.argmin(diff))
        return idx

    indices_grid = np.array([_lookup(float(t)) for t in taus], dtype=int)
    frames_v1 = frames_v1_all[indices_grid]
    frames_v2 = frames_v2_all[indices_grid]

    idx_tau_line = _lookup(line_tau_clamped)
    origin_line_tau = cl_eval.eval([line_tau_clamped]).reshape(1, 3)[0] * scale
    v1_line_tau = frames_v1_all[idx_tau_line]
    v2_line_tau = frames_v2_all[idx_tau_line]
    t_line_tau = tangents_all[idx_tau_line]

    vertices, face_array = _grid_vertices_faces(centerline_grid, frames_v1, frames_v2, radii, thetas)

    cos_theta_line = math.cos(line_theta_wrapped)
    sin_theta_line = math.sin(line_theta_wrapped)
    highlight_theta = centerline_grid + rho_theta[:, None] * (
        cos_theta_line * frames_v1 + sin_theta_line * frames_v2
    )
    highlight_tau = origin_line_tau + rho_tau[:, None] * (
        np.cos(thetas)[:, None] * v1_line_tau + np.sin(thetas)[:, None] * v2_line_tau
    )

    rho_intersection = float(
        rs_eval.eval_grid([line_tau_clamped], [line_theta_wrapped]).reshape(-1)[0]
    ) * scale
    intersection_point = origin_line_tau + rho_intersection * (
        cos_theta_line * v1_line_tau + sin_theta_line * v2_line_tau
    )

    idx_p1 = min(n_tau - 1, max(0, int(round(0.75 * (n_tau - 1)))))
    idx_p2 = min(n_theta - 1, max(0, int(round(0.15 * (n_theta - 1)))))
    sample_point_theta = highlight_theta[idx_p1]
    sample_point_tau = highlight_tau[idx_p2]

    z_min = float(np.nanmin(radii))
    z_max = float(np.nanmax(radii))
    clim = (z_min, z_max)
    norm = colors.Normalize(vmin=z_min, vmax=z_max)

    theta_period = 2.0 * math.pi
    z_axis_min = 0.0
    z_axis_max = float(np.ceil((z_max + 1e-6) * 20.0) / 20.0)
    slab_span = max(z_axis_max - z_axis_min, 1e-6)

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 960))
    plotter.set_background("white")
    plotter.enable_parallel_projection()
    try:
        plotter.enable_lightkit()
    except AttributeError:
        pass
    surface = pv.PolyData(vertices, face_array.ravel())
    surface["rho"] = radii.ravel(order="C")

    cut_theta = (line_theta_wrapped + math.radians(50.0)) % (2.0 * math.pi)
    cut_dirs = np.cos(cut_theta) * frames_v1 + np.sin(cut_theta) * frames_v2
    cut_dir_mean = cut_dirs.mean(axis=0)
    if np.linalg.norm(cut_dir_mean) < 1e-8:
        cut_dir_mean = np.array([1.0, 0.0, 0.0], dtype=float)
    cut_normal = cut_dir_mean / np.linalg.norm(cut_dir_mean)
    cut_origin = intersection_point

    surface_open = surface.clip(normal=cut_normal, origin=cut_origin, invert=True)
    clip_face = surface.slice(normal=cut_normal, origin=cut_origin)

    if "rho" not in surface_open.array_names:
        surface_open = surface_open.interpolate(surface)
    if "rho" not in clip_face.array_names:
        clip_face = clip_face.interpolate(surface)

    if surface_open.n_points == 0:
        surface_open = surface

    plotter.add_mesh(
        surface_open,
        scalars="rho",
        cmap=cmap,
        clim=clim,
        smooth_shading=False,
        ambient=0.18,
        specular=0.05,
        diffuse=0.9,
        show_edges=True,
        edge_color="#1b1b1b",
        line_width=0.55,
        show_scalar_bar=False,
        nan_color="white",
    )
    if clip_face.n_points > 0:
        plotter.add_mesh(
            clip_face,
            color="#e0e0e0",
            smooth_shading=False,
            show_edges=True,
            edge_color="#555555",
            line_width=0.8,
            opacity=1.0,
        )

    bbox = vertices.ptp(axis=0)
    diag = float(np.linalg.norm(bbox)) or 1.0
    tube_radius = 0.008 * diag
    spline_centerline = pv.Spline(centerline_grid, max(n_tau * 2, 400))
    plotter.add_mesh(spline_centerline.tube(radius=tube_radius, n_sides=24), color="#ff8c00")
    plotter.add_mesh(
        pv.Spline(highlight_theta, highlight_theta.shape[0]).tube(radius=0.8 * tube_radius, n_sides=20),
        color="#ffd400",
    )
    plotter.add_mesh(
        pv.Spline(highlight_tau, highlight_tau.shape[0]).tube(radius=0.8 * tube_radius, n_sides=20),
        color="#ffd400",
    )

    glyph_scale = max(1e-6, 0.2 * diag)
    plotter.add_mesh(_make_arrow(intersection_point, t_line_tau, glyph_scale), color="#000000")
    plotter.add_mesh(_make_arrow(intersection_point, v1_line_tau, 0.85 * glyph_scale), color="#d62828")
    plotter.add_mesh(_make_arrow(intersection_point, v2_line_tau, 0.85 * glyph_scale), color="#006bff")

    sphere_radius = max(1e-6, 0.014 * diag)
    for pt in (sample_point_theta, sample_point_tau):
        plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=pt), color="#111111", specular=0.0)

    if vessel_camera:
        for key, value in vessel_camera.items():
            setattr(plotter.camera, key, value)
    else:
        center = intersection_point
        view_dir = -cut_normal * 0.9 + 0.55 * t_line_tau + 0.20 * v2_line_tau
        eye = center + view_dir * (diag if diag > 0.0 else 1.0)
        plotter.camera.position = tuple(eye)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.up = (0.0, 0.0, 1.0)
        plotter.camera.zoom(1.55)
    plotter.add_axes(line_width=1.0, color="black")

    with tempfile.NamedTemporaryFile(suffix="_pv.png", delete=False) as tmp_png:
        left_image_path = tmp_png.name
    plotter.show(screenshot=left_image_path, auto_close=True)

    fig = plt.figure(figsize=(13.0, 6.6), dpi=dpi)
    ax_left = fig.add_axes([0.03, 0.10, 0.46, 0.84])
    ax_left.axis("off")
    left_image = plt.imread(left_image_path)
    ax_left.imshow(left_image)

    ax_right = fig.add_axes([0.53, 0.12, 0.44, 0.80], projection="3d")
    tau_grid, theta_grid = np.meshgrid(taus, thetas, indexing="ij")
    zero_plane = np.full_like(tau_grid, z_axis_min)

    ax_right.plot_surface(
        tau_grid,
        theta_grid,
        zero_plane,
        color="#d9d9d9",
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=1.0,
    )
    ax_right.plot_surface(
        tau_grid,
        theta_grid,
        radii,
        cmap=cmap,
        norm=norm,
        rstride=2,
        cstride=2,
        linewidth=0.0,
        antialiased=False,
        shade=True,
    )
    ax_right.plot_wireframe(
        tau_grid,
        theta_grid,
        radii,
        color="#1c1c1c",
        rstride=max(1, n_tau // 18),
        cstride=max(1, n_theta // 36),
        linewidth=0.45,
    )

    tau_side = np.tile(taus[:, None], (1, 2))
    theta_side_lo = np.tile(np.array([thetas[0], thetas[0]]), (n_tau, 1))
    theta_side_hi = np.tile(np.array([thetas[-1], thetas[-1]]), (n_tau, 1))
    z_side_lo = np.column_stack([np.full(n_tau, z_axis_min), radii[:, 0]])
    z_side_hi = np.column_stack([np.full(n_tau, z_axis_min), radii[:, -1]])
    ax_right.plot_surface(
        tau_side,
        theta_side_lo,
        z_side_lo,
        color="#c8c8c8",
        linewidth=0,
        shade=False,
        antialiased=False,
    )
    ax_right.plot_surface(
        tau_side,
        theta_side_hi,
        z_side_hi,
        color="#c8c8c8",
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    theta_span = np.tile(thetas[None, :], (2, 1))
    tau_lo = np.tile(np.array([taus[0], taus[0]])[:, None], (1, n_theta))
    tau_hi = np.tile(np.array([taus[-1], taus[-1]])[:, None], (1, n_theta))
    z_tau_lo = np.vstack([np.full(n_theta, z_axis_min), radii[0, :]])
    z_tau_hi = np.vstack([np.full(n_theta, z_axis_min), radii[-1, :]])
    ax_right.plot_surface(
        tau_lo,
        theta_span,
        z_tau_lo,
        color="#c1c1c1",
        linewidth=0,
        shade=False,
        antialiased=False,
    )
    ax_right.plot_surface(
        tau_hi,
        theta_span,
        z_tau_hi,
        color="#c1c1c1",
        linewidth=0,
        shade=False,
        antialiased=False,
    )

    ax_right.plot(
        np.full_like(thetas, line_tau_clamped),
        thetas,
        rho_tau,
        color="#ffd400",
        linewidth=3.0,
        solid_capstyle="round",
    )
    ax_right.plot(
        taus,
        np.full_like(taus, line_theta_wrapped),
        rho_theta,
        color="#ffd400",
        linewidth=3.0,
        solid_capstyle="round",
    )
    point_theta = (taus[idx_p1], line_theta_wrapped, rho_theta[idx_p1])
    point_tau = (line_tau_clamped, thetas[idx_p2], rho_tau[idx_p2])
    ax_right.scatter(
        [point_theta[0], point_tau[0]],
        [point_theta[1], point_tau[1]],
        [point_theta[2], point_tau[2]],
        color="#111111",
        s=36,
        depthshade=False,
    )
    ax_right.set_xlabel(r"$\tau$", labelpad=8)
    ax_right.set_ylabel(r"$\theta$", labelpad=8)
    ax_right.set_zlabel(r"$\rho$ (mm)", labelpad=8)
    ax_right.view_init(elev=right_panel_elev, azim=right_panel_azim)
    ax_right.dist = 7.5
    ax_right.set_xlim(0.0, 1.0)
    ax_right.set_ylim(0.0, theta_period)
    ax_right.set_zlim(z_axis_min, z_axis_max)
    ax_right.xaxis.pane.set_edgecolor("#bcbcbc")
    ax_right.yaxis.pane.set_edgecolor("#bcbcbc")
    ax_right.zaxis.pane.set_edgecolor("#bcbcbc")
    ax_right.grid(False)
    vertical_exaggeration = 3.2
    ax_right.set_box_aspect((1.0, theta_period, slab_span * vertical_exaggeration))
    ax_right.tick_params(axis="both", labelsize=9)
    ax_right.zaxis.set_rotate_label(False)
    ax_right.set_xticks(np.linspace(0.0, 1.0, 5))
    theta_ticks = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, theta_period]
    theta_labels = [r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax_right.set_yticks(theta_ticks)
    ax_right.set_yticklabels(theta_labels)
    ax_right.set_zticks(np.linspace(z_axis_min, z_axis_max, 5))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.035])
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_label("Rho (mm)", labelpad=4)
    ticks = np.linspace(z_min, z_max, num=5)
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.outline.set_visible(False)

    extension = os.path.splitext(out_path)[1].lower()
    if extension not in {".png", ".pdf", ".svg"}:
        out_path = f"{out_path}.png"

    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    try:
        os.remove(left_image_path)
    except OSError:
        pass
