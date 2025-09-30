#!/usr/bin/env python3
"""Produce the τ–θ–ρ surface visualization for the reference VCS encoding."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from trimesh.proximity import ProximityQuery

from vcsencode.io import load_stl, clean_mesh
from vcsencode.encoding.forward import build_model
from vcsencode.encoding.inverse import surface_mesh, cap_meshes_autoslice, cap_meshes
from vcsencode.models import VCSModel

REPO_ROOT = Path(__file__).resolve().parent.parent
STL_PATH = REPO_ROOT / "vcsencode" / "vessel_segment.stl"
OUTPUT_SURFACE_PATH = REPO_ROOT / "figure_vcs_encoding_surface.png"
OUTPUT_COMPARISON_PATH = REPO_ROOT / "figure_vcs_geometry_comparison.png"

MODEL_PARAMS = dict(
    unit_scale=1.0,
    pad_ends_mm=6.0,
    # L=15,
    # K=31,
    # R=25,
    L=9,
    K=19,
    R=15,
    resampling=0.05,
    rays_thetas=512,
    rays_tau_samples=420,
)

FIGURE_PARAMS = dict(
    n_tau=260,
    n_theta=400,
    line_tau=0.20,
    line_theta=5.0 * math.pi / 4.0,
    dpi=600,
    figsize=(13.0, 7.5),
)

COMPARISON_FIG_PARAMS = dict(
    dpi=500,
    figsize=(16.0, 7.6),
    view=(28.0, -55.0),
    centerline_samples=400,
    diff_cmap="magma",
    diff_quantile=0.98,
    subplot_adjust=dict(left=0.025, right=0.975, bottom=0.04, top=0.96, wspace=0.04),
)


def _plot_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    facecolor: tuple | str,
    alpha: float,
    edgecolor: tuple | str = "none",
    linewidth: float = 0.0,
    color_values: np.ndarray | None = None,
    cmap=None,
    norm=None,
) -> None:
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int32)
    if verts.size == 0 or tris.size == 0:
        return
    collection = Poly3DCollection(verts[tris])

    if color_values is not None and cmap is not None:
        values = np.asarray(color_values, dtype=float)
        if values.ndim != 1 or values.shape[0] != tris.shape[0]:
            raise ValueError("color_values must be 1D with length equal to number of faces")
        norm = norm or colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
        face_rgba = cmap(norm(values))
        if alpha is not None:
            face_rgba[:, 3] = np.clip(float(alpha), 0.0, 1.0)
        collection.set_facecolor(face_rgba)
        face_alpha = float(face_rgba[0, 3]) if face_rgba.ndim == 2 else float(alpha)
    else:
        # Handle transparent faces explicitly to keep 3D projection bookkeeping happy.
        if isinstance(facecolor, str) and facecolor.lower() == "none":
            face_rgba = (0.0, 0.0, 0.0, 0.0)
        else:
            face_rgba = colors.to_rgba(facecolor, alpha=max(float(alpha), 0.0))
        collection.set_facecolor(face_rgba)
        face_alpha = float(face_rgba[3]) if isinstance(face_rgba, tuple) else float(alpha)

    if isinstance(edgecolor, str) and edgecolor.lower() == "none":
        edge_rgba = (0.0, 0.0, 0.0, 0.0)
    else:
        edge_rgba = colors.to_rgba(edgecolor)
        if alpha is not None and color_values is None:
            # Apply alpha to edges only if faces are transparent; otherwise keep edge opacity.
            if face_alpha < 1.0:
                edge_rgba = (
                    edge_rgba[0],
                    edge_rgba[1],
                    edge_rgba[2],
                    min(edge_rgba[3], max(float(alpha), 0.0)),
                )
    collection.set_edgecolor(edge_rgba)

    if linewidth is not None:
        collection.set_linewidth(max(float(linewidth), 0.0))
    ax.add_collection3d(collection)


def _compute_bounds(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid = [np.asarray(arr, dtype=float) for arr in arrays if arr is not None and np.asarray(arr).size]
    if not valid:
        return None
    stacked = np.vstack(valid)
    return stacked.min(axis=0), stacked.max(axis=0)


def _apply_bounds(ax, bounds: tuple[np.ndarray, np.ndarray] | None, pad_fraction: float = 0.12) -> None:
    if bounds is None:
        return
    lo, hi = bounds
    center = 0.5 * (lo + hi)
    span = hi - lo
    half = 0.5 * float(np.max(span))
    if not np.isfinite(half) or half <= 0.0:
        half = 1.0
    pad = half * float(np.clip(pad_fraction, 0.0, 1.0))
    radius = half + pad
    for idx, axis in enumerate("xyz"):
        getattr(ax, f"set_{axis}lim")(center[idx] - radius, center[idx] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _sample_centerline(model: VCSModel, n_samples: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, int(max(2, n_samples)), dtype=float)
    return np.asarray(model.centerline.eval(ts), dtype=float)


def _radius_grid(model, taus: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    radius = model.radius
    if not hasattr(radius, "eval_grid"):
        raise AttributeError("Radius surface does not provide eval_grid().")
    return np.asarray(radius.eval_grid(taus, thetas), dtype=float)


def _wrap_theta(theta: float) -> float:
    twopi = 2.0 * math.pi
    return (theta % twopi + twopi) % twopi


def main() -> None:
    if not STL_PATH.exists():
        raise FileNotFoundError(f"Reference STL not found at {STL_PATH}")

    mesh = clean_mesh(load_stl(str(STL_PATH)), repair=True)
    model = build_model(mesh, params=MODEL_PARAMS)

    # Reconstruct the model strictly from the encoding vector to verify that the
    # visualization reflects the serialized representation.
    encoding = model.pack()
    meta = dict(model.meta)
    model_encoded = VCSModel.unpack(encoding, meta)

    taus = np.linspace(0.0, 1.0, FIGURE_PARAMS["n_tau"], dtype=float)
    thetas = np.linspace(0.0, 2.0 * math.pi, FIGURE_PARAMS["n_theta"], endpoint=False, dtype=float)
    rho_direct = _radius_grid(model, taus, thetas)
    rho = _radius_grid(model_encoded, taus, thetas)
    max_diff = float(np.max(np.abs(rho - rho_direct)))
    print(f"max |rho_direct - rho_encoded| = {max_diff:.6e} mm")
    model = model_encoded

    rho_min = float(np.nanmin(rho))
    rho_max = float(np.nanmax(rho))
    rho_mean = float(np.mean(rho))
    rho_std = float(np.std(rho))

    print(
        "rho stats (mm): min={:.4f} max={:.4f} mean={:.4f} std={:.4f}".format(
            rho_min,
            rho_max,
            rho_mean,
            rho_std,
        )
    )
    print(
        "rho at tau=0 range (mm): {:.4f} – {:.4f}".format(
            float(rho[0].min()), float(rho[0].max())
        )
    )

    line_tau = float(np.clip(FIGURE_PARAMS["line_tau"], 0.0, 1.0))
    line_theta = _wrap_theta(float(FIGURE_PARAMS["line_theta"]))

    rho_tau = _radius_grid(model, np.array([line_tau]), thetas).reshape(-1)
    rho_theta = _radius_grid(model, taus, np.array([line_theta])).reshape(-1)
    line_lift = 0.025 * max(rho_max - rho_min, 1e-6)
    rho_tau_vis = rho_tau + line_lift
    rho_theta_vis = rho_theta + line_lift

    tau_grid, theta_grid = np.meshgrid(taus, thetas, indexing="ij")

    rho_floor = 0.0
    norm = colors.Normalize(vmin=rho_min, vmax=rho_max)
    cmap = matplotlib.colormaps.get_cmap("viridis")

    fig = plt.figure(figsize=FIGURE_PARAMS["figsize"], dpi=FIGURE_PARAMS["dpi"])
    ax = fig.add_subplot(111, projection="3d")

    base_plane = np.full_like(tau_grid, rho_floor)
    ax.plot_surface(
        tau_grid,
        theta_grid,
        base_plane,
        color="#d9d9d9",
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=1.0,
    )

    surf = ax.plot_surface(
        tau_grid,
        theta_grid,
        rho,
        cmap=cmap,
        norm=norm,
        linewidth=0.15,
        antialiased=True,
        shade=True,
    )
    ax.plot_wireframe(
        tau_grid,
        theta_grid,
        rho,
        color="#1c1c1c",
        rstride=max(1, FIGURE_PARAMS["n_tau"] // 22),
        cstride=max(1, FIGURE_PARAMS["n_theta"] // 44),
        linewidth=0.45,
    )

    theta_span = np.tile(thetas[None, :], (2, 1))
    tau_lo = np.tile(np.array([taus[0], taus[0]])[:, None], (1, FIGURE_PARAMS["n_theta"]))
    tau_hi = np.tile(np.array([taus[-1], taus[-1]])[:, None], (1, FIGURE_PARAMS["n_theta"]))
    z_tau_lo = np.vstack([np.full(FIGURE_PARAMS["n_theta"], rho_floor), rho[0, :]])
    z_tau_hi = np.vstack([np.full(FIGURE_PARAMS["n_theta"], rho_floor), rho[-1, :]])
    ax.plot_surface(
        tau_lo,
        theta_span,
        z_tau_lo,
        color="#bfbfbf",
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot_surface(
        tau_hi,
        theta_span,
        z_tau_hi,
        color="#bfbfbf",
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    tau_side = np.tile(taus[:, None], (1, 2))
    theta_side_lo = np.tile(np.array([thetas[0], thetas[0]]), (FIGURE_PARAMS["n_tau"], 1))
    theta_side_hi = np.tile(np.array([thetas[-1], thetas[-1]]), (FIGURE_PARAMS["n_tau"], 1))
    z_side_lo = np.column_stack([np.full(FIGURE_PARAMS["n_tau"], rho_floor), rho[:, 0]])
    z_side_hi = np.column_stack([np.full(FIGURE_PARAMS["n_tau"], rho_floor), rho[:, -1]])
    ax.plot_surface(
        tau_side,
        theta_side_lo,
        z_side_lo,
        color="#b5b5b5",
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot_surface(
        tau_side,
        theta_side_hi,
        z_side_hi,
        color="#b5b5b5",
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    highlight_kwargs = dict(color="#ffd400", linewidth=3.4, solid_capstyle="round", zorder=10)
    ax.plot(
        np.full_like(thetas, line_tau),
        thetas,
        rho_tau_vis,
        **highlight_kwargs,
    )
    ax.plot(
        taus,
        np.full_like(taus, line_theta),
        rho_theta_vis,
        **highlight_kwargs,
    )

    idx_tau_line = min(len(taus) - 1, max(0, int(round(0.75 * (len(taus) - 1)))))
    idx_theta_line = min(len(thetas) - 1, max(0, int(round(0.15 * (len(thetas) - 1)))))
    point_theta = (taus[idx_tau_line], line_theta, rho_theta_vis[idx_tau_line])
    point_tau = (line_tau, thetas[idx_theta_line], rho_tau_vis[idx_theta_line])

    ax.scatter(
        [point_theta[0], point_tau[0]],
        [point_theta[1], point_tau[1]],
        [point_theta[2], point_tau[2]],
        color="#111111",
        s=45,
        depthshade=False,
    )

    def annotate(point, label, offset):
        ax.text(
            point[0] + offset[0],
            point[1] + offset[1],
            point[2] + offset[2],
            label,
            fontsize=11,
            ha="left",
            va="bottom",
        )
        ax.plot([point[0], point[0] + offset[0] * 0.55], [point[1], point[1] + offset[1] * 0.55], [point[2], point[2] + offset[2] * 0.55], color="#111111", linewidth=1.0)

    annotate(point_theta, r"$p_1 (\tau_1, \theta_1, \rho_1)$", offset=(0.06, 0.55, 0.04 * (rho_max - rho_min + 1e-6)))
    annotate(point_tau, r"$p_2 (\tau_2, \theta_2, \rho_2)$", offset=(-0.12, -0.45, 0.06 * (rho_max - rho_min + 1e-6)))

    ax.set_xlabel(r"$\tau$", labelpad=18)
    ax.set_ylabel(r"$\theta$", labelpad=18)
    ax.set_zlabel(r"$\rho$ (mm)", labelpad=18)

    # Invert the tau axis so the orientation runs from 1 -> 0 when viewed left to right.
    ax.set_xlim(1.0, 0.0)
    ax.set_ylim(0.0, 2.0 * math.pi)
    ax.set_zlim(rho_floor, rho_max)

    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    theta_ticks = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 2.0 * math.pi]
    theta_labels = [r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_yticks(theta_ticks)
    ax.set_yticklabels(theta_labels)
    ax.tick_params(axis="both", labelsize=11)
    ax.tick_params(axis="z", labelsize=11)

    ax.view_init(elev=28.0, azim=-55.0)
    ax.dist = 8.0
    ax.set_box_aspect((1.8, 0.9, 0.8))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.23, 0.08, 0.54, 0.03])
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_label(r"$\rho$ (mm)", labelpad=6)
    colorbar.ax.tick_params(labelsize=10)
    colorbar.outline.set_visible(False)

    ax.grid(False)

    OUTPUT_SURFACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_SURFACE_PATH, dpi=FIGURE_PARAMS["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUTPUT_SURFACE_PATH.resolve()}")

    # ------------------------------------------------------------------
    # Figure: combined overview (original+centerline | reconstructed difference)
    cl_samples = _sample_centerline(model, COMPARISON_FIG_PARAMS["centerline_samples"])
    verts_orig = np.asarray(mesh.vertices, dtype=float)
    faces_orig = np.asarray(mesh.faces, dtype=np.int32)

    wall_mesh = surface_mesh(model, n_tau=FIGURE_PARAMS["n_tau"], n_theta=FIGURE_PARAMS["n_theta"])
    caps_mesh = cap_meshes_autoslice(model, wall_mesh)
    if caps_mesh.vertices.size == 0 or caps_mesh.faces.size == 0:
        caps_mesh = cap_meshes(model, n_theta=FIGURE_PARAMS["n_theta"])

    recon_vertices = np.asarray(wall_mesh.vertices, dtype=float)
    recon_faces = np.asarray(wall_mesh.faces, dtype=np.int32)
    if caps_mesh.vertices.size and caps_mesh.faces.size:
        cap_vertices = np.asarray(caps_mesh.vertices, dtype=float)
        cap_faces = np.asarray(caps_mesh.faces, dtype=np.int32) + recon_vertices.shape[0]
        recon_vertices = np.vstack([recon_vertices, cap_vertices])
        recon_faces = np.vstack([recon_faces, cap_faces])

    bounds_compare = _compute_bounds(verts_orig, recon_vertices)

    diff_face_values = None
    diff_norm = None
    diff_cmap = matplotlib.colormaps.get_cmap(COMPARISON_FIG_PARAMS["diff_cmap"])
    diff_stats = None
    try:
        tm_orig = trimesh.Trimesh(vertices=verts_orig, faces=faces_orig, process=False)
        tm_recon = trimesh.Trimesh(vertices=recon_vertices, faces=recon_faces, process=False)
        query = ProximityQuery(tm_orig)
        dists = np.abs(query.signed_distance(tm_recon.vertices))
        if np.all(np.isfinite(dists)) and dists.size:
            diff_stats = dict(
                mean=float(np.mean(dists)),
                q95=float(np.quantile(dists, 0.95)),
                max=float(np.max(dists)),
            )
            face_values = np.mean(dists[recon_faces], axis=1)
            vmax = float(np.quantile(face_values, COMPARISON_FIG_PARAMS["diff_quantile"]))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = float(np.max(face_values))
            if np.isfinite(vmax) and vmax > 0.0:
                diff_norm = colors.Normalize(vmin=0.0, vmax=vmax)
                diff_face_values = face_values
            print(
                "difference stats (mm): mean={mean:.4f} q95={q95:.4f} max={max:.4f}".format(
                    **diff_stats
                )
            )
    except Exception as exc:
        print(f"difference visualization unavailable: {exc}")

    fig_cmp = plt.figure(figsize=COMPARISON_FIG_PARAMS["figsize"], dpi=COMPARISON_FIG_PARAMS["dpi"])
    ax_orig = fig_cmp.add_subplot(1, 2, 1, projection="3d")
    ax_recon = fig_cmp.add_subplot(1, 2, 2, projection="3d")

    _plot_mesh(ax_orig, verts_orig, faces_orig, facecolor="#6baed6", alpha=0.18, edgecolor="#4682b4", linewidth=0.15)
    ax_orig.plot3D(
        cl_samples[:, 0],
        cl_samples[:, 1],
        cl_samples[:, 2],
        color="#ff8c00",
        linewidth=2.6,
        solid_capstyle="round",
    )
    _apply_bounds(ax_orig, bounds_compare)
    ax_orig.view_init(*COMPARISON_FIG_PARAMS["view"])
    ax_orig.set_axis_off()
    ax_orig.set_title("Original Geometry + Centerline", pad=12.0)

    if diff_face_values is not None and diff_norm is not None:
        _plot_mesh(
            ax_recon,
            recon_vertices,
            recon_faces,
            facecolor="#ffb347",
            alpha=0.92,
            edgecolor=(0.0, 0.0, 0.0, 0.15),
            linewidth=0.08,
            color_values=diff_face_values,
            cmap=diff_cmap,
            norm=diff_norm,
        )
    else:
        _plot_mesh(
            ax_recon,
            recon_vertices,
            recon_faces,
            facecolor="#ffb347",
            alpha=0.30,
            edgecolor="#d17d00",
            linewidth=0.15,
        )

    _plot_mesh(
        ax_recon,
        verts_orig,
        faces_orig,
        facecolor="none",
        alpha=0.0,
        edgecolor=(0.2, 0.2, 0.2, 0.75),
        linewidth=0.12,
    )
    _apply_bounds(ax_recon, bounds_compare)
    ax_recon.view_init(*COMPARISON_FIG_PARAMS["view"])
    ax_recon.set_axis_off()
    ax_recon.set_title("Reconstruction vs Original", pad=12.0)

    if diff_face_values is not None and diff_norm is not None:
        mappable_diff = cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap)
        mappable_diff.set_array([])
        cbar = fig_cmp.colorbar(
            mappable_diff,
            ax=ax_recon,
            shrink=0.65,
            pad=0.03,
            fraction=0.04,
        )
        cbar.set_label("Wall distance to original (mm)", labelpad=6)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_visible(False)

    fig_cmp.subplots_adjust(**COMPARISON_FIG_PARAMS["subplot_adjust"])
    fig_cmp.savefig(OUTPUT_COMPARISON_PATH, dpi=COMPARISON_FIG_PARAMS["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig_cmp)

    print(f"Wrote {OUTPUT_COMPARISON_PATH.resolve()}")


if __name__ == "__main__":
    main()
