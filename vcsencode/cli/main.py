#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vcsencode.io import load_stl, clean_mesh
from vcsencode.encoding.forward import build_model
from vcsencode.encoding.inverse import export_stl
from vcsencode.encoding.npzio import save_npz, load_npz
from vcsencode.figures import make_vcs_overview


def _optional_float(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {"none", "auto"}:
        return None
    return float(value)


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).expanduser().parent
    if parent != Path('.'):
        parent.mkdir(parents=True, exist_ok=True)


def _resolve_model_resource(resource: str) -> Any:
    path = Path(resource)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return path
    if not path.exists():
        raise FileNotFoundError(f"Model resource '{resource}' not found.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _add_encode_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--stl", required=True, help="Input STL file path.")
    sub.add_argument("--npz", required=True, help="Output NPZ file path.")
    sub.add_argument("--unit-scale", type=float, default=1.0, help="Scale factor to convert input units to mm.")
    sub.add_argument("--pad-ends-mm", type=_optional_float, default=60.0, help="Extrusion length for end padding (use 'auto' to detect automatically).")
    sub.add_argument("--L", type=int, default=15, help="Centerline control points.")
    sub.add_argument("--K", type=int, default=31, help="Longitudinal basis count for radius surface.")
    sub.add_argument("--R", type=int, default=25, help="Angular basis count for radius surface.")
    sub.add_argument("--resampling", type=float, default=0.20, help="VMTK centerline resampling step (mm).")
    sub.add_argument("--rays-thetas", type=int, default=512, help="Theta samples for radius casting.")
    sub.add_argument("--rays-tau-samples", type=int, default=240, help="Tau samples for radius casting.")
    sub.add_argument("--rmf-step-mm", type=float, default=0.5, help="Step size (mm) for RMF sampling (reserved).")
    sub.add_argument("--theta-anchor", default="rho_argmax", choices=["rho_argmax", "none"], help="Deterministic theta anchoring strategy.")
    sub.add_argument("--theta-periodic", dest="theta_periodic", action="store_true", help="Use periodic cubic B-spline in theta.")
    sub.add_argument("--no-theta-periodic", dest="theta_periodic", action="store_false", help="Disable periodic theta spline (use open-clamped basis).")
    sub.set_defaults(theta_periodic=True)
    sub.add_argument("--theta-periodic-lambda", type=float, default=1e-10, help="Regularization for periodic theta fit (ridge term).")
    sub.add_argument("--clip-percentile", type=float, default=99.5, help="High-percentile clipping for radius samples.")
    sub.add_argument("--clip-hi-factor", type=float, default=3.0, help="Median-based clipping factor for radius samples.")
    sub.add_argument("--clip-enable", dest="clip_enable", action="store_true", help="Enable clipping of outlier radii.")
    sub.add_argument("--clip-disable", dest="clip_enable", action="store_false", help="Disable clipping of outlier radii.")
    sub.set_defaults(clip_enable=True)


def _add_decode_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--npz", required=True, help="Input NPZ model path.")
    sub.add_argument("--stl", required=True, help="Output STL path.")
    sub.add_argument("--n-tau", type=int, default=320, help="Tau samples for reconstruction.")
    sub.add_argument("--n-theta", type=int, default=420, help="Theta samples for reconstruction.")
    sub.add_argument("--cap-mode", choices=["autoslice", "rho"], default="autoslice", help="Cap generation strategy.")


def _add_figure_arguments(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model", required=True, help="Path to NPZ or pickled VCSModel.")
    sub.add_argument("--out", required=True, help="Output figure path.")
    sub.add_argument("--n-tau", type=int, default=240, help="Tau samples for wall reconstruction.")
    sub.add_argument("--n-theta", type=int, default=360, help="Theta samples for wall reconstruction.")
    sub.add_argument("--line-tau", type=float, default=0.20, help="Highlighted tau value.")
    sub.add_argument("--line-theta", type=float, default=5.0 * np.pi / 4.0, help="Highlighted theta value (radians).")
    sub.add_argument("--dpi", type=int, default=400, help="Figure DPI for raster outputs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vcsencode", description="VCS encode/decode/figure CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode STL into NPZ model.")
    _add_encode_arguments(encode_parser)

    decode_parser = subparsers.add_parser("decode", help="Decode NPZ model into STL.")
    _add_decode_arguments(decode_parser)

    figure_parser = subparsers.add_parser("figure", help="Generate overview figure from model.")
    _add_figure_arguments(figure_parser)

    return parser


def _run_encode(args: argparse.Namespace) -> None:
    mesh = clean_mesh(load_stl(args.stl), repair=True)
    pad_value: float | None
    if args.pad_ends_mm is None or np.isnan(args.pad_ends_mm):
        pad_value = None
    else:
        pad_value = float(args.pad_ends_mm)

    params = {
        "unit_scale": float(args.unit_scale),
        "pad_ends_mm": pad_value,
        "L": int(args.L),
        "K": int(args.K),
        "R": int(args.R),
        "resampling": float(args.resampling),
        "rays_thetas": int(args.rays_thetas),
        "rays_tau_samples": int(args.rays_tau_samples),
        "rmf_step_mm": float(args.rmf_step_mm),
        "theta_anchor": args.theta_anchor,
        "theta_periodic": bool(args.theta_periodic),
        "theta_periodic_lambda": float(args.theta_periodic_lambda),
        "clip_enable": bool(args.clip_enable),
        "clip_percentile": float(args.clip_percentile),
        "clip_hi_factor": float(args.clip_hi_factor),
    }
    model = build_model(mesh, params=params)

    _ensure_parent(args.npz)
    save_npz(model, args.npz)
    print(f"[encode] wrote {args.npz}")


def _run_decode(args: argparse.Namespace) -> None:
    model = load_npz(args.npz)
    _ensure_parent(args.stl)
    export_stl(model, args.stl, n_tau=int(args.n_tau), n_theta=int(args.n_theta), cap_mode=args.cap_mode)
    print(f"[decode] wrote {args.stl}")


def _run_figure(args: argparse.Namespace) -> None:
    model_resource = _resolve_model_resource(args.model)
    _ensure_parent(args.out)
    make_vcs_overview(
        model_resource,
        args.out,
        n_tau=int(args.n_tau),
        n_theta=int(args.n_theta),
        line_tau=float(args.line_tau),
        line_theta=float(args.line_theta),
        dpi=int(args.dpi),
    )
    print(f"[figure] wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "encode":
        _run_encode(args)
    elif args.command == "decode":
        _run_decode(args)
    elif args.command == "figure":
        _run_figure(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
