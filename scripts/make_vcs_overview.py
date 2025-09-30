#!/usr/bin/env python3
"""CLI wrapper for producing the publication-style VCS overview figure."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from vcsencode.figures import make_vcs_overview


def _resolve_model(resource: str) -> Any:
    path = Path(resource)
    if path.suffix.lower() == ".npz":
        return path
    if not path.exists():
        raise FileNotFoundError(f"Model resource '{resource}' not found.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the two-panel VCS overview figure.")
    parser.add_argument("--model", required=True, help="Path to .npz or pickled VCSModel.")
    parser.add_argument("--out", required=True, help="Output figure path (.png/.pdf/.svg).")
    parser.add_argument("--n-tau", type=int, default=240, help="Number of τ samples for wall reconstruction.")
    parser.add_argument("--n-theta", type=int, default=360, help="Number of θ samples for wall reconstruction.")
    parser.add_argument("--line-tau", type=float, default=0.20, help="τ value for the highlight curve.")
    parser.add_argument("--line-theta", type=float, default=5.0 * 3.141592653589793 / 4.0, help="θ value for the highlight curve.")
    parser.add_argument("--dpi", type=int, default=400, help="Output figure DPI.")
    parser.add_argument("--style", default=None, help="Reserved for future style overrides.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model = _resolve_model(args.model)
    make_vcs_overview(
        model,
        args.out,
        n_tau=args.n_tau,
        n_theta=args.n_theta,
        line_tau=args.line_tau,
        line_theta=args.line_theta,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

