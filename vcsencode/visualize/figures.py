"""Legacy shim forwarding to the modern figure generator in :mod:`vcsencode.figures`."""

from ..figures.overview import make_vcs_overview as make_fig3_overview
from ..figures.overview import make_vcs_overview

__all__ = ["make_vcs_overview", "make_fig3_overview"]
