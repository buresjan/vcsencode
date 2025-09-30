"""
vcsencode: Modular, reversible Vessel Coordinate System (VCS) pipeline.

This package will expose:
- Core dataclasses (Mesh3D, Polyline3D, CenterlineBSpline, RMF, RadiusSurfaceBSpline, VCSModel)
- High-level build/encode/decode utilities (implemented in later prompts)
"""

from .models import (
    Mesh3D,
    Polyline3D,
    CenterlineBSpline,
    RMF,
    RadiusSurfaceBSpline,
    VCSModel,
)

__all__ = [
    "Mesh3D",
    "Polyline3D",
    "CenterlineBSpline",
    "RMF",
    "RadiusSurfaceBSpline",
    "VCSModel",
]

__version__ = "0.1.0"
