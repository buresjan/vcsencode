import os
import numpy as np
import pytest

from vcsencode.io import load_stl, clean_mesh, detect_caps, cap_centers, inward_seed_points

STL_PATH = "vessel_segment.stl"

@pytest.mark.skipif(not os.path.exists(STL_PATH), reason="vessel_segment.stl not found in repo root")
def test_caps_and_seeds():
    mesh = load_stl(STL_PATH)
    mesh = clean_mesh(mesh, repair=True)
    caps = detect_caps(mesh)
    centers = cap_centers(caps)
    assert centers.shape == (2, 3)
    p0, p1 = inward_seed_points(mesh, caps, offset_mm=2.0)
    # Sanity: seeds differ from centers and are finite
    assert np.isfinite(p0).all() and np.isfinite(p1).all()
    assert np.linalg.norm(p0 - centers[0]) > 0.1
    assert np.linalg.norm(p1 - centers[1]) > 0.1
