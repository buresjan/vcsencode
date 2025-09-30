def test_imports():
    import vcsencode
    from vcsencode import Mesh3D, Polyline3D, CenterlineBSpline, RMF, RadiusSurfaceBSpline, VCSModel
    assert hasattr(vcsencode, "__version__")
    assert Mesh3D and Polyline3D and CenterlineBSpline and RMF and RadiusSurfaceBSpline and VCSModel
