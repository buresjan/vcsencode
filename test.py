# boolean_difference_surfaces_vtk.py
# Usage: python boolean_difference_surfaces_vtk.py A.stl B.stl out.stl
import sys
import math
import vtk

def read_stl(path):
    r = vtk.vtkSTLReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()

def polydata_bounds_size(pd):
    b = [0]*6
    pd.GetBounds(b)
    dx, dy, dz = b[1]-b[0], b[3]-b[2], b[5]-b[4]
    return (dx, dy, dz), b

def make_implicit_from_surface(polydata, max_dist, sample_spacing):
    """
    Turn an open surface into a 'thickened' implicit function using vtkImplicitModeller.
    - max_dist controls the thickness/halo captured around the surface (in your units).
    - sample_spacing controls voxel resolution.
    """
    imp = vtk.vtkImplicitModeller()
    imp.SetInputData(polydata)
    imp.SetMaximumDistance(max_dist)         # influence radius around the surface
    imp.SetAdjustBounds(1)                   # expand bounds by max_dist
    imp.SetCapping(1)                        # helps produce closed shells
    # Set sample grid dimensions based on bounding box and spacing
    (dx, dy, dz), bounds = polydata_bounds_size(polydata)
    nx = max(8, int(math.ceil(dx / sample_spacing)))
    ny = max(8, int(math.ceil(dy / sample_spacing)))
    nz = max(8, int(math.ceil(dz / sample_spacing)))
    imp.SetSampleDimensions(nx, ny, nz)
    imp.Update()
    # vtkImplicitModeller outputs a vtkImageData where value == 0 on the surface.
    # Convert to an implicit function via vtkImplicitDataSet so we can boolean them.
    ids = vtk.vtkImplicitDataSet()
    ids.SetDataSet(imp.GetOutput())
    return ids

def boolean_difference_implicit(impA, impB):
    """ Build an implicit boolean: A \ B """
    boolean = vtk.vtkImplicitBoolean()
    # Difference means: inside A AND NOT inside B (implicit <= 0 is 'inside')
    boolean.SetOperationTypeToDifference()
    boolean.AddFunction(impA)
    boolean.AddFunction(impB)
    return boolean

def sample_and_extract_surface(implicit_fn, bounds, sample_spacing):
    # Sample the implicit function on a regular grid that covers both inputs
    sampler = vtk.vtkSampleFunction()
    sampler.SetImplicitFunction(implicit_fn)
    # Expand bounds a hair so we don't clip the surface
    pad = sample_spacing * 2.0
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    sampler.SetModelBounds(xmin-pad, xmax+pad, ymin-pad, ymax+pad, zmin-pad, zmax+pad)
    nx = max(16, int(math.ceil((xmax - xmin) / sample_spacing)))
    ny = max(16, int(math.ceil((ymax - ymin) / sample_spacing)))
    nz = max(16, int(math.ceil((zmax - zmin) / sample_spacing)))
    sampler.SetSampleDimensions(nx, ny, nz)
    sampler.ComputeNormalsOff()
    sampler.Update()

    # Extract the 0-isosurface (implicit == 0)
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(sampler.GetOutputPort())
    contour.SetValue(0, 0.0)
    contour.Update()

    # Optional: clean + smooth a bit
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(contour.GetOutputPort())
    clean.Update()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(clean.GetOutputPort())
    smooth.SetNumberOfIterations(15)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.SetPassBand(0.01)  # gentle
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(smooth.GetOutputPort())
    tri.Update()

    return tri.GetOutput()

def write_stl(polydata, path):
    w = vtk.vtkSTLWriter()
    w.SetFileName(path)
    w.SetInputData(polydata)
    w.SetFileTypeToBinary()
    w.Write()

def combine_bounds(a_bounds, b_bounds):
    return (min(a_bounds[0], b_bounds[0]),
            max(a_bounds[1], b_bounds[1]),
            min(a_bounds[2], b_bounds[2]),
            max(a_bounds[3], b_bounds[3]),
            min(a_bounds[4], b_bounds[4]),
            max(a_bounds[5], b_bounds[5]))

def main():
    if len(sys.argv) < 4:
        print("Usage: python boolean_difference_surfaces_vtk.py A.stl B.stl out.stl")
        sys.exit(1)

    pathA, pathB, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    A = read_stl(pathA)
    B = read_stl(pathB)

    # --- Choose resolution parameters ---
    # Inspect bounding box to pick reasonable defaults:
    (dxA, dyA, dzA), bA = polydata_bounds_size(A)
    (dxB, dyB, dzB), bB = polydata_bounds_size(B)
    world_bounds = combine_bounds(bA, bB)
    diag = math.sqrt((world_bounds[1]-world_bounds[0])**2 +
                     (world_bounds[3]-world_bounds[2])**2 +
                     (world_bounds[5]-world_bounds[4])**2)

    # Set sample spacing ~1/300 of scene diagonal (tune as needed)
    sample_spacing = max( (diag / 300.0), 1e-3 )
    # Make the implicit influence radius a few spacings thick
    max_dist = 3.0 * sample_spacing

    # --- Build implicit fields from the (open) surfaces ---
    impA = make_implicit_from_surface(A, max_dist=max_dist, sample_spacing=sample_spacing)
    impB = make_implicit_from_surface(B, max_dist=max_dist, sample_spacing=sample_spacing)

    # --- Boolean (A \ B) in implicit space ---
    impDiff = boolean_difference_implicit(impA, impB)

    # --- Extract final surface ---
    diff_surf = sample_and_extract_surface(impDiff, world_bounds, sample_spacing)

    # --- Save ---
    write_stl(diff_surf, out_path)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
