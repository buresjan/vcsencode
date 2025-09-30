from __future__ import annotations
import numpy as np
import trimesh

from ..models import Mesh3D
from ..io import CapInfo


def _to_trimesh(mesh: Mesh3D) -> trimesh.Trimesh:
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    tm.remove_unreferenced_vertices()
    return tm


def _plane_basis(normal: np.ndarray):
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], float)
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], float)
    u = a - np.dot(a, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v = v / (np.linalg.norm(v) + 1e-12)
    return u, v, n


def _cap_boundary_ring(mesh: Mesh3D, cap: CapInfo) -> tuple[np.ndarray, np.ndarray]:
    """
    Return an ordered list of vertex indices forming the boundary loop between the cap patch and the wall,
    and the corresponding Nx3 coordinates (ordered CCW when seen from outside along +normal).
    We order points by angle in the cap plane around the cap center.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    capF = np.asarray(cap.face_indices, dtype=np.int64)
    cap_verts = np.unique(F[capF].ravel())
    # Boundary ring vertices are those that also appear in any NON-cap face
    in_cap = np.zeros(V.shape[0], dtype=bool)
    in_cap[cap_verts] = True
    # Faces not in cap
    otherF = np.delete(F, capF, axis=0)
    other_verts = np.unique(otherF.ravel())
    ring_ids = np.intersect1d(cap_verts, other_verts, assume_unique=False)

    # Order by polar angle in cap plane
    c = np.asarray(cap.center, dtype=float).reshape(3)
    n = np.asarray(cap.plane_normal, dtype=float).reshape(3)
    u, v, n = _plane_basis(n)
    P = V[ring_ids]
    Q = P - c[None, :]
    x = Q @ u; y = Q @ v
    th = np.arctan2(y, x)
    order = np.argsort(th)
    ring_ids = ring_ids[order]
    ring_xyz = V[ring_ids]
    return ring_ids, ring_xyz


def _triangulate_between_rings(ringA_ids: np.ndarray, ringB_ids: np.ndarray, V: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Triangulate a cylindrical strip between two rings with consistent indexing.
    Returns faces (M, 3) with correct orientation (outward ~ 'normal').
    """
    m = len(ringA_ids)
    assert m == len(ringB_ids) and m >= 3
    F = []
    n = np.asarray(normal, dtype=float).reshape(3)
    for j in range(m):
        j2 = (j + 1) % m
        a = ringA_ids[j]; b = ringA_ids[j2]
        a2 = ringB_ids[j]; b2 = ringB_ids[j2]
        # Candidate triangles
        t1 = [a, b, b2]
        t2 = [a, b2, a2]
        # Orientation check on first triangle only (cheap heuristic)
        if j == 0:
            p0, p1, p2 = V[t1]
            nr = np.cross(p1 - p0, p2 - p0)
            if float(np.dot(nr, n)) < 0.0:
                # swap winding
                t1 = [a, b2, b]
                t2 = [a, a2, b2]
        F.append(t1); F.append(t2)
    return np.asarray(F, dtype=np.int32)


def extrude_cap_end(mesh: Mesh3D, cap: CapInfo, length_mm: float) -> Mesh3D:
    """
    Remove the given cap patch and extrude outward by 'length_mm' along its plane normal.
    Produces a new watertight mesh with a translated copy of the cap at the far end.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int32)
    capF = np.asarray(cap.face_indices, dtype=np.int64)
    keep = np.ones(len(F), dtype=bool)
    keep[capF] = False

    ring_ids, ring_xyz = _cap_boundary_ring(mesh, cap)
    n = np.asarray(cap.plane_normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    # New ring at the far end
    ring2_xyz = ring_xyz + length_mm * n[None, :]
    # Append ring2 vertices
    base_count = V.shape[0]
    ring2_ids = np.arange(base_count, base_count + len(ring_ids), dtype=np.int32)
    V_ext = np.vstack([V, ring2_xyz])

    # Side strip faces between ring and ring2
    F_side = _triangulate_between_rings(ring_ids.astype(np.int32), ring2_ids, V_ext, normal=n)

    # Build far cap by translating all unique cap vertices
    cap_unique = np.unique(F[capF].ravel())
    cap_map = {int(vid): int(i) for i, vid in enumerate(cap_unique)}
    cap_new_ids = np.arange(V_ext.shape[0], V_ext.shape[0] + cap_unique.size, dtype=np.int32)
    # Append all translated cap vertices
    V_translated = V[cap_unique] + length_mm * n[None, :]
    V_ext = np.vstack([V_ext, V_translated])
    # Remap original cap faces to the translated vertex block
    F_cap_far = []
    offset = V_ext.shape[0] - V_translated.shape[0]  # start index of translated block
    for tri in F[capF]:
        a, b, c = [cap_map[int(x)] for x in tri]
        F_cap_far.append([offset + a, offset + b, offset + c])
    F_cap_far = np.asarray(F_cap_far, dtype=np.int32)

    # Combine: keep other faces, plus side, plus far cap
    F_new = np.vstack([F[keep], F_side, F_cap_far]).astype(np.int32)
    return Mesh3D(vertices=V_ext, faces=F_new, units=mesh.units)


def pad_mesh_ends(mesh: Mesh3D, caps: list[CapInfo], length_mm: float | None = None,
                  factor: float = 3.0, min_len: float = 30.0, max_len: float = 200.0) -> Mesh3D:
    """
    Extrude BOTH ends outward in a single pass using cap face indices that refer
    to the ORIGINAL mesh. This avoids invalidating indices mid-way.

    If length_mm is None, choose per-end length as:
        L_end = clip(factor * median_ring_radius, [min_len, max_len])
    """
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int32)

    # Per-end geometry on the ORIGINAL mesh
    ring0_ids, ring0_xyz = _cap_boundary_ring(mesh, caps[0])
    ring1_ids, ring1_xyz = _cap_boundary_ring(mesh, caps[1])
    n0 = np.asarray(caps[0].plane_normal, dtype=float).reshape(3)
    n1 = np.asarray(caps[1].plane_normal, dtype=float).reshape(3)
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)

    # Extrusion lengths
    if length_mm is None:
        r0 = np.linalg.norm(ring0_xyz - np.asarray(caps[0].center, float)[None, :], axis=1)
        r1 = np.linalg.norm(ring1_xyz - np.asarray(caps[1].center, float)[None, :], axis=1)
        L0 = float(np.clip(factor * float(np.median(r0)), min_len, max_len))
        L1 = float(np.clip(factor * float(np.median(r1)), min_len, max_len))
    else:
        L0 = L1 = float(length_mm)

    # Start building extended vertex buffer
    V_ext = V.copy()

    # ---- End 0 ----
    ring2_0_ids = np.arange(V_ext.shape[0], V_ext.shape[0] + ring0_ids.size, dtype=np.int32)
    V_ext = np.vstack([V_ext, ring0_xyz + L0 * n0[None, :]])
    F_side0 = _triangulate_between_rings(ring0_ids.astype(np.int32), ring2_0_ids, V_ext, normal=n0)

    cap0_unique = np.unique(F[np.asarray(caps[0].face_indices, dtype=np.int64)].ravel())
    cap0_trans_ids = np.arange(V_ext.shape[0], V_ext.shape[0] + cap0_unique.size, dtype=np.int32)
    V_ext = np.vstack([V_ext, V[cap0_unique] + L0 * n0[None, :]])
    map0 = {int(old): int(new) for old, new in zip(cap0_unique, cap0_trans_ids)}
    F_cap_far0 = np.asarray([[map0[int(a)], map0[int(b)], map0[int(c)]]
                             for (a, b, c) in F[np.asarray(caps[0].face_indices, dtype=np.int64)]], dtype=np.int32)

    # ---- End 1 ----
    ring2_1_ids = np.arange(V_ext.shape[0], V_ext.shape[0] + ring1_ids.size, dtype=np.int32)
    V_ext = np.vstack([V_ext, ring1_xyz + L1 * n1[None, :]])
    F_side1 = _triangulate_between_rings(ring1_ids.astype(np.int32), ring2_1_ids, V_ext, normal=n1)

    cap1_unique = np.unique(F[np.asarray(caps[1].face_indices, dtype=np.int64)].ravel())
    cap1_trans_ids = np.arange(V_ext.shape[0], V_ext.shape[0] + cap1_unique.size, dtype=np.int32)
    V_ext = np.vstack([V_ext, V[cap1_unique] + L1 * n1[None, :]])
    map1 = {int(old): int(new) for old, new in zip(cap1_unique, cap1_trans_ids)}
    F_cap_far1 = np.asarray([[map1[int(a)], map1[int(b)], map1[int(c)]]
                             for (a, b, c) in F[np.asarray(caps[1].face_indices, dtype=np.int64)]], dtype=np.int32)

    # Remove BOTH original caps in one go
    keep = np.ones(len(F), dtype=bool)
    keep[np.asarray(caps[0].face_indices, dtype=np.int64)] = False
    keep[np.asarray(caps[1].face_indices, dtype=np.int64)] = False

    F_new = np.vstack([F[keep], F_side0, F_cap_far0, F_side1, F_cap_far1]).astype(np.int32)
    return Mesh3D(vertices=V_ext, faces=F_new, units=mesh.units)
