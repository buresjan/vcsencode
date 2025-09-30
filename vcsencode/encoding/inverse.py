"""
Inverse VCS: reconstruct a surface mesh from (centerline spline, RMF, rho surface).

We evaluate
    x(tau,theta) = c(tau) + rho_w(tau,theta) * [ v1(tau) cos(theta) + v2(tau) sin(theta) ]
on a uniform tensor grid (tau, theta), triangulate with a periodic seam in theta,
and add planar caps at tau=0 and tau=1.

Outputs are Mesh3D in millimeters.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Set
import numpy as np
import trimesh

from ..models import VCSModel, Mesh3D
from ..geom.frames import compute_rmf
from ..io import save_stl as _save_stl


def _concat_meshes(meshes: list[Mesh3D]) -> Mesh3D:
    verts = []
    faces = []
    off = 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int32)
        verts.append(v)
        faces.append(f + off)
        off += v.shape[0]
    V = np.vstack(verts)
    F = np.vstack(faces)
    return Mesh3D(vertices=V, faces=F, units="mm")


def _grid_vertices(model: VCSModel, n_tau: int, n_theta: int) -> np.ndarray:
    """Evaluate vertices on an (n_tau x n_theta) param grid (no seam duplication)."""
    cl = model.centerline
    rs = model.radius

    # Parameter grids
    # Use a tiny interior margin for the first/last rows if available
    tau_margin = float(model.meta.get("tau_margin", 0.0))
    if tau_margin > 0:
        T = np.linspace(tau_margin, 1.0 - tau_margin, int(n_tau), dtype=float)
    else:
        T = np.linspace(0.0, 1.0, int(n_tau), dtype=float)
    TH = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=float)

    # Frame evaluated along T
    step_mm = max(cl.length() / max(n_tau - 1, 1), 1e-3)
    rmf = compute_rmf(cl, step_mm=step_mm)
    C = cl.eval(T)               # (n_tau, 3)
    V1 = rmf.v1(T)               # (n_tau, 3)
    V2 = rmf.v2(T)               # (n_tau, 3)

    # Radius surface on grid
    RHO = rs.eval_grid(T, TH)    # (n_tau, n_theta)

    # Build vertices
    V = np.empty((len(T), len(TH), 3), dtype=float)
    cosTH = np.cos(TH)[:, None]  # (n_theta, 1)
    sinTH = np.sin(TH)[:, None]

    for i in range(len(T)):
        dir_ij = cosTH @ V1[i:i+1, :] + sinTH @ V2[i:i+1, :]    # (n_theta, 3)
        V[i, :, :] = C[i][None, :] + RHO[i, :, None] * dir_ij   # (n_theta, 3)

    return V  # shape (n_tau, n_theta, 3)


def _tube_faces(n_tau: int, n_theta: int) -> np.ndarray:
    """Two triangles per quad with periodic wrap in theta; no caps."""
    faces = []
    def idx(i, j): return i * n_theta + j
    for i in range(n_tau - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j2)
            d = idx(i, j2)
            # (a,b,c) and (a,c,d) produce consistent outward normals for standard tubes
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.asarray(faces, dtype=np.int32)


def surface_mesh(model: VCSModel, n_tau: int = 200, n_theta: int = 256) -> Mesh3D:
    """
    Reconstruct the wall surface (without caps) on a regular grid.

    Args
    ----
    model : VCSModel
    n_tau : longitudinal samples (>= 2)
    n_theta : angular samples (>= 3)

    Returns
    -------
    Mesh3D of the wall surface (open at both ends).
    """
    n_tau = int(max(n_tau, 2))
    n_theta = int(max(n_theta, 3))
    V = _grid_vertices(model, n_tau, n_theta)               # (n_tau, n_theta, 3)
    F = _tube_faces(n_tau, n_theta)                         # (2*(n_tau-1)*n_theta, 3)
    Vflat = V.reshape(-1, 3)
    return Mesh3D(vertices=Vflat, faces=F, units="mm")


def _cap_fan(center: np.ndarray, ring: np.ndarray, desired_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a ring with a center point using a fan. Ensure orientation so that
    triangle normals point along desired_normal on average.
    """
    n = ring.shape[0]
    verts = np.vstack([ring, center[None, :]])  # last vertex is center
    center_idx = n
    faces = []
    for j in range(n):
        j2 = (j + 1) % n
        faces.append([center_idx, j, j2])
    F = np.asarray(faces, dtype=np.int32)

    # Check orientation using the first triangle; flip if needed
    v0, v1, v2 = verts[F[0, 0]], verts[F[0, 1]], verts[F[0, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    # Robust dot that accepts any array-like
    nrm_unit = nrm.reshape(3) / (np.linalg.norm(nrm) + 1e-12)
    dn_vec = np.asarray(desired_normal, dtype=float).reshape(-1)
    if dn_vec.size == 0:
        dn_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    elif dn_vec.size == 1:
        dn_vec = float(dn_vec[0]) * nrm_unit
    elif dn_vec.size >= 3:
        dn_vec = dn_vec[:3]
    dn_vec = dn_vec / (np.linalg.norm(dn_vec) + 1e-12)
    dn = float(np.dot(nrm_unit, dn_vec))
    if dn < 0.0:
        # flip all triangles (swap last two indices)
        F[:, [1, 2]] = F[:, [2, 1]]
    return verts, F

# -------------------- Autoslice capping helpers --------------------
def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], float)
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], float)
    u = a - np.dot(a, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v, n


def _poly_area_signed_2d(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _point_in_tri_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _triangulate_polygon_ear(P: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation for a CCW polygon P (N,2)."""
    n = P.shape[0]
    idx: List[int] = list(range(n))
    tris: List[Tuple[int, int, int]] = []
    if _poly_area_signed_2d(P) < 0:
        idx.reverse()
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        ear_found = False
        m = len(idx)
        for k in range(m):
            i0 = idx[(k - 1) % m]
            i1 = idx[k]
            i2 = idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if _poly_area_signed_2d(np.vstack([a, b, c])) <= 1e-14:
                continue
            any_inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri_2d(P[j], a, b, c):
                    any_inside = True
                    break
            if any_inside:
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            ear_found = True
            break
        if not ear_found:
            root = idx[0]
            for q in range(1, len(idx) - 1):
                tris.append((root, idx[q], idx[q + 1]))
            idx = [root, idx[-1], idx[-2]]
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return np.asarray(tris, dtype=np.int32)


def _boundary_loops(vertices: np.ndarray, faces: np.ndarray) -> List[np.ndarray]:
    """Extract open boundary loops from a triangle mesh."""
    edges_count: Dict[Tuple[int, int], int] = {}
    faces_int = np.asarray(faces, dtype=np.int64)
    for tri in faces_int:
        a, b, c = map(int, tri)
        for i, j in ((a, b), (b, c), (c, a)):
            key = (i, j) if i < j else (j, i)
            edges_count[key] = edges_count.get(key, 0) + 1
    boundary_edges = [edge for edge, cnt in edges_count.items() if cnt == 1]
    if not boundary_edges:
        return []

    adj: Dict[int, List[int]] = {}
    edge_set: Set[Tuple[int, int]] = set(boundary_edges)
    for i, j in boundary_edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    loops: List[np.ndarray] = []
    used_edges: Set[Tuple[int, int]] = set()
    for i, j in boundary_edges:
        if (i, j) in used_edges or (j, i) in used_edges:
            continue
        loop = [i, j]
        used_edges.add((i, j))
        used_edges.add((j, i))
        prev, curr = i, j
        guard = 0
        while guard < 100000:
            guard += 1
            neighbors = adj.get(curr, [])
            nxt = None
            for nb in neighbors:
                if nb == prev:
                    continue
                key = (curr, nb) if curr < nb else (nb, curr)
                if key not in edge_set:
                    continue
                if key in used_edges or (key[1], key[0]) in used_edges:
                    continue
                nxt = nb
                used_edges.add(key)
                used_edges.add((key[1], key[0]))
                break
            if nxt is None:
                break
            loop.append(nxt)
            prev, curr = curr, nxt
            if curr == loop[0]:
                break
        if loop[0] == loop[-1]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(vertices[np.asarray(loop, dtype=int)])
    return loops


def _cap_from_wall_autoslice(wall: Mesh3D, center: np.ndarray, normal: np.ndarray) -> Mesh3D:
    tm = trimesh.Trimesh(vertices=np.asarray(wall.vertices, float),
                         faces=np.asarray(wall.faces, np.int64),
                         process=False)
    ring = None
    try:
        sec = tm.section(plane_origin=np.asarray(center, float),
                         plane_normal=np.asarray(normal, float))
        if sec is not None and len(sec.entities) > 0:
            loops = sec.discrete
            if loops:
                lengths = [float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()) for loop in loops]
                ring = loops[int(np.argmax(lengths))]
                if np.linalg.norm(ring[0] - ring[-1]) < 1e-9:
                    ring = ring[:-1]
    except Exception:
        ring = None

    if ring is None:
        loops_xyz = _boundary_loops(np.asarray(wall.vertices, float), np.asarray(wall.faces, np.int64))
        if not loops_xyz:
            return Mesh3D(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32), units=wall.units)
        C = np.asarray(center, float).reshape(1, 3)
        dists = [float(np.linalg.norm(np.mean(loop, axis=0) - C)) for loop in loops_xyz]
        ring = loops_xyz[int(np.argmin(dists))]
    u, v, n = _plane_basis(normal)
    Q = ring - np.asarray(center, float)[None, :]
    P2 = np.column_stack([Q @ u, Q @ v])
    F2 = _triangulate_polygon_ear(P2)
    Vcap = ring.copy()
    Fcap = F2.astype(np.int32)
    if Fcap.shape[0] > 0:
        a, b, c = Vcap[Fcap[0]]
        nrm = np.cross(b - a, c - a)
        if float(np.dot(nrm, n)) < 0.0:
            Fcap[:, [1, 2]] = Fcap[:, [2, 1]]
    return Mesh3D(vertices=Vcap, faces=Fcap, units=wall.units)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (u, v, n) given a plane normal n."""
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    # pick an arbitrary vector least aligned with n
    a = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=float)
    u = a - np.dot(a, n) * n
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v, n


def cap_meshes(model: VCSModel, n_theta: int = 256) -> Mesh3D:
    """
    Generate planar end caps at tau=0 and tau=1 via triangle fans.

    The normal targets are:
      - tau=0 cap normal ~ -t(0)  (outward)
      - tau=1 cap normal ~ +t(1)  (outward)
    """
    cl = model.centerline
    rs = model.radius

    # Prefer stored cap planes and centers from forward pass
    planes = model.meta.get("cap_planes", None)
    if planes is None:
        # Fallback to tangent-based planes
        rmf = compute_rmf(cl, step_mm=max(cl.length() / 100.0, 1e-3))
        n0 = -rmf.t(0.0).reshape(3)
        n1 = rmf.t(1.0).reshape(3)
        c0 = cl.eval(0.0).reshape(3)
        c1 = cl.eval(1.0).reshape(3)
    else:
        n0 = np.asarray(planes[0]["normal"], dtype=float).reshape(3)
        n1 = np.asarray(planes[1]["normal"], dtype=float).reshape(3)
        c0 = np.asarray(planes[0]["center"], dtype=float).reshape(3)
        c1 = np.asarray(planes[1]["center"], dtype=float).reshape(3)

    TH = np.linspace(0.0, 2.0*np.pi, int(max(3, n_theta)), endpoint=False)
    # Use margin rings for stability near the ends
    tau_margin = float(model.meta.get("tau_margin", 0.0))
    t0 = max(0.0, min(1.0, tau_margin if tau_margin > 0 else 1e-3))
    t1 = max(0.0, min(1.0, 1.0 - (tau_margin if tau_margin > 0 else 1e-3)))
    rho0 = rs.eval_grid(np.array([t0]), TH)[0, :]
    rho1 = rs.eval_grid(np.array([t1]), TH)[0, :]

    # Build cap rings in the exact cap planes
    u0, v0, n0 = _plane_basis(n0)
    u1, v1, n1 = _plane_basis(n1)
    ring0 = c0[None, :] + rho0[:, None]*(np.cos(TH)[:, None]*u0[None, :] + np.sin(TH)[:, None]*v0[None, :])
    ring1 = c1[None, :] + rho1[:, None]*(np.cos(TH)[:, None]*u1[None, :] + np.sin(TH)[:, None]*v1[None, :])

    V0, F0 = _cap_fan(center=c0, ring=ring0, desired_normal=n0)
    V1, F1 = _cap_fan(center=c1, ring=ring1, desired_normal=n1)

    # Stitch into one mesh (offset faces of second cap)
    V = np.vstack([V0, V1])
    F1_off = F1 + V0.shape[0]
    F = np.vstack([F0, F1_off]).astype(np.int32)
    return Mesh3D(vertices=V, faces=F, units="mm")


def cap_meshes_autoslice(model: VCSModel, wall: Mesh3D) -> Mesh3D:
    planes = model.meta.get("cap_planes", None)
    if not planes:
        return cap_meshes(model, n_theta=256)
    c0 = np.asarray(planes[0]["center"], float)
    n0 = np.asarray(planes[0]["normal"], float)
    c1 = np.asarray(planes[1]["center"], float)
    n1 = np.asarray(planes[1]["normal"], float)
    cap0 = _cap_from_wall_autoslice(wall, c0, n0)
    cap1 = _cap_from_wall_autoslice(wall, c1, n1)
    parts = []
    if cap0.vertices.size:
        parts.append(cap0)
    if cap1.vertices.size:
        if parts:
            off = parts[0].vertices.shape[0]
            V_all = np.vstack([parts[0].vertices, cap1.vertices])
            F_all = np.vstack([parts[0].faces, cap1.faces + off]).astype(np.int32)
            return Mesh3D(vertices=V_all, faces=F_all, units=wall.units)
        else:
            return cap1
    return cap0


def export_stl(model: VCSModel, path: str, n_tau: int = 200, n_theta: int = 256, cap_mode: str = "autoslice") -> None:
    """
    Compose wall+caps and save STL to `path`.
    """
    wall = surface_mesh(model, n_tau=n_tau, n_theta=n_theta)
    if cap_mode == "autoslice":
        caps = cap_meshes_autoslice(model, wall)
        if caps.vertices.size == 0:
            caps = cap_meshes(model, n_theta=n_theta)
    else:
        caps = cap_meshes(model, n_theta=n_theta)
    full = _concat_meshes([wall, caps])

    # Light welding/cleanup via trimesh before writing
    tm = trimesh.Trimesh(vertices=full.vertices, faces=full.faces, process=False)
    try:
        tm.remove_unreferenced_vertices()
        tm.merge_vertices()
    except Exception:
        pass

    _save_stl(Mesh3D(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces, dtype=np.int32), units="mm"), path)
