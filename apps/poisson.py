"""Poisson PDE FEM matrix generator for NeuralIF (paper §4.2 / Appendix A.2).

Generates 2D Poisson stiffness matrices on three families of randomly
sampled domains (convex, convex-with-hole, simple polytope), discretized
with P1 triangular finite elements via scikit-fem. The boundary DOFs are
removed (homogeneous Dirichlet u = 0) so the resulting matrix is SPD on
the interior, ready for the (preconditioned) conjugate gradient method.

Defaults not pinned down by the paper (Appendix A.2 is sparse on details):
    - Source function f(x) = 1 (constant load)
    - Boundary u_D = 0 (homogeneous Dirichlet)
    - Mesh base points drawn from N(0, 1) in 2D
    - Inner-hole variance: sigma_in = 0.3
    - Refinement: uniform (mesh.refined(k)), k chosen to hit the target size

Output format matches apps/synthetic.py: one PyG Data per .pt file under
./data/Poisson/{train,val,test}/, named "{n_actual}_{idx}.pt".
"""
import math
import time
from pathlib import Path

import numpy as np
import skfem
import torch
from matplotlib.path import Path as MplPath
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg as sp_cg
from scipy.spatial import ConvexHull, Delaunay
from skfem.helpers import dot, grad

from data import matrix_to_graph


# ── FEM forms ───────────────────────────────────────────────────────────────
@skfem.BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


@skfem.LinearForm
def load_one(v, _):
    return 1.0 * v


# ── Mesh shape generators ──────────────────────────────────────────────────
def make_convex(rng, n=100):
    pts = rng.standard_normal((n, 2))
    tri = Delaunay(pts).simplices
    return pts, tri


def make_hole(rng, n_out=80, n_in=40, sigma_in=0.3):
    outer = rng.standard_normal((n_out, 2))
    inner = sigma_in * rng.standard_normal((n_in, 2))
    inner_boundary = inner[ConvexHull(inner).vertices]

    pts = np.vstack([outer, inner_boundary])
    tri = Delaunay(pts).simplices

    centroids = pts[tri].mean(axis=1)
    inside_hole = MplPath(inner_boundary).contains_points(centroids)
    tri = tri[~inside_hole]
    return _compact(pts, tri)


def make_polytope(rng, n_boundary=25, n_interior=80, sigma_interior=0.6):
    bnd = rng.standard_normal((n_boundary, 2))
    centroid = bnd.mean(axis=0)
    angles = np.arctan2(bnd[:, 1] - centroid[1], bnd[:, 0] - centroid[0])
    bnd = bnd[np.argsort(angles)]

    polygon = MplPath(bnd)
    cands = sigma_interior * rng.standard_normal((n_interior * 3, 2))
    inside = cands[polygon.contains_points(cands)][:n_interior]

    pts = np.vstack([bnd, inside])
    tri = Delaunay(pts).simplices
    centroids = pts[tri].mean(axis=1)
    tri = tri[polygon.contains_points(centroids)]
    return _compact(pts, tri)


def _compact(pts, tri):
    used = np.unique(tri.ravel())
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return pts[used], remap[tri]


# ── FEM assembly ───────────────────────────────────────────────────────────
def assemble_poisson(mesh):
    basis = skfem.CellBasis(mesh, skfem.ElementTriP1())
    K = laplace.assemble(basis)
    f = load_one.assemble(basis)

    boundary = basis.get_dofs().flatten()
    interior_mask = np.ones(K.shape[0], dtype=bool)
    interior_mask[boundary] = False
    interior = np.where(interior_mask)[0]

    A = K.tocsr()[interior][:, interior].tocsr()
    b = f[interior]
    return A, b


# ── Size targeting ─────────────────────────────────────────────────────────
def estimate_refinements(base_n_vertices, target_n):
    if base_n_vertices >= target_n:
        return 0
    return max(0, round(math.log(target_n / base_n_vertices) / math.log(4)))


def generate_one(target_n, rng, mesh_type):
    if mesh_type == 'convex':
        pts, tri = make_convex(rng)
    elif mesh_type == 'hole':
        pts, tri = make_hole(rng)
    elif mesh_type == 'polytope':
        pts, tri = make_polytope(rng)
    else:
        raise ValueError(f"unknown mesh_type: {mesh_type}")

    mesh = skfem.MeshTri(pts.T, tri.T.astype(np.int32))
    k = estimate_refinements(pts.shape[0], target_n)
    if k > 0:
        mesh = mesh.refined(k)

    A, b = assemble_poisson(mesh)
    return A, b, A.shape[0]


# ── Verification ───────────────────────────────────────────────────────────
def verify(A, b, target_min, target_max):
    n = A.shape[0]
    if A.shape[0] != A.shape[1] or b.shape[0] != n:
        return False, f"shape mismatch (A={A.shape}, b={b.shape})"
    if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(b))):
        return False, "NaN/Inf in A or b"
    if not (target_min <= n <= target_max):
        return False, f"size {n} out of [{target_min}, {target_max}]"

    a_max = abs(A).max()
    sym_err = abs(A - A.T).max() / max(a_max, 1e-300)
    if sym_err >= 1e-10:
        return False, f"asymmetric (rel err {sym_err:.2e})"

    diag = A.diagonal()
    if not np.all(diag > 0):
        return False, f"non-positive diagonal (min {diag.min():.3e})"

    n_comp = connected_components(A, directed=False, return_labels=False)
    if n_comp != 1:
        return False, f"disconnected ({n_comp} components)"

    rng = np.random.default_rng(0)
    rb = rng.standard_normal(n)
    _, info = sp_cg(A, rb, rtol=1e-6, maxiter=10_000)
    if info != 0:
        return False, f"CG did not converge (info={info})"

    return True, "ok"


# ── Driver ─────────────────────────────────────────────────────────────────
MESH_TYPES = ('convex', 'hole', 'polytope')
MAX_RETRIES = 10


def create_dataset(samples, target_n_min, target_n_max, mode, rs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{mode}] generating {samples} samples (n in [{target_n_min}, {target_n_max}]) -> {out_dir}",
          flush=True)

    n_ok = 0
    n_skip = 0
    n_exists = 0
    for sam in range(samples):
        if any(out_dir.glob(f"*_{sam}.pt")):
            n_exists += 1
            continue
        t0 = time.time()
        ok = False
        for attempt in range(MAX_RETRIES):
            rng = np.random.RandomState(rs + sam * 100 + attempt)
            mesh_type = MESH_TYPES[rng.randint(len(MESH_TYPES))]
            target_n = rng.randint(target_n_min, target_n_max + 1)
            try:
                A, b, n_actual = generate_one(target_n, rng, mesh_type)
                ok, reason = verify(A, b, target_n_min, target_n_max)
            except Exception as e:
                ok = False
                reason = f"exception {type(e).__name__}: {e}"
            if ok:
                break
            print(f"[{mode}/{sam:4d}] attempt {attempt} failed: {reason} "
                  f"(mesh={mesh_type}, target={target_n})", flush=True)

        if not ok:
            n_skip += 1
            print(f"[skip] {mode}/{sam:4d} failed all {MAX_RETRIES} retries", flush=True)
            continue

        graph = matrix_to_graph(A, b)
        graph.n = n_actual
        torch.save(graph, out_dir / f"{n_actual}_{sam}.pt")
        n_ok += 1
        dt = time.time() - t0
        print(f"[{mode}/{sam:4d}] ok n={n_actual} mesh={mesh_type} ({dt:.2f}s)", flush=True)

    print(f"[{mode}] done: {n_ok} saved, {n_skip} skipped, {n_exists} preexisting",
          flush=True)


if __name__ == '__main__':
    create_dataset(samples=750, target_n_min=20_000,  target_n_max=150_000,
                   mode='train', rs=0,          out_dir='./data/Poisson/train')
    create_dataset(samples=15,  target_n_min=20_000,  target_n_max=150_000,
                   mode='val',   rs=10_000_000, out_dir='./data/Poisson/val')
    create_dataset(samples=300, target_n_min=100_000, target_n_max=500_000,
                   mode='test',  rs=20_000_000, out_dir='./data/Poisson/test')
