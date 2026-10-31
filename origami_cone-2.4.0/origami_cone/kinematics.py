"""Exact rigid-origami kinematics: the consistency constraint and its solution.

For each interior vertex v with creases taken counter-clockwise and outward
unit axes u_1..u_n, the folded sectors close up iff

    Phi_v(rho) = prod_{k=1..n} R(u_k, rho_k) = I .

The fold angle rho_e belongs to the *crease*, not to the vertex: at both of
its endpoints one uses the outward axis and the same rho_e, and the sign then
works out automatically (if Q_L = Q_R R(u, rho) seen from one end, then
Q_R = Q_L R(-u, rho) seen from the other).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

_IU = np.triu_indices(3)


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rodrigues(u, t):
    U = hat(np.array([u[0], u[1], 0.0]) if len(u) == 2 else np.asarray(u, float))
    return np.eye(3) + np.sin(t) * U + (1.0 - np.cos(t)) * (U @ U)


def axis3(u):
    return np.array([u[0], u[1], 0.0])


def vertex_closure(P, k: int, rho) -> np.ndarray:
    M = np.eye(3)
    for _, e, u in P.vdata[k]:
        M = M @ rodrigues(axis3(u), rho[e])
    return M


def residual(P, rho) -> np.ndarray:
    """Six independent components of Phi_v - I per interior vertex."""
    return np.concatenate([(vertex_closure(P, k, rho) - np.eye(3))[_IU]
                           for k in range(P.V)])


def jacobian(P, rho) -> np.ndarray:
    """Analytic derivative of `residual`."""
    J = np.zeros((6 * P.V, P.E))
    for k in range(P.V):
        items = P.vdata[k]
        Rs = [rodrigues(axis3(u), rho[e]) for _, e, u in items]
        n = len(items)
        pre = [np.eye(3)] * (n + 1)
        post = [np.eye(3)] * (n + 1)
        for m in range(n):
            pre[m + 1] = pre[m] @ Rs[m]
        for m in range(n - 1, -1, -1):
            post[m] = Rs[m] @ post[m + 1]
        for m, (_, e, u) in enumerate(items):
            D = pre[m] @ (hat(axis3(u)) @ Rs[m]) @ post[m + 1]
            J[6 * k:6 * k + 6, e] += D[_IU]
    return J


def local_dof(P, rho, tol: float = 1e-6) -> int:
    """Dimension of the configuration space at rho (rank of the full 9-component
    Jacobian of Phi - I subtracted from the number of creases)."""
    Jf = np.zeros((9 * P.V, P.E))
    h = 1e-7
    for e in range(P.E):
        d = np.zeros(P.E)
        d[e] = h
        up = np.concatenate([vertex_closure(P, k, rho + d).ravel() for k in range(P.V)])
        dn = np.concatenate([vertex_closure(P, k, rho - d).ravel() for k in range(P.V)])
        Jf[:, e] = (up - dn) / (2 * h)
    sv = np.linalg.svd(Jf, compute_uv=False)
    return P.E - int((sv > tol).sum())


def solve_slice(P, v, eps: float, x0=None):
    """Nearest folded state on the affine slice <rho, v> = eps."""
    v = np.asarray(v, float)
    n = P.E
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, float)
    f = lambda r: np.concatenate([residual(P, r), [r @ v - eps]])
    Jf = lambda r: np.vstack([jacobian(P, r), v[None, :]])
    s = least_squares(f, x0, jac=Jf, bounds=(-np.pi * np.ones(n), np.pi * np.ones(n)),
                      xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return s.x


def continue_branch(P, v, target: float = 0.8, steps: int = 36):
    """Predictor-corrector continuation from the flat state along direction v.
    Returns (rho, residual_norm)."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    x = np.zeros(P.E)
    for eps in np.linspace(target / steps, target, steps):
        x = solve_slice(P, v, eps, x)
    return x, float(np.linalg.norm(residual(P, x)))


def find_folded_states(P, tries: int = 400, min_norm: float = 0.1,
                       tol: float = 1e-10, seed: int = 0):
    """Random multistart solve of Phi_v = I for all v. Used to certify that a
    pattern really is (or is not) rigidly foldable away from the flat state."""
    rng = np.random.default_rng(seed)
    out = []
    n = P.E
    for _ in range(tries):
        x0 = rng.uniform(-0.95 * np.pi, 0.95 * np.pi, n)
        s = least_squares(lambda r: residual(P, r), x0, jac=lambda r: jacobian(P, r),
                          bounds=(-np.pi * np.ones(n), np.pi * np.ones(n)),
                          xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(residual(P, s.x)) < tol and np.linalg.norm(s.x) > min_norm:
            out.append(s.x.copy())
    return out


def certify_cells(P, res, eps=(0.1, 0.05), tol: float = 1e-12, seed: int = 0):
    """Decide, cell by cell, whether a tangent-cone component is a real branch.

    `decompose` solves the SECOND-ORDER condition, which is necessary but not
    sufficient: `patterns.second_order_impostor` has a nonempty tangent cone and
    does not fold at all. The test is to correct a cone direction onto the exact
    constraint at two step sizes. A genuine branch holds its residual at machine
    precision; a second-order impostor cannot get below O(eps^4), which shows up
    as a residual well above `tol` falling by roughly 16 per halving.

    Returns a list of dicts, one per cell, with keys
    dim, kind, residuals, genuine.
    """
    from .cone import cell_directions
    out = []
    for i, c in enumerate(res["cells"]):
        v = cell_directions(res, c, seed=seed + i)[0]
        r = []
        for e in eps:
            y = solve_slice(P, v, e, e * np.asarray(v))
            r.append(float(np.linalg.norm(residual(P, y))))
        out.append(dict(dim=c.dim, kind=c.kind, residuals=r, genuine=bool(r[0] < tol)))
    return out


def certified_branch_count(P, res=None, **kw) -> int:
    """Number of tangent-cone components that survive `certify_cells`."""
    from .cone import decompose
    res = decompose(P) if res is None else res
    return sum(1 for c in certify_cells(P, res, **kw) if c["genuine"])
