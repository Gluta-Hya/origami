"""Tangent cone of the rigid-origami configuration space at the flat state.

Expanding the per-vertex consistency constraint around rho = 0 gives, with
u_k the in-plane crease axes and [hat u_e, hat u_f] = sin(alpha_f - alpha_e) hat z,

    log Phi_v(rho) = ( sum_e rho_e cos alpha_e ,
                       sum_e rho_e sin alpha_e ,
                       rho^T Q_v rho              ) + O(rho^3),
    (Q_v)_ef = (1/4) sin(alpha_f - alpha_e)  for e != f in star(v).

So the in-plane part is linear (two equations per vertex, hence the flat state
is always a singular point) and the out-of-plane part starts at second order,
with the commutators as its coefficients. The tangent cone is

    C = { rho : J0 rho = 0 } ∩ ∩_v { rho^T Q_v rho = 0 }.

A vertex of degree 4 has rank(Q_v|_K) <= 2, and a rank-2 indefinite form
factors into two real linear forms, so such a vertex contributes a pair of
hyperplanes rather than a curved quadric. Vertices of degree >= 5 generally
contribute an irreducible quadric.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Cell:
    dim: int                 # dimension of this component of the tangent cone
    basis: np.ndarray        # orthonormal basis of the ambient linear cell (K coords)
    choice: tuple            # which factor was taken at each factorable vertex
    kind: str                # "flat" or "curved"
    points: np.ndarray = None    # sampled unit points of the cone, in cell coords


def first_order_matrix(P) -> np.ndarray:
    """J0, of shape (2V, E). Row pair per interior vertex."""
    rows = []
    for items in P.vdata:
        rc = np.zeros(P.E)
        rs = np.zeros(P.E)
        for _, e, u in items:
            rc[e] += u[0]
            rs[e] += u[1]
        rows += [rc, rs]
    return np.array(rows)


def hessian_blocks(P) -> list:
    """One symmetric E x E block per interior vertex, supported on its star."""
    out = []
    for items in P.vdata:
        Q = np.zeros((P.E, P.E))
        for p in range(len(items)):
            for q in range(p + 1, len(items)):
                (ap, ep, _), (aq, eq, _) = items[p], items[q]
                w = 0.25 * np.sin(aq - ap)
                Q[ep, eq] += w
                Q[eq, ep] += w
        out.append(Q)
    return out


def kernel(P, tol: float = 1e-9):
    """Orthonormal basis B of ker J0 (columns), and rank J0."""
    J = first_order_matrix(P)
    _, s, Vt = np.linalg.svd(J)
    rank = int((s > tol).sum())
    return Vt[rank:].T, rank


def classify_block(Qt: np.ndarray, tol: float = 1e-9):
    """Classify a restricted Hessian block.

    Returns (kind, payload):
      "zero"     -> no condition at all
      "linear"   -> [f1] (double hyperplane) or [f1, f2] (two hyperplanes)
      "definite" -> None; the only real solution is the origin (vertex is rigid)
      "quadric"  -> Qt; irreducible, the cone is genuinely curved
    """
    ev, V = np.linalg.eigh(Qt)
    keep = np.abs(ev) > tol
    r = int(keep.sum())
    if r == 0:
        return "zero", None
    if r == 1:
        return "linear", [V[:, int(np.argmax(np.abs(ev)))]]
    if r == 2:
        ip, im = int(np.argmax(ev)), int(np.argmin(ev))
        if ev[ip] <= tol or ev[im] >= -tol:
            return "definite", None
        a = np.sqrt(ev[ip]) * V[:, ip] - np.sqrt(-ev[im]) * V[:, im]
        b = np.sqrt(ev[ip]) * V[:, ip] + np.sqrt(-ev[im]) * V[:, im]
        return "linear", [a, b]
    if (ev > tol).sum() == 0 or (ev < -tol).sum() == 0:
        return "definite", None
    return "quadric", Qt


def signature(Qt: np.ndarray, tol: float = 1e-9):
    ev = np.linalg.eigvalsh(Qt)
    return int((ev > tol).sum()), int((ev < -tol).sum()), int((np.abs(ev) <= tol).sum())


def decompose(P, tol_rank: float = 1e-9, tol_nest: float = 1e-7,
              quadric_samples: int = 160, seed: int = 7, maximal_only: bool = True):
    """Full tangent-cone decomposition.

    Returns dict with keys:
      B, rank_J0, dim_K, blocks (classification per vertex), cells.
    """
    B, rank = kernel(P, tol_rank)
    k = B.shape[1]
    blocks = [classify_block(B.T @ Q @ B, tol_rank) for Q in hessian_blocks(P)]
    if any(kind == "definite" for kind, _ in blocks):
        return dict(B=B, rank_J0=rank, dim_K=k, blocks=blocks, cells=[],
                    note="a vertex is second-order rigid: tangent cone = {0}")
    lin = [pay for kind, pay in blocks if kind == "linear"]
    quads = [pay for kind, pay in blocks if kind == "quadric"]

    raw = []
    for choice in itertools.product(*[range(len(f)) for f in lin]):
        if lin:
            M = np.array([lin[t][c] for t, c in enumerate(choice)])
            _, s, Vt = np.linalg.svd(M)
            r = int((s > 1e-8).sum())
            N = Vt[r:].T
        else:
            N = np.eye(k)
        if N.shape[1] > 0:
            raw.append((N.shape[1], N, choice))

    if maximal_only:
        keep = []
        for d, N, ch in sorted(raw, key=lambda t: -t[0]):
            if not any(d <= d2 and np.linalg.norm(N - N2 @ (N2.T @ N)) < tol_nest
                       for d2, N2, _ in keep):
                keep.append((d, N, ch))
        raw = keep

    cells = []
    rng = np.random.default_rng(seed)
    for d, N, ch in raw:
        if not quads:
            cells.append(Cell(dim=d, basis=N, choice=ch, kind="flat"))
            continue
        Qr = [N.T @ Q @ N for Q in quads]
        live = [q for q in Qr if np.abs(q).max() > tol_rank]
        if not live:
            cells.append(Cell(dim=d, basis=N, choice=ch, kind="flat"))
            continue
        if len(live) == 1:
            # A single real quadratic form has a nonzero real zero iff it is
            # indefinite (or degenerate with a kernel); this is exact, so no
            # sampling is needed and no thin cell can be missed.
            p_, n_, z_ = signature(live[0], tol_rank)
            if p_ and n_:
                pts = _sample_quadric(live[0], d, quadric_samples, rng)
                cells.append(Cell(dim=d - 1, basis=N, choice=ch, kind="curved",
                                  points=pts))
            elif z_ and not (p_ and n_):
                # semidefinite: the real zero set is exactly the kernel
                ev, V = np.linalg.eigh(live[0])
                K0 = V[:, np.abs(ev) <= tol_rank]
                cells.append(Cell(dim=K0.shape[1], basis=N @ K0, choice=ch,
                                  kind="flat"))
            # definite: only the origin survives, so this cell contributes nothing
            continue
        f = lambda y: np.array([y @ q @ y for q in live] + [y @ y - 1.0])
        pts = []
        for _ in range(quadric_samples):
            y0 = rng.normal(size=d)
            y0 /= np.linalg.norm(y0)
            r = least_squares(f, y0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.linalg.norm(f(r.x)) < 1e-10:
                pts.append(r.x / np.linalg.norm(r.x))
        if pts:
            cells.append(Cell(dim=d - len(live), basis=N, choice=ch, kind="curved",
                              points=np.array(pts)))
    return dict(B=B, rank_J0=rank, dim_K=k, blocks=blocks, cells=cells, note="")


def _sample_quadric(Q, d, count, rng):
    """Unit points of the real cone {y : y^T Q y = 0} in R^d, via its
    eigenbasis; exact up to floating point, no iterative solve needed."""
    ev, V = np.linalg.eigh(Q)
    pos = [i for i in range(d) if ev[i] > 0]
    neg = [i for i in range(d) if ev[i] < 0]
    zer = [i for i in range(d) if ev[i] == 0]
    out = []
    for _ in range(count):
        y = np.zeros(d)
        cp = rng.normal(size=len(pos)); cn = rng.normal(size=len(neg))
        sp = float(np.sum(ev[pos] * cp ** 2)); sn = float(-np.sum(ev[neg] * cn ** 2))
        if sp <= 0 or sn <= 0:
            continue
        t = np.sqrt(sn / sp)
        for j, i in enumerate(pos):
            y[i] = t * cp[j]
        for j, i in enumerate(neg):
            y[i] = cn[j]
        for i in zer:
            y[i] = rng.normal() * 0.3
        nrm = np.linalg.norm(y)
        if nrm > 1e-12:
            out.append(V @ (y / nrm))
    return np.array(out) if out else None


def cell_directions(res, cell: Cell, count: int = 1, seed: int = 0):
    """Representative fold-angle directions (length E) inside a cell."""
    B = res["B"]
    rng = np.random.default_rng(seed)
    out = []
    if cell.kind == "flat":
        for _ in range(count):
            w = cell.basis @ rng.normal(size=cell.basis.shape[1])
            out.append(B @ w)
    else:
        idx = rng.choice(len(cell.points), size=min(count, len(cell.points)), replace=False)
        for i in np.atleast_1d(idx):
            out.append(B @ (cell.basis @ cell.points[i]))
    return [v / np.linalg.norm(v) for v in out]


def linear_cells(P, tol_rank: float = 1e-9, tol_nest: float = 1e-7):
    """The maximal linear cells (before any quadric cut), as (dim, basis, choice)."""
    B, _ = kernel(P, tol_rank)
    k = B.shape[1]
    blocks = [classify_block(B.T @ Q @ B, tol_rank) for Q in hessian_blocks(P)]
    lin = [pay for kind, pay in blocks if kind == "linear"]
    raw = []
    for choice in itertools.product(*[range(len(f)) for f in lin]):
        if lin:
            M = np.array([lin[t][c] for t, c in enumerate(choice)])
            _, s, Vt = np.linalg.svd(M)
            r = int((s > 1e-8).sum())
            N = Vt[r:].T
        else:
            N = np.eye(k)
        if N.shape[1] > 0:
            raw.append((N.shape[1], N, choice))
    keep = []
    for d, N, ch in sorted(raw, key=lambda t: -t[0]):
        if not any(d <= d2 and np.linalg.norm(N - N2 @ (N2.T @ N)) < tol_nest
                   for d2, N2, _ in keep):
            keep.append((d, N, ch))
    return keep, blocks


def definiteness_count(P, tol_rank: float = 1e-9, tol_nest: float = 1e-7):
    """Mechanism behind the branch count of a single-implant mixed-degree pattern.

    Established in this project and checked on 486/486 configurations: with
    exactly one vertex contributing an irreducible quadric Q_v, the tangent cone
    has one component per maximal linear cell L_c on which Q_v restricts to a
    form that is NOT definite. A definite restriction leaves only the origin, so
    that cell contributes nothing:

        #components = #{ c : Q_v|_{L_c} is indefinite or degenerate }

    Position enters the count only through those signatures, so the walls in
    position space are exactly the loci where one restricted form becomes
    degenerate. Returns (predicted count, per-cell list of (dim, signature)).
    """
    keep, blocks = linear_cells(P, tol_rank, tol_nest)
    quads = [pay for kind, pay in blocks if kind == "quadric"]
    if len(quads) != 1:
        return None, None
    rows, n = [], 0
    for d, N, ch in keep:
        Q = N.T @ quads[0] @ N
        p, m, z = signature(Q, tol_rank)
        dead = np.abs(Q).max() > tol_rank and z == 0 and (p == 0 or m == 0)
        rows.append((d, (p, m, z), not dead))
        n += int(not dead)
    return n, rows


# Placement classes for one implanted vertex on a flat-foldable quad mesh whose
# interior vertices have degree 4. Sectors are numbered CCW from the vertex's
# first crease: at a Miura vertex leaning to the right, sectors 1 and 2 are the
# wide pair and sectors 0 and 3 the narrow pair. The closed forms below are for
# bisecting stubs; `criterion.predict_N` handles arbitrary stub positions.
_WIDE_PAIR = {(1, 2)}
_NARROW_PAIR = {(0, 3)}
_MIXED_PAIR = {(0, 1), (2, 3)}
_OPPOSITE_PAIR = {(0, 2), (1, 3)}


def predicted_components(nrows: int, ncols: int, col: int, row: int, order,
                         _allow_pair: bool = False):
    """LEGACY (kept for backward compatibility; use `predicted_at` or, for any
    stub position, `criterion.predict_N`).

    Count of tangent-cone components for ONE implanted vertex whose stubs
    BISECT their sectors, in the sector-multiset notation of the old index
    rule. Write A = 2^(col-1) and C = 2^(ncols-col); sectors 0 and 3 are taken
    to be the narrow pair. Then

        outer row (row = 1 or nrows), any placement : 2^ncols   (Theorem 7.5)
        interior row, one stub in any sector        : 2^ncols + 2
        interior row, both stubs in the wide pair   : 2^ncols + 2A
        interior row, both stubs in the narrow pair : 2^ncols + 2C
        interior row, adjacent pair (0,1) or (2,3)  : 2^ncols + A + C
        interior row, one stub per sector           : 2^ncols + 2A + 2C - 2

    The entry "two stubs stacked in one sector" of earlier versions came from a
    repeated index that produced two COINCIDENT creases; it is not a count for
    two stubs in one sector and is kept only so old scripts still run.

    The diagonal pairs (0,2) and (1,3) return None unless `_allow_pair` is set,
    in which case the two bisector values are returned (narrow side on the left,
    narrow side on the right; see `predicted_at`). Off the bisectors every
    class with an empty associated sector can change value (Corollary 7.3).
    """
    B = 2 ** ncols
    if row == 1 or row == nrows:
        return B
    A, C = 2 ** (col - 1), 2 ** (ncols - col)
    o = tuple(order)
    if len(o) == 1:
        return B + 2
    if len(o) == 2:
        p = tuple(sorted(o))
        if p[0] == p[1]:
            return B + 2 * A + 2 * C - 2          # stacked
        if p in _WIDE_PAIR:
            return B + 2 * A
        if p in _NARROW_PAIR:
            return B + 2 * C
        if p in _MIXED_PAIR:
            return B + A + C
        if p in _OPPOSITE_PAIR:
            # Bisector values with the narrow side on the left / on the right
            # (Corollary 7.3(c)); they coincide when A == C.
            return None if not _allow_pair else (B + 2 * A + C - 1,
                                                 B + A + 2 * C - 1)
    if len(o) == 4 and tuple(sorted(o)) == (0, 1, 2, 3):
        return B + 2 * A + 2 * C - 2
    return None


def opposite_pair_values(ncols: int, col: int):
    """The two BISECTOR values for a diagonal sector pair, (0,2) or (1,3), at
    an interior-row vertex in column `col`:

        narrow sectors on the LEFT   ->  2^ncols + 2A + C - 1   (first value)
        narrow sectors on the RIGHT  ->  2^ncols + A + 2C - 1   (second value)

    By Corollary 7.3(c), when both stubs bisect their sectors the family on the
    narrow side survives and the one on the wide side dies. This is a statement
    about bisecting stubs only. Away from the bisectors either family can die
    (Corollary 7.3(b)), never both (Corollary 7.4), so a diagonal pair takes the
    values |L|, |L| - a and |L| - c, all three occurring on one vertex; see
    `criterion.predict_N` for the count at arbitrary stub positions."""
    A, C = 2 ** (col - 1), 2 ** (ncols - col)
    B = 2 ** ncols
    return (B + 2 * A + C - 1, B + A + 2 * C - 1)


def incremental_linear_cells(P, tol_rank: float = 1e-9, tol_nest: float = 1e-7,
                             progress=None, prune: bool = True, stats=None):
    """The maximal linear cells, without enumerating 2^V4 hyperplane choices.

    Each degree-4 vertex contributes a rank-2 indefinite form on ker J0, which
    factors into two real hyperplanes, and a cell is one hyperplane per vertex.
    Enumerating every choice costs 2^(V-1) and is what made a 5x5 patch
    hopeless: 16.7 million choices per configuration. But the map from choice
    to SUBSPACE has enormous fibres. A 3x7 patch collapses 1,048,576 choices
    onto 256 distinct subspaces, so almost all of that work is spent
    rediscovering subspaces already found.

    So intersect one vertex at a time and keep the distinct subspaces as we go.
    Cutting a subspace S by a hyperplane {n . x = 0} is one projection: if
    N^T n is zero then S already lies in the hyperplane and is unchanged,
    otherwise the intersection is S with that one direction removed. The state
    set stays in the hundreds, so the cost is V4 times the number of distinct
    subspaces rather than 2^V4.

    Returns (cells, blocks) with cells as (dim, basis) in descending dimension,
    already filtered for maximality, matching `linear_cells` exactly.
    """
    B, _ = kernel(P, tol_rank)
    k = B.shape[1]
    blocks = [classify_block(B.T @ Q @ B, tol_rank) for Q in hessian_blocks(P)]
    lin = [pay for kind, pay in blocks if kind == "linear"]

    states = [np.eye(k)]
    for step, factors in enumerate(lin):
        nxt = []
        for N in states:
            for f in factors:
                w = N.T @ np.asarray(f, float)
                nw = float(np.linalg.norm(w))
                if nw < 1e-10:
                    M = N                      # S already inside this hyperplane
                else:
                    d = N.shape[1]
                    if d == 1:
                        continue               # cut down to the origin
                    # orthonormal complement of w inside the cell's own coords
                    Qf, _ = np.linalg.qr(np.eye(d) - np.outer(w, w) / nw ** 2)
                    _, s, Vt = np.linalg.svd(np.atleast_2d(w / nw))
                    M = N @ Vt[1:].T
                if M.shape[1] == 0:
                    continue
                if not any(M.shape[1] == O.shape[1] and
                           np.linalg.norm(M - O @ (O.T @ M)) < tol_nest for O in nxt):
                    nxt.append(M)
        if prune:
            # keep only the maximal subspaces: a subspace contained in another
            # one can only produce subspaces contained in that one's offspring
            nxt.sort(key=lambda X: -X.shape[1])
            kept = []
            for M in nxt:
                if not any(M.shape[1] <= O.shape[1] and
                           np.linalg.norm(M - O @ (O.T @ M)) < tol_nest for O in kept):
                    kept.append(M)
            nxt = kept
        if stats is not None:
            stats.append(len(nxt))
        states = nxt
        if progress:
            progress(step + 1, len(lin), len(states))
        if not states:
            break

    keep = []
    for N in sorted(states, key=lambda X: -X.shape[1]):
        if not any(N.shape[1] <= O.shape[1] and
                   np.linalg.norm(N - O @ (O.T @ N)) < tol_nest for O in keep):
            keep.append(N)
    return [(N.shape[1], N) for N in keep], blocks


def fast_component_count(P, tol_rank: float = 1e-9, tol_nest: float = 1e-7,
                         progress=None):
    """Component count via `incremental_linear_cells` plus the definiteness test.
    Same answer as len(decompose(P)["cells"]) for a single implanted vertex, at a
    fraction of the cost. Returns (alive, n_maximal_cells)."""
    cells, blocks = incremental_linear_cells(P, tol_rank, tol_nest, progress)
    quads = [pay for kind, pay in blocks if kind == "quadric"]
    if len(quads) != 1:
        raise ValueError(f"need exactly one quadric block, got {len(quads)}")
    Qv, alive = quads[0], 0
    for d, N in cells:
        Qr = N.T @ Qv @ N
        p, m, z = signature(Qr, tol_rank)
        dead = np.abs(Qr).max() > tol_rank and z == 0 and (p == 0 or m == 0)
        alive += int(not dead)
    return alive, len(cells)


def cell_budget(ncols: int, col: int, places):
    """The closed form split into the two counts a proof has to establish.

    Write a = 2^(col-1) - 1 and c = 2^(ncols-col) - 1.

      Theorem 5.2 (how many flats there are)
          |L| = 2^ncols + 2a + 2c + 2 = B + 2A + 2C - 2 for v* in an interior
          row, independent of the stubs.

      Theorems 6.2 and 6.7 (how many of them die)
          the four families have sizes a, a, c, c and each lives or dies as a
          whole; for BISECTING stubs (or the even spacing of two stubs in one
          sector used by the test suite), with sectors 0 and 3 narrow, the
          deaths are  alpha * a + beta * c  with

              (alpha, beta)   placement
              (0, 0)          one stub per sector, degree 8
              (1, 0)          diagonal pair, narrow side on the right
              (0, 1)          diagonal pair, narrow side on the left
              (2, 0)          the narrow pair {0, 3}
              (0, 2)          the wide pair {1, 2}
              (1, 1)          an adjacent pair {0,1} or {2,3}
              (2, 1)          two stubs inside ONE narrow sector
              (1, 2)          two stubs inside ONE wide sector
              (2, 2)          a single stub

    Away from the bisectors the entries with an empty associated sector can
    change (Corollary 7.3); `criterion.predict_N` evaluates Theorem 6.7 at the
    actual stub directions.

    `places` is a list of (sector, fraction). Returns (|L|, deaths), with deaths
    the pair (c, a) for a diagonal pair, in the order narrow-left, narrow-right.
    """
    a, c = 2 ** (col - 1) - 1, 2 ** (ncols - col) - 1
    N = 2 ** ncols + 2 * a + 2 * c + 2
    secs = tuple(sorted(s for s, _ in places))
    if len(secs) == 1:
        return N, 2 * a + 2 * c
    if len(secs) == 4 and secs == (0, 1, 2, 3):
        return N, 0
    if len(secs) == 2:
        if secs in ((0, 2), (1, 3)):
            return N, (c, a)
        if secs == (0, 3):
            return N, 2 * a
        if secs == (1, 2):
            return N, 2 * c
        if secs in ((0, 1), (2, 3)):
            return N, a + c
        if secs in ((0, 0), (3, 3)):
            return N, 2 * a + c
        if secs in ((1, 1), (2, 2)):
            return N, a + 2 * c
    return N, None


def predicted_at(nrows: int, ncols: int, col: int, row: int, places):
    """Component count N for BISECTING stubs, given as (sector, fraction).

    Valid for stubs that bisect their sectors (two stubs in one sector: at 1/3
    and 2/3 of it) at a Miura vertex whose narrow sectors are 0 and 3. For any
    other position use `criterion.predict_N`, which evaluates Theorem 6.7 at
    the actual stub directions; Corollary 7.3 shows that off the bisectors the
    classes with an empty associated sector change value.

    A = 2^(col-1), C = 2^(ncols-col), B = 2^ncols.

        outer row, anything                           B   (Theorem 7.5)
        one stub, any sector                          B + 2
        sectors {1,2}, the wide pair                  B + 2A
        sectors {0,3}, the narrow pair                B + 2C
        sectors {0,1} or {2,3}, adjacent              B + A + C
        sectors {0,2} or {1,3}, diagonal              B + 2A + C - 1 (narrow left)
                                                      B + A + 2C - 1 (narrow right)
        two stubs inside ONE wide sector              B + A + 1
        two stubs inside ONE narrow sector            B + C + 1
        one stub per sector, degree 8                 B + 2A + 2C - 2

    The outer-row and degree-8 rows hold for every stub position (Theorem 7.5
    and Corollary 7.2(a)). For a diagonal pair both values are returned; which
    one applies is decided by the side of the narrow sectors (`predict_pattern`).
    Checked 234/234 for ncols = 3..6 with bisecting stubs.
    """
    B = 2 ** ncols
    if row == 1 or row == nrows:
        return B
    A, C = 2 ** (col - 1), 2 ** (ncols - col)
    secs = tuple(sorted(s for s, _ in places))
    if len(secs) == 1:
        return B + 2
    if len(secs) == 4 and secs == (0, 1, 2, 3):
        return B + 2 * A + 2 * C - 2
    if len(secs) == 2:
        if secs in ((0, 1), (2, 3)):
            return B + A + C
        if secs == (1, 2):
            return B + 2 * A
        if secs == (0, 3):
            return B + 2 * C
        if secs in ((0, 2), (1, 3)):
            return (B + 2 * A + C - 1, B + A + 2 * C - 1)
        if secs in ((1, 1), (2, 2)):
            return B + A + 1
        if secs in ((0, 0), (3, 3)):
            return B + C + 1
    return None


def narrow_side(P, slot) -> str:
    """'R' if the two narrow sectors of interior vertex #slot are on the right
    (sectors 0 and 3), 'L' if on the left (sectors 1 and 2). This is the lean
    of the zigzag creases at that vertex."""
    s = P.sectors(slot)
    return "R" if s[0] < s[1] else "L"


def predict_pattern(P, nrows, ncols, slot, places):
    """Bisector count for an actual pattern: like `predicted_at`, but a diagonal
    pair is resolved from the side of the narrow sectors (Corollary 7.3(c)):
    narrow-right gives 2^nc + A + 2C - 1 and narrow-left gives 2^nc + 2A + C - 1.
    Bisecting stubs only; use `criterion.predict_N` otherwise."""
    col, row = P.interior[slot]
    v = predicted_at(nrows, ncols, col, row, places)
    if isinstance(v, tuple):
        B = 2 ** ncols
        A, C = 2 ** (col - 1), 2 ** (ncols - col)
        return B + A + 2 * C - 1 if narrow_side(P, slot) == "R" else B + 2 * A + C - 1
    return v
