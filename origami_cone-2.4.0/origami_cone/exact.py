"""Exact rational arithmetic: the component count with no floating point at all.

Every count this project reports is a count of subspaces and of signatures, and
both were decided with tolerances (tol_rank, tol_nest). Tolerances were already
caught inventing one result in development, so the headline numbers deserve to
be certified without them.

They can be. Write the fold angle of crease e as rho_e = sigma_e * |w_e|, where
w_e is the crease vector from the vertex to its neighbour. Then

    first order   sum_e rho_e u_e = sum_e sigma_e w_e
    second order  rho_p rho_q sin(a_q - a_p) = sigma_p sigma_q (w_p x w_q)

so with rational vertex positions BOTH J0 and every Hessian block have rational
entries in sigma coordinates, with no square roots anywhere. A Miura patch with
d = 3/5 and unit spacing is rational, and so is every degree raise that places
a stub at a rational point.

Everything below is Fraction arithmetic: nullspace by exact row reduction,
subspaces canonicalised by reduced row echelon form so that equality is
equality of matrices rather than a norm below a threshold, and definiteness by
exact symmetric elimination rather than eigenvalues.
"""
from __future__ import annotations
from fractions import Fraction as F
import itertools


# ------------------------------------------------------------ linear algebra
def rref(rows):
    """Reduced row echelon form over Q. Returns (rows, pivot columns)."""
    M = [list(r) for r in rows if any(x != 0 for x in r)]
    if not M:
        return [], []
    ncol = len(M[0])
    piv, r = [], 0
    for c in range(ncol):
        k = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if k is None:
            continue
        M[r], M[k] = M[k], M[r]
        inv = F(1, 1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return [row for row in M[:r]], piv


def nullspace(rows, ncol):
    """Basis of {x : A x = 0} over Q, as a list of row vectors."""
    R, piv = rref(rows)
    free = [c for c in range(ncol) if c not in piv]
    basis = []
    for fc in free:
        v = [F(0)] * ncol
        v[fc] = F(1)
        for i, pc in enumerate(piv):
            v[pc] = -R[i][fc]
        basis.append(v)
    return basis


def mat_vec(B, f):
    return [sum(b[k] * f[k] for k in range(len(f))) for b in B]


def cut(B, f):
    """B is a basis (rows) of a subspace; return a basis of B intersect {f.x=0}."""
    w = mat_vec(B, f)
    if all(x == 0 for x in w):
        return B
    cs = nullspace([w], len(B))
    out = []
    for c in cs:
        out.append([sum(c[i] * B[i][k] for i in range(len(B))) for k in range(len(B[0]))])
    R, _ = rref(out)
    return R


def key_of(B):
    R, _ = rref(B)
    return tuple(tuple(r) for r in R)


def contains(T, S):
    """Is span(S) contained in span(T)?"""
    rT, _ = rref(T)
    rTS, _ = rref(list(T) + list(S))
    return len(rT) == len(rTS) and rT == rTS


def signature(M):
    """Exact (positive, negative, zero) inertia of a symmetric rational matrix,
    by symmetric Gaussian elimination with a symmetric pivot search. No
    eigenvalues, no tolerance."""
    A = [list(r) for r in M]
    n = len(A)
    pos = neg = 0
    idx = list(range(n))
    while idx:
        d = next((i for i in idx if A[i][i] != 0), None)
        if d is None:
            off = next(((i, j) for i in idx for j in idx if i < j and A[i][j] != 0), None)
            if off is None:
                break
            i, j = off
            for k in range(n):
                A[i][k] += A[j][k]
            for k in range(n):
                A[k][i] += A[k][j]
            continue
        p = A[d][d]
        pos += int(p > 0)
        neg += int(p < 0)
        for i in idx:
            if i == d or A[i][d] == 0:
                continue
            f = A[i][d] / p
            for k in idx:
                A[i][k] -= f * A[d][k]
            for k in idx:
                A[k][i] -= f * A[k][d]
        idx.remove(d)
    return pos, neg, n - pos - neg


# ---------------------------------------------------------------- patterns
def _cmp_key(w):
    """Exact CCW ordering key for a rational direction, without trigonometry:
    half-plane first, then cross product inside the half-plane."""
    x, y = w
    half = 0 if (y > 0 or (y == 0 and x > 0)) else 1
    return half, x, y


def _ccw_sort(items):
    """Sort (index, w) by angle, exactly. Comparison is by cross product."""
    import functools

    def cmp(a, b):
        ha, hb = _cmp_key(a[1])[0], _cmp_key(b[1])[0]
        if ha != hb:
            return -1 if ha < hb else 1
        cr = a[1][0] * b[1][1] - a[1][1] * b[1][0]
        if cr > 0:
            return -1
        if cr < 0:
            return 1
        return 0

    return sorted(items, key=functools.cmp_to_key(cmp))


def miura_rational(nrows, ncols, d=F(3, 5)):
    """Miura patch with exactly rational vertex positions."""
    pos = {}
    for j in range(nrows + 2):
        off = d if j % 2 == 1 else F(0)
        for i in range(ncols + 2):
            pos[(i, j)] = (F(i) + off, F(j))
    interior = [(i, j) for j in range(1, nrows + 1) for i in range(1, ncols + 1)]
    edges = []
    for (i, j) in interior:
        for nb in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            edges.append(((i, j), nb))
    return pos, interior, edges


def build(pos, interior, edges):
    """(cidx, vdata) with vdata[v] = list of (crease index, w) in CCW order."""
    iset = set(interior)
    cidx = {}
    for u, v in edges:
        if u in iset or v in iset:
            k = frozenset((u, v))
            if k not in cidx:
                cidx[k] = len(cidx)
    adj = {v: [] for v in interior}
    for k in cidx:
        u, v = tuple(k)
        if u in iset:
            adj[u].append(v)
        if v in iset:
            adj[v].append(u)
    vdata = []
    for v in interior:
        items = [(cidx[frozenset((v, nb))],
                  (pos[nb][0] - pos[v][0], pos[nb][1] - pos[v][1])) for nb in adj[v]]
        vdata.append(_ccw_sort(items))
    return cidx, vdata


def raise_rational(pos, interior, edges, slot, places):
    """Implant stubs at interior vertex #slot with RATIONAL directions.

    `places` is a list of (sector, fraction) exactly as `raise_degree_at` takes,
    with fraction in {1/3, 1/2, 2/3}. The float builder uses the angular
    bisector, whose direction is irrational; here the stub direction is a
    rational combination of the two crease vectors bounding the sector,

        1/2 -> w_p + w_q        1/3 -> 2 w_p + w_q      2/3 -> w_p + 2 w_q

    which is rational and, since every Miura sector is under 180 degrees,
    strictly inside the sector and in the same cyclic order. The component
    count is constant on an open set of vertex positions, which the position
    sweeps established, so this lands in the same class as the bisector.
    Sector indices always refer to the ORIGINAL vertex, as in the float builder.
    """
    pos = dict(pos)
    edges = list(edges)
    v = interior[slot]
    _, vdata = build(pos, interior, edges)
    orig = list(vdata[slot])
    nS = len(orig)
    mix = {F(1, 2): (1, 1), F(1, 3): (2, 1), F(2, 3): (1, 2)}
    for t, (sec, frac) in enumerate(places):
        frac = F(frac).limit_denominator(12)
        if frac not in mix:
            raise ValueError("fraction must be 1/3, 1/2 or 2/3 for a rational stub")
        cp, cq = mix[frac]
        wp = orig[sec % nS][1]
        wq = orig[(sec + 1) % nS][1]
        w = (cp * wp[0] + cq * wq[0], cp * wp[1] + cq * wq[1])
        stub = ("stub", slot, t)
        pos[stub] = (pos[v][0] + w[0], pos[v][1] + w[1])
        edges.append((v, stub))
    return pos, interior, edges


# ------------------------------------------------------------ the arrangement
def first_order_rows(vdata, E):
    rows = []
    for items in vdata:
        rc = [F(0)] * E
        rs = [F(0)] * E
        for e, w in items:
            rc[e] += w[0]
            rs[e] += w[1]
        rows += [rc, rs]
    return rows


def hessian_block(items, E):
    Q = [[F(0)] * E for _ in range(E)]
    for p in range(len(items)):
        for q in range(p + 1, len(items)):
            ep, wp = items[p]
            eq, wq = items[q]
            c = F(1, 4) * (wp[0] * wq[1] - wp[1] * wq[0])
            Q[ep][eq] += c
            Q[eq][ep] += c
    return Q


def restrict(B, Q):
    """B Q B^T for a basis given as rows."""
    n, E = len(B), len(B[0])
    QB = [[sum(Q[r][c] * b[c] for c in range(E) if Q[r][c] != 0) for r in range(E)]
          for b in B]
    return [[sum(B[i][r] * QB[j][r] for r in range(E)) for j in range(n)]
            for i in range(n)]


def _is_square(q: F):
    from math import isqrt
    if q < 0:
        return None
    n, d = q.numerator, q.denominator
    rn, rd = isqrt(n), isqrt(d)
    if rn * rn == n and rd * rd == d:
        return F(rn, rd)
    return None


def factor_rank2(M):
    """Write x^T M x = (u.x)(v.x) for a rank-2 symmetric rational M.

    Both factors vanish on ker M, so they are determined by their values on any
    complement, where the form is a binary quadratic a s^2 + b s t + c t^2. It
    splits over Q exactly when b^2 - 4ac is a rational square, and for every
    degree-4 Miura vertex it is. Returns (u, v) or None.
    """
    n = len(M)
    K = nullspace(M, n)
    comp = []
    for i in range(n):
        e = [F(0)] * n
        e[i] = F(1)
        if not contains(K + comp, [e]):
            comp.append(e)
        if len(comp) == 2:
            break
    if len(comp) != 2:
        return None
    s, t = comp
    Ms = [sum(M[i][j] * s[j] for j in range(n)) for i in range(n)]
    Mt = [sum(M[i][j] * t[j] for j in range(n)) for i in range(n)]
    a = sum(s[i] * Ms[i] for i in range(n))
    b = 2 * sum(s[i] * Mt[i] for i in range(n))
    c = sum(t[i] * Mt[i] for i in range(n))
    disc = b * b - 4 * a * c
    r = _is_square(disc)
    if r is None:
        return None
    # a s^2 + b s t + c t^2 = (p1 s + p2 t)(q1 s + q2 t)
    if a != 0:
        p = (F(1), (b - r) / (2 * a))
        q = (a, a * (b + r) / (2 * a))
    elif c != 0:
        p = ((b - r) / (2 * c), F(1))
        q = (c * (b + r) / (2 * c), c)
    else:
        p, q = (F(1), F(0)), (F(0), b)
    # lift: u vanishes on ker M and takes value p1 on s, p2 on t
    rows = K + [s, t]
    targets_u = [F(0)] * len(K) + [p[0], p[1]]
    targets_v = [F(0)] * len(K) + [q[0], q[1]]
    aug = [list(rows[i]) + [targets_u[i], targets_v[i]] for i in range(len(rows))]
    # solve rows . u = targets, i.e. u is a vector with rows[i] . u = t_i
    R, piv = rref([list(rows[i]) + [F(0)] * 0 for i in range(len(rows))])
    # do it by explicit elimination on the augmented system
    A = [list(rows[i]) + [targets_u[i], targets_v[i]] for i in range(len(rows))]
    m = len(A)
    r0 = 0
    pivots = []
    for col in range(n):
        k = next((i for i in range(r0, m) if A[i][col] != 0), None)
        if k is None:
            continue
        A[r0], A[k] = A[k], A[r0]
        inv = F(1) / A[r0][col]
        A[r0] = [x * inv for x in A[r0]]
        for i in range(m):
            if i != r0 and A[i][col] != 0:
                f = A[i][col]
                A[i] = [x - f * y for x, y in zip(A[i], A[r0])]
        pivots.append(col)
        r0 += 1
    u = [F(0)] * n
    v = [F(0)] * n
    for i, col in enumerate(pivots):
        u[col] = A[i][n]
        v[col] = A[i][n + 1]
    return u, v


def exact_component_count(pos, interior, edges, verbose=False):
    """Number of tangent-cone components, certified in exact rational arithmetic.

    No tolerance is used anywhere: subspaces are compared by reduced row echelon
    form, ranks come from exact elimination, and definiteness from exact inertia.
    Returns (alive, n_maximal_cells).
    """
    cidx, vdata = build(pos, interior, edges)
    E = len(cidx)
    B = nullspace(first_order_rows(vdata, E), E)
    k = len(B)
    lin, quad = [], []
    for items in vdata:
        M = restrict(B, hessian_block(items, E))
        p, n, z = signature(M)
        if p + n == 0:
            continue
        if p == 1 and n == 1:
            f = factor_rank2(M)
            if f is None:
                raise ValueError("rank-2 block does not split over Q")
            lin.append(f)
        else:
            quad.append(M)
    if verbose:
        print(f"    dim ker {k}, blok linear {len(lin)}, blok quadric {len(quad)}")
    if len(quad) != 1:
        raise ValueError(f"need exactly one quadric block, got {len(quad)}")

    ident = [[F(1) if i == j else F(0) for j in range(k)] for i in range(k)]
    states = {key_of(ident): ident}
    for step, (u, v) in enumerate(lin):
        nxt = {}
        for S in states.values():
            for f in (u, v):
                T = cut(S, f)
                if not T:
                    continue
                nxt.setdefault(key_of(T), T)
        states = nxt
        if verbose:
            print(f"      simpul {step+1}/{len(lin)}: subruang berbeda {len(states)}",
                  flush=True)
    cells = sorted(states.values(), key=lambda S: -len(S))
    keep = []
    for S in cells:
        if not any(len(S) <= len(T) and contains(T, S) for T in keep):
            keep.append(S)
    Q = quad[0]
    alive = 0
    for S in keep:
        R = restrict(S, Q)
        p, n, z = signature(R)
        dead = (p + n > 0) and z == 0 and (p == 0 or n == 0)
        alive += int(not dead)
    return alive, len(keep)
