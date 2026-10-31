"""Tachi's exact finite foldability condition for quad meshes.

Established numerically in this project: at a developable degree-4 vertex the
ratio tan(rho_i/2)/tan(rho_j/2) is EXACTLY constant along a branch if and only
if the vertex is flat-foldable (Kawasaki). Measured drift is 1e-15 for
flat-foldable vertices and 1.5 to 7.9 percent otherwise. The conversion
coefficient machinery is therefore valid exactly on the flat-foldable class,
and that is where this module applies.

With creases c1..c4 counter-clockwise and theta_i the sector between c_i and
c_{i+1}, flat-foldability forces theta_1 + theta_3 = theta_2 + theta_4 = pi.
Writing t_i = tan(rho_{c_i}/2), the two branches are exactly

    branch A:  (t1, t2, t3, t4) proportional to (1,  k_plus, -1,  k_plus)
    branch B:  (t1, t2, t3, t4) proportional to (1, k_minus, +1, -k_minus)
    k_plus  =  sin((th1+th2)/2) / sin((th1-th2)/2)
    k_minus = -cos((th1+th2)/2) / cos((th1-th2)/2)

verified against numerical continuation to 4e-15 over 40 branch pairs.

Around an inner facet each corner relates the two facet creases meeting there
by a constant, and going once around requires the SIGNED product to be +1.
Using |product| = 1 instead admits false positives: a constructed example whose
only branch choice gives -1 has an empty tangent cone and does not fold.
"""
import numpy as np
import itertools
from .patterns import Pattern

import numpy as np

def ratios_closed(P, slot):
    """Both branches as dicts {global crease id: t_i / t_1}."""
    items = P.vdata[slot]
    if len(items) != 4:
        return None
    ids = [e for _, e, _ in items]
    th = P.sectors(slot)
    t1, t2 = th[0], th[1]
    sm, sd = np.sin((t1 + t2) / 2), np.sin((t1 - t2) / 2)
    cm, cd = np.cos((t1 + t2) / 2), np.cos((t1 - t2) / 2)
    if abs(sd) < 1e-14 or abs(cd) < 1e-14:
        return None
    kp, km = sm / sd, -cm / cd
    A = [1.0, kp, -1.0, kp]
    B = [1.0, km, 1.0, -km]
    return [{ids[i]: A[i] for i in range(4)}, {ids[i]: B[i] for i in range(4)}]


def inner_facets(P):
    """Quad faces all four of whose corners are interior vertices, as
    (corners, bounding creases) in matching cyclic order."""
    iset = set(P.interior)
    out = []
    for v in P.interior:
        i, j = v
        quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        if all(q in iset for q in quad):
            creases, ok = [], True
            for a in range(4):
                key = frozenset((quad[a], quad[(a + 1) % 4]))
                if key not in P.cidx:
                    ok = False
                    break
                creases.append(P.cidx[key])
            if ok:
                out.append((quad, creases))
    return out


def loop_products(P):
    """Signed loop product per inner facet, for every branch choice.
    Returns a list of (choice, [product per facet]) or None if inapplicable."""
    slot = {v: k for k, v in enumerate(P.interior)}
    R = {}
    for v in P.interior:
        c = ratios_closed(P, slot[v])
        if c is None:
            return None
        R[v] = c
    facets = inner_facets(P)
    if not facets:
        return None
    out = []
    for ch in itertools.product((0, 1), repeat=len(P.interior)):
        sel = {v: R[v][ch[slot[v]]] for v in P.interior}
        prods, ok = [], True
        for quad, cr in facets:
            p = 1.0
            for a in range(4):
                corner = quad[(a + 1) % 4]
                ein, eout = cr[a], cr[(a + 1) % 4]
                t = sel[corner]
                if ein not in t or eout not in t or abs(t[ein]) < 1e-13:
                    ok = False
                    break
                p *= t[eout] / t[ein]
            if not ok:
                break
            prods.append(p)
        if ok:
            out.append((ch, prods))
    return out


def foldability_defect(P, tol: float = 1e-9):
    """min over branch choices of max_facet |product - 1|. Zero iff Tachi's
    condition holds. Only meaningful when every vertex is flat-foldable."""
    lp = loop_products(P)
    if lp is None:
        return None
    best = np.inf
    for ch, prods in lp:
        best = min(best, max(abs(p - 1.0) for p in prods))
    return None if best == np.inf else float(best)


def is_rigid_foldable(P, tol: float = 1e-9) -> bool:
    d = foldability_defect(P, tol)
    return d is not None and d < tol


def flat_foldable_vertices(P, tol_deg: float = 1e-6) -> bool:
    return all(abs(np.degrees(P.kawasaki(k))) < tol_deg for k in range(P.V))
