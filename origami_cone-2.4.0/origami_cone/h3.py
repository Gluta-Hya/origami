"""The single-raise hypothesis, stated intrinsically, and its known scope.

Background. For a quadrilateral mesh, each interior vertex has degree 4, its
restricted second-order form has rank 2, and it therefore contributes a pair of
hyperplanes rather than a curved quadric. Raising one vertex to degree >= 5
replaces its hyperplane pair by a single irreducible quadric, which changes the
arrangement and can change the number of tangent-cone components.

At a degree-4 vertex of a Miura-like mesh exactly two of the four creases are
collinear with each other (the straight fold line running through the vertex);
the other two are not. Call the non-collinear pair the vertex's ZIG-ZAG pair.
This is intrinsic: it needs no grid, no rows and no columns.

    H3 (single raise). Raising exactly one vertex changes the number of
    tangent-cone components away from the unraised baseline IF AND ONLY IF both
    of that vertex's zig-zag creases are shared with another interior vertex.

Status:
  * verified on 114/114 single raises to degree 5 and 6, across Miura patches
    of several shapes, their transposes, and the non-Miura flat-foldable family
    of `patterns.flat_foldable_quad` with non-congruent and sign-flipped
    columns. The branch counts came out identical across all those families,
    which suggests the whole count is combinatorial and does not see the
    crease angles at all;
  * FALSE for two or more simultaneous raises. Every counterexample found so
    far has the two raised vertices adjacent along one zig-zag chain, and the
    deviation can go either way (Miura 2x3: 8 -> 4; Miura 4x2: 4 -> 8), so the
    multi-raise case needs a chain-level statement that has not been found.

H3 is an observation supported by computation, not a theorem. `scan` below is
the falsification harness; run it on anything new before trusting H3 there.
"""
from __future__ import annotations
import itertools
import numpy as np


def zigzag_creases(P, slot: int, tol: float = 1e-9):
    """The non-collinear crease pair at a degree-4 vertex, or None."""
    items = P.vdata[slot]
    if len(items) != 4:
        return None
    angs = [t[0] for t in items]
    es = [t[1] for t in items]
    straight = set()
    for a in range(4):
        for b in range(a + 1, 4):
            if abs(abs(angs[b] - angs[a]) - np.pi) < tol:
                straight |= {es[a], es[b]}
    return [e for e in es if e not in straight]


def shared_creases(P) -> set:
    iset = set(P.interior)
    return {e for key, e in P.cidx.items() if len(set(key) & iset) == 2}


def crease_partners(P) -> dict:
    slot = {v: k for k, v in enumerate(P.interior)}
    return {e: {slot[x] for x in key if x in slot} for key, e in P.cidx.items()}


def predict(P, raised) -> bool:
    """H3's prediction: does raising this set of vertices move the count?

    Only trustworthy for |raised| == 1; see the module docstring.
    """
    sh = shared_creases(P)
    part = crease_partners(P)
    raised = set(raised)
    for v in raised:
        z = zigzag_creases(P, v)
        if z is None:
            continue
        if all(e in sh and not (part[e] - {v}) <= raised for e in z):
            return True
    return False


def scan(build, extras=(1, 2), pairs: int = 0, quadric_samples: int = 20,
         verbose: bool = True):
    """Falsification harness. `build` must return a fresh Pattern each call.

    Returns (n_tested, n_disagreements, failures).
    """
    from .patterns import raise_degree
    from .sweep import branch_data
    P0 = build()
    base = branch_data(P0)["n"]
    sets = [(v,) for v in range(P0.V)]
    if pairs:
        sets += list(itertools.combinations(range(P0.V), 2))[:pairs]
    tot = bad = 0
    fails = []
    for S in sets:
        for e in extras:
            Q = build()
            for v in S:
                Q = raise_degree(Q, v, e)
            n = branch_data(Q, quadric_samples=quadric_samples)["n"]
            dev = bool(n != base)
            pred = predict(P0, S)
            tot += 1
            if dev != pred:
                bad += 1
                fails.append(dict(raised=[P0.interior[v] for v in S], degree=4 + e,
                                  base=base, actual=n, predicted=pred, observed=dev))
    if verbose:
        print(f"{P0.name}: baseline {base}, {tot} tested, {tot - bad} agree, {bad} disagree")
        for f in fails[:8]:
            print(f"   raise {f['raised']} to degree {f['degree']}: "
                  f"{f['base']} -> {f['actual']}, H3 said {f['predicted']}")
    return tot, bad, fails
