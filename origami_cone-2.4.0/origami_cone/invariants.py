"""Mountain-valley signatures of branches, and forcing sets.

The mountain-valley pattern of a folding direction is the sign vector of its
fold-angle rates, taken modulo the global flip rho -> -rho. On a pure
quadrilateral mesh each branch is a single ray family and realises exactly one
such pattern, so the pattern labels the branch. Once a vertex of degree >= 5 is
present the branch is a curved cone of positive dimension and realises many
patterns, so the label breaks down. `cell_mv_patterns` measures this.
"""
from __future__ import annotations
import itertools
import numpy as np
from .cone import cell_directions


def mv_pattern(vec, tol: float = 1e-7) -> tuple:
    """Sign vector modulo the global mountain-valley flip."""
    v = np.asarray(vec, float)
    s = np.where(np.abs(v) > tol * max(1.0, np.abs(v).max()), np.sign(v), 0.0).astype(int)
    nz = [x for x in s if x]
    if nz and nz[0] < 0:
        s = -s
    return tuple(int(x) for x in s)


def cell_mv_patterns(res, cell, samples: int = 400, tol: float = 1e-7,
                     seed: int = 3) -> set:
    """Every MV pattern realised inside one cell."""
    B = res["B"]
    out = set()
    if cell.kind == "flat":
        rng = np.random.default_rng(seed)
        for _ in range(samples):
            w = cell.basis @ rng.normal(size=cell.basis.shape[1])
            out.add(mv_pattern(B @ w, tol))
    else:
        for y in cell.points:
            out.add(mv_pattern(B @ (cell.basis @ y), tol))
    return out


def mv_report(res, samples: int = 400, tol: float = 1e-7) -> dict:
    pats = [cell_mv_patterns(res, c, samples, tol) for c in res["cells"]]
    allp = set().union(*pats) if pats else set()
    shared = sum(1 for p in allp
                 if sum(1 for s in pats if p in s) > 1)
    return dict(per_cell=[len(s) for s in pats], total=len(allp), shared=shared,
                labels_branches=all(len(s) == 1 for s in pats) and shared == 0,
                sets=pats)


def forcing_sets(res, E: int, max_size: int = 5, samples: int = 600,
                 tol: float = 1e-7, want_all: bool = True):
    """Smallest sets of creases whose MV reading separates every branch.

    Returns (size, list_of_sets) or (None, []) if none of size <= max_size.
    A set S separates if the collections of patterns-restricted-to-S realised
    by distinct cells are pairwise disjoint.
    """
    pats = [cell_mv_patterns(res, c, samples, tol) for c in res["cells"]]
    if len(pats) < 2:
        return 0, [()]

    def canon(sig):
        nz = [x for x in sig if x]
        if nz and nz[0] < 0:
            sig = tuple(-x for x in sig)
        return tuple(sig)

    for size in range(1, max_size + 1):
        good = []
        for S in itertools.combinations(range(E), size):
            sets = [{canon(tuple(p[e] for e in S)) for p in ps} for ps in pats]
            ok = True
            for a in range(len(sets)):
                for b in range(a + 1, len(sets)):
                    if sets[a] & sets[b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                good.append(S)
                if not want_all:
                    return size, good
        if good:
            return size, good
    return None, []
