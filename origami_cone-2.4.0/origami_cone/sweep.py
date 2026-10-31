"""Systematic sweeps: where does the branch count follow a formula, and where not.

Two candidate formulas were refuted in development and are kept here as null
hypotheses so that any future claim has to clear them:

  H1: #branches = 2^V4                  (V4 = number of degree-4 vertices)
  H2: #branches = 2^(dim K - d_max)     (d_max = dimension of the largest cell)

  H3: raising one vertex of a Miura patch changes the branch count away from
      2^ncols exactly when that vertex lies in an INTERIOR row (1 < j < nrows).

H1 fails badly on pure Miura patches: 2^ncols is the right answer there, not
2^V4. H2 holds on every pure Miura tested and on degree-raised variants whose
raised vertex is in an outer row, but fails otherwise, and the count is then
not even a power of two (6, 10, 12, 14 observed). H3 is the current best
description of where that happens and holds on 40/40 positions tested; the
plausible mechanism is that an interior-row vertex has BOTH of its zig-zag
creases shared with other interior vertices, whereas an outer-row vertex has
one of them as a boundary stub, so raising the former breaks a shared coupling
and raising the latter does not. H3 is an observation, not a theorem. It has since been restated intrinsically
and its scope pinned down in `origami_cone.h3`: it holds on 114/114 single
raises across Miura patches, their transposes and a non-Miura flat-foldable
family, and it is FALSE for two simultaneous raises.
"""
from __future__ import annotations
import numpy as np
from .patterns import miura, raise_degree
from .cone import decompose


def branch_data(P, quadric_samples: int = 60) -> dict:
    r = decompose(P, quadric_samples=quadric_samples)
    cells = r["cells"]
    V4 = sum(1 for kind, pay in r["blocks"] if kind == "linear")
    n = len(cells)
    dmax = max((c.dim for c in cells), default=0)
    dims = sorted((c.dim for c in cells), reverse=True)
    # d_max in H2 is the dimension of the largest LINEAR cell, before the
    # irreducible quadrics cut, so add back one per curved-cutting vertex.
    nq = sum(1 for kind, pay in r["blocks"] if kind == "quadric")
    dmax_lin = dmax + (nq if any(c.kind == "curved" for c in cells) else 0)
    return dict(V=P.V, E=P.E, dimK=r["dim_K"], V4=V4, nq=nq, n=n,
                dims=dims, dmax=dmax, dmax_lin=dmax_lin,
                H1=2 ** V4, H2=2 ** (r["dim_K"] - dmax_lin) if n else 0,
                pow2=(n > 0 and (n & (n - 1)) == 0))


def sweep_pure(sizes=((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4),
                      (3, 1), (3, 2), (3, 3), (4, 2), (4, 3))):
    print("PURE MIURA PATCHES")
    print(f"{'patch':>8s} {'V':>3s} {'E':>3s} {'dimK':>5s} {'V4':>3s} "
          f"{'H1=2^V4':>8s} {'H2':>5s} {'actual':>7s} {'2^ncols':>8s}  dims")
    bad = []
    for nr, nc in sizes:
        P = miura(nr, nc)
        d = branch_data(P)
        tag = []
        if d["n"] != d["H1"]:
            tag.append("H1x")
        if d["n"] != d["H2"]:
            tag.append("H2x")
        if d["n"] != 2 ** nc:
            tag.append("NCx")
            bad.append((nr, nc, d["n"]))
        print(f"{nr}x{nc:<6d} {d['V']:>3d} {d['E']:>3d} {d['dimK']:>5d} {d['V4']:>3d} "
              f"{d['H1']:>8d} {d['H2']:>5d} {d['n']:>7d} {2 ** nc:>8d}  "
              f"{d['dims']} {' '.join(tag)}")
    print(f"\n2^ncols held on {len(sizes) - len(bad)}/{len(sizes)} pure patches"
          + (f"; failures {bad}" if bad else ""))
    return bad


def sweep_positions(nr: int, nc: int, extras=(1, 2, 4)):
    """Raise each interior vertex in turn and record the branch count."""
    base = branch_data(miura(nr, nc))
    print(f"\nMIURA {nr}x{nc}: RAISE ONE VERTEX, POSITION BY POSITION"
          f"   (baseline {base['n']} branches, dimK={base['dimK']})")
    head = " | ".join(f"deg{4 + e}: dimK dmax  H2  act" for e in extras)
    print(f"{'slot':>4s} {'vertex':>8s} | {head}")
    anomalies = []
    P0 = miura(nr, nc)
    for v in range(P0.V):
        row = []
        for e in extras:
            d = branch_data(raise_degree(miura(nr, nc), v, e))
            flag = "" if d["n"] == d["H2"] else " X"
            if d["n"] != d["H2"]:
                anomalies.append(dict(slot=v, vertex=P0.interior[v], degree=4 + e,
                                      actual=d["n"], H2=d["H2"], pow2=d["pow2"],
                                      dims=d["dims"]))
            row.append(f"{d['dimK']:4d} {d['dmax_lin']:4d} {d['H2']:3d} {d['n']:3d}{flag}")
        print(f"{v:>4d} {str(P0.interior[v]):>8s} | " + " | ".join(row))
    if anomalies:
        print("\nANOMALIES (H2 violated):")
        for a in anomalies:
            print(f"  vertex {a['vertex']} raised to degree {a['degree']}: "
                  f"H2 predicts {a['H2']}, actual {a['actual']} "
                  f"({'a power of two' if a['pow2'] else 'NOT a power of two'}), dims {a['dims']}")
    else:
        print("\nno H2 violations at this size")
    return anomalies


def test_H3(sizes=((2, 2), (1, 3), (3, 2), (4, 2), (5, 2), (3, 3)), extra: int = 1):
    """Check H3: deviation from 2^ncols happens iff the raised vertex is in an
    interior row. Returns (n_positions, n_disagreements, rows)."""
    rows = []
    bad = 0
    for nr, nc in sizes:
        P0 = miura(nr, nc)
        base = branch_data(P0)["n"]
        for v in range(P0.V):
            i, j = P0.interior[v]
            inner = bool(1 < j < nr)
            d = branch_data(raise_degree(miura(nr, nc), v, extra))
            deviates = bool(d["n"] != base)
            bad += int(deviates != inner)
            rows.append(dict(patch=(nr, nc), vertex=(i, j), inner_row=inner,
                             base=base, actual=d["n"], deviates=deviates))
    return len(rows), bad, rows


if __name__ == "__main__":
    import sys
    sweep_pure()
    todo = [(2, 2), (2, 3), (3, 3)] if len(sys.argv) < 2 else [tuple(map(int, sys.argv[1].split("x")))]
    allan = []
    for nr, nc in todo:
        allan += sweep_positions(nr, nc)
    print(f"\n=== total anomalies across sweeps: {len(allan)} ===")
    n, bad, _ = test_H3()
    print(f"H3 (deviation iff interior row): {n - bad}/{n} positions agree")
