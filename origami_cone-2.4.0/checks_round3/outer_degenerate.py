"""Appendix A, 'Outer rows': the degenerate case of Theorem 7.5 and
Corollary 8.6, a stub continuing the boundary-side zigzag crease through v*
(z^- in the first row, z^+ in the last row). For every outer-row vertex of the
Miura patches below, with that stub alone and with one more stub at a random
position, we check |L| = N = B and that every flat carries a branch of its own
(the test of the 'Branches' paragraph, with up to 30 sampled directions)."""
import os, sys
import numpy as np
_h = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_h, ".."))
sys.path.insert(0, os.path.join(_h, "..", "checks_round2", "model"))
from branches_all import sample_iso, lifts            # noqa: E402
from origami_cone import miura, raise_degree_at       # noqa: E402
from origami_cone.cone import kernel, incremental_linear_cells  # noqa: E402

rng = np.random.default_rng(75)
tot = bad = 0
for nr, nc in ((3, 3), (3, 4), (4, 4), (3, 5)):
    P0 = miura(nr, nc)
    for k, (i0, j0) in enumerate(P0.interior):
        if 1 < j0 < nr:
            continue
        it = sorted(t[0] for t in P0.vdata[k])
        z = it[3] if j0 == 1 else it[1]          # boundary-side zigzag
        ext = (z + np.pi) % (2 * np.pi)
        A = it + [2 * np.pi]
        sec = max(s for s in range(4) if A[s] < ext)
        frac = (ext - A[sec]) / (A[sec + 1] - A[sec])
        for extra in (False, True):
            pl = [(sec, frac)]
            if extra:
                pl.append((int(rng.integers(4)), float(rng.uniform(.05, .95))))
            P = raise_degree_at(P0, k, pl)
            B, _ = kernel(P)
            cells, blocks = incremental_linear_cells(P)
            Qv = [p for kk, p in blocks if kk == "quadric"][0]
            alive = own = 0
            for idx, (d, N) in enumerate(cells):
                M = N.T @ Qv @ N
                M = (M + M.T) / 2
                if sample_iso(M, rng)[1] == "definite":
                    continue
                alive += 1
                for _ in range(30):
                    x, _st = sample_iso(M, rng)
                    X = B @ N @ x
                    X /= np.linalg.norm(X)
                    only = all(np.linalg.norm(X - B @ N2 @ (N2.T @ (B.T @ X))) > 1e-6
                               for j, (d2, N2) in enumerate(cells) if j != idx)
                    if only and lifts(P, X):
                        own += 1
                        break
            ok = len(cells) == alive == own == 2 ** nc
            tot += 1
            bad += int(not ok)
            if not ok:
                print("BAD", nr, nc, (i0, j0), pl, len(cells), alive, own)
print("outer rows, stub continuing the boundary zigzag: |L| = N = N_br = B in",
      tot - bad, "/", tot)
