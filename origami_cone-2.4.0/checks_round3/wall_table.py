"""Appendix A, 'The death criterion': recompute the 24 entries of Table 2
(tab:wall). 3x4 Miura patch, v* = v_{2,2}, two stubs in the narrow sector 0
at the listed fractions and one stub bisecting sector 3; four vertex positions,
listed by the width theta of the narrow sector. Each entry is computed with the
full incremental count and compared with Corollary 7.3(a) and with the table."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from origami_cone import miura, raise_degree_at
from origami_cone.cone import fast_component_count
from origami_cone.criterion import vertex_angles

FR = [(0.2, 0.8), (1 / 3, 2 / 3), (0.4, 0.6), (0.1, 0.5), (0.5, 0.9), (0.3, 0.4)]
TABLE = {35.2: [26, 26, 24, 24, 26, 24], 44.8: [26, 24, 24, 24, 26, 24],
         59.0: [24, 24, 24, 24, 26, 24], 73.0: [24, 24, 24, 24, 24, 24]}
bad = tot = 0
for deg, row in TABLE.items():
    P0 = miura(3, 4, d=1 / np.tan(np.radians(deg)))
    slot = [k for k, v in enumerate(P0.interior) if v == (2, 2)][0]
    th = vertex_angles(P0, slot)[0]
    out = []
    for (f1, f2), want in zip(FR, row):
        N = fast_component_count(raise_degree_at(P0, slot, [(0, f1), (0, f2), (3, .5)]))[0]
        # Corollary 7.3(a): the left side is empty; the stub nearest z+ is the
        # upper one in sector 0, the one nearest z- bisects sector 3
        gp, bp = (1 - f2) * th, f2 * th
        gm = bm = th / 2
        live = np.cos(th) ** 2 * np.sin(bp) * np.sin(bm) >= np.sin(gp) * np.sin(gm)
        pred = 26 if live else 26 - 2 * 1
        tot += 1
        bad += int(not (N == want == pred))
        out.append(N)
    print(f"theta = {np.degrees(th):5.1f} deg: {out}  table {row}")
print("wall table: agree", tot - bad, "/", tot)
