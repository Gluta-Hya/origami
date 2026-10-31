"""Appendix A, 'The closed form': the counts certified in exact rational
arithmetic (Proposition 11.2), with the rational stubs of Section 11 at the
listed sector fractions. 3 x ncols Miura patch with offset 3/5,
v* = v_{col,2}; 'upper pair' = sectors {0,1}, 'right pair' = sectors {0,3}."""
import os, sys, time
from fractions import Fraction as Fr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from origami_cone.exact import miura_rational, raise_rational, exact_component_count

H = Fr(1, 2)
CASES = [(3, 1, [(0, H), (1, H)], 13), (3, 2, [(0, Fr(1, 3)), (0, Fr(2, 3))], 11),
         (4, 1, [(0, H), (1, H)], 25), (5, 1, [(0, H), (1, H)], 49),
         (5, 1, [(0, H), (3, H)], 64), (5, 3, [(0, H), (1, H)], 40),
         (6, 1, [(0, H), (1, H)], 97), (7, 1, [(0, H), (1, H)], 193),
         (7, 4, [(0, H), (1, H)], 144), (8, 1, [(0, H), (1, H)], 385)]
ok = 0
for nc, col, places, want in CASES:
    t = time.time()
    pos, interior, edges = miura_rational(3, nc)
    slot = [k for k, v in enumerate(interior) if v == (col, 2)][0]
    got, _ = exact_component_count(*raise_rational(pos, interior, edges, slot, places))
    ok += int(got == want)
    print(f"3x{nc} col {col} {places}: exact {got}, paper {want}, {time.time() - t:.1f} s")
print("exact certification:", ok, "/", len(CASES))
