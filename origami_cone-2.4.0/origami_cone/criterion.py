"""The death criterion of the paper (Theorem 6.7), evaluated at the implanted
vertex alone, and the count N it predicts for an interior-row implant.

For a family F = F(side, eta) at a Miura vertex with narrow angle theta and
lean lam, with stubs at the given directions, F dies iff t = 1, or t >= 2 and
Q1*Q2 < Q12**2, where the Q's are read off the angles between neighbouring
creases (Lemma 6.3 and the proof of Theorem 6.7). The implementation below
builds the Gram matrix G = [f_i(y_k)] from the dual vectors D_j directly."""
from __future__ import annotations
import numpy as np

FAMILIES = (("L", "-"), ("L", "+"), ("R", "-"), ("R", "+"))


def _vertex(theta, lam, stub_angles):
    au = theta if lam == 1 else np.pi - theta
    ad = 2 * np.pi - theta if lam == 1 else np.pi + theta
    cr = [("Hr", 0.0), ("zp", au), ("Hl", np.pi), ("zm", ad)]
    cr += [(f"s{k}", float(a) % (2 * np.pi)) for k, a in enumerate(stub_angles)]
    cr.sort(key=lambda c: c[1])
    return cr


def _dual(cr, j):
    n = len(cr)
    a = np.array([c[1] for c in cr])
    out = np.zeros(n)
    gp = (a[j] - a[j - 1]) % (2 * np.pi)
    gn = (a[(j + 1) % n] - a[j]) % (2 * np.pi)
    out[j] = -1 / np.tan(gp) - 1 / np.tan(gn)
    out[j - 1] += 1 / np.sin(gp)
    out[(j + 1) % n] += 1 / np.sin(gn)
    return out


def gram_determinant(theta, lam, stub_angles, side, eta):
    """det G for the family F(side, eta); Theorem 6.7 says F dies iff det G < 0
    (t >= 2)."""
    cr = _vertex(theta, lam, stub_angles)
    idx = {c[0]: i for i, c in enumerate(cr)}
    n = len(cr)
    f1 = np.zeros(n)
    f1[idx["zp" if eta == "-" else "zm"]] = 1
    f2 = np.zeros(n)
    zm = idx["zm" if eta == "-" else "zp"]
    if side == "L":
        f2[idx["Hl"]] = 1
        f2[zm] = -lam * np.cos(theta)
    else:
        f2[idx["Hr"]] = 1
        f2[zm] = lam * np.cos(theta)
    Y = [2 * sum(c * _dual(cr, k) for k, c in enumerate(f) if c != 0) for f in (f1, f2)]
    G = np.array([[f @ y for y in Y] for f in (f1, f2)])
    return float(np.linalg.det(G))


def family_dies(theta, lam, stub_angles, side, eta, tol=1e-12):
    if len(stub_angles) == 1:
        return True
    return gram_determinant(theta, lam, stub_angles, side, eta) < -tol


def vertex_angles(P0, slot):
    """theta and lean of an interior Miura (or one-angle-per-column) vertex."""
    it = sorted(t[0] for t in P0.vdata[slot])
    zp = [a for a in it if 0 < a < np.pi][0]
    lam = 1 if zp < np.pi / 2 else -1
    return (zp if lam == 1 else np.pi - zp), lam, it


def predict_N(P0, nr, nc, slot, places):
    """N predicted by Theorems 5.2, 6.2 and 6.7 for v* in an interior row.
    `places` are (sector, fraction) pairs as in `raise_degree_at`."""
    i0, j0 = P0.interior[slot]
    if not 1 < j0 < nr:
        raise ValueError("predict_N is for interior rows")
    theta, lam, it = vertex_angles(P0, slot)
    A4 = list(it) + [2 * np.pi]
    stubs = [A4[s] + f * (A4[s + 1] - A4[s]) for s, f in places]
    A, C = 2 ** (i0 - 1), 2 ** (nc - i0)
    N = 2 ** nc + 2 * A + 2 * C - 2
    for side, eta in FAMILIES:
        if family_dies(theta, lam, stubs, side, eta):
            N -= (A - 1) if side == "L" else (C - 1)
    return N
