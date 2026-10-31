"""Appendix A, 'Branches': Lemma 8.1 checked symbolically (needs sympy, the
optional 'symbolic' extra). At a Miura vertex with lean lambda, creases H_r,
z_u, H_l, z_d at angles 0, theta, pi, 2 pi - theta (lambda = +1) or 0,
pi - theta, pi, pi + theta (lambda = -1), the assignment
z_u = z_d = zeta, H_l = -H_r = lambda G(zeta), G = 2 arctan(cos(theta) tan(zeta/2)),
makes the product of the four rotations the identity, identically in theta and
zeta. Each rotation is written through the tangent of its half angle, so the
check is an identity of polynomials in c = cos(theta), s = sin(theta) and
t = tan(zeta/2) modulo s^2 + c^2 = 1."""
import sympy as sp

c, s, t = sp.symbols("c s t", real=True)


def rot(ux, uy, T):
    """Rotation about (ux, uy, 0) by the angle whose half-angle tangent is T,
    times (1 + T^2) so that every entry is a polynomial."""
    U = sp.Matrix([[0, 0, uy], [0, 0, -ux], [-uy, ux, 0]])
    return (1 + T ** 2) * sp.eye(3) + 2 * T * U + 2 * T ** 2 * (U * U)


ok = True
for lam in (1, -1):
    G = lam * c * t                                   # tan(G/2)
    dirs = [(1, 0), (c, s) if lam == 1 else (-c, s), (-1, 0), (c, -s) if lam == 1 else (-c, -s)]
    halfs = [-G, t, G, t]                             # H_r, z_u, H_l, z_d
    M = sp.eye(3)
    for (ux, uy), T in zip(dirs, halfs):
        M = M * rot(ux, uy, T)
    scale = sp.prod([1 + T ** 2 for T in halfs])
    D = (M - scale * sp.eye(3)).applyfunc(sp.expand)
    D = D.applyfunc(lambda e: sp.expand(sp.rem(sp.Poly(e, s), sp.Poly(s ** 2 + c ** 2 - 1, s)).as_expr()))
    zero = all(sp.simplify(e) == 0 for e in D)
    print(f"lambda = {lam:+d}: Phi_v = I identically: {zero}")
    ok &= zero
    # the straight branch (S): z = 0, H_l = H_r
    h = sp.Symbol("h")
    M = rot(1, 0, h) * rot(-1, 0, h)
    print(f"lambda = {lam:+d}: branch (S) closes: {sp.simplify(M - (1 + h ** 2) ** 2 * sp.eye(3)) == sp.zeros(3)}")
print("Lemma 8.1 symbolic check:", "OK" if ok else "FAILED")
