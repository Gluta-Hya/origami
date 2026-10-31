"""Flat crease patterns and their combinatorics.

A pattern is a planar straight-line graph. Vertices are either *interior*
(paper all around, sector angles summing to 2*pi) or *boundary*. Every edge
incident to at least one interior vertex is a *crease* and carries one fold
angle variable; edges between two boundary vertices are paper boundary and
carry nothing.

Coordinates are exact rationals where possible, so that the rescaled
first-order matrix and Hessian blocks (see cone.py) have rational entries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import numpy as np


@dataclass
class Pattern:
    pos: dict                      # vertex id -> np.array([x, y])
    interior: list                 # ids of interior vertices, in a fixed order
    cidx: dict                     # frozenset({u, v}) -> crease index
    vdata: list                    # per interior vertex: [(angle, crease, unit_axis), ...] CCW
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def E(self) -> int:
        return len(self.cidx)

    @property
    def V(self) -> int:
        return len(self.interior)

    def degrees(self) -> list:
        return [len(it) for it in self.vdata]

    def sectors(self, k: int) -> np.ndarray:
        a = np.array([t[0] for t in self.vdata[k]])
        return np.diff(np.append(a, a[0] + 2 * np.pi))

    def kawasaki(self, k: int) -> float:
        s = self.sectors(k)
        return float(sum((-1) ** i * s[i] for i in range(len(s))))

    def crease_of(self, u, v) -> int:
        return self.cidx[frozenset((u, v))]

    def summary(self) -> str:
        return (f"{self.name or 'pattern'}: V={self.V} E={self.E} "
                f"degrees={self.degrees()} naive_DOF={self.E - 3 * self.V}")


def from_vertices(pos: dict, interior, edges, name: str = "") -> Pattern:
    """Build a Pattern. `edges` is any iterable of (u, v) pairs; only those
    incident to an interior vertex become creases."""
    pos = {k: np.asarray(v, dtype=float) for k, v in pos.items()}
    interior = list(interior)
    iset = set(interior)
    cidx = {}
    for u, v in edges:
        if u in iset or v in iset:
            key = frozenset((u, v))
            if key not in cidx:
                cidx[key] = len(cidx)
    adj = {v: [] for v in interior}
    for key in cidx:
        u, v = tuple(key)
        if u in iset:
            adj[u].append(v)
        if v in iset:
            adj[v].append(u)
    vdata = []
    for v in interior:
        items = []
        for nb in adj[v]:
            w = pos[nb] - pos[v]
            n = float(np.linalg.norm(w))
            if n == 0.0:
                raise ValueError(f"zero-length crease at {v}-{nb}")
            u = w / n
            ang = float(np.arctan2(u[1], u[0]) % (2 * np.pi))
            items.append((ang, cidx[frozenset((v, nb))], u))
        items.sort(key=lambda t: t[0])
        angs = [t[0] for t in items]
        if len(set(np.round(angs, 12))) != len(angs):
            raise ValueError(f"coincident crease directions at vertex {v}")
        vdata.append(items)
    return Pattern(pos=pos, interior=interior, cidx=cidx, vdata=vdata, name=name)


# ---------------------------------------------------------------- builders

def single_vertex(angles_deg, name: str = "") -> Pattern:
    """One interior vertex with creases at the given directions (degrees)."""
    ang = np.radians(np.asarray(angles_deg, dtype=float))
    pos = {0: np.zeros(2)}
    edges = []
    for k, a in enumerate(ang):
        pos[("b", k)] = np.array([np.cos(a), np.sin(a)])
        edges.append((0, ("b", k)))
    return from_vertices(pos, [0], edges,
                         name or f"single vertex deg {len(ang)}")


def single_vertex_sectors(sectors_deg, name: str = "") -> Pattern:
    s = np.asarray(sectors_deg, dtype=float)
    if abs(s.sum() - 360.0) > 1e-9:
        raise ValueError(f"sectors must sum to 360, got {s.sum()}")
    ang = np.concatenate([[0.0], np.cumsum(s)[:-1]])
    return single_vertex(ang, name or f"vertex sectors {list(s)}")


def miura(nrows: int, ncols: int, d=Fraction(3, 5), a=1, b=1,
          jitter: float = 0.0, seed: int = 0, name: str = "") -> Pattern:
    """Standard Miura-ori patch with `nrows` x `ncols` interior vertices.

    Rows j = 0..nrows+1, columns i = 0..ncols+1; odd rows are shifted by d.
    Interior vertices are (i, j) with 1 <= i <= ncols, 1 <= j <= nrows.
    """
    pos = {}
    for j in range(nrows + 2):
        off = d if j % 2 == 1 else 0
        for i in range(ncols + 2):
            pos[(i, j)] = np.array([float(i * a + off), float(j * b)])
    if jitter > 0:
        rng = np.random.default_rng(seed)
        for k in pos:
            pos[k] = pos[k] + rng.normal(0.0, jitter, 2)
    interior = [(i, j) for j in range(1, nrows + 1) for i in range(1, ncols + 1)]
    edges = []
    for (i, j) in interior:
        for nb in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            edges.append(((i, j), nb))
    P = from_vertices(pos, interior, edges,
                      name or f"Miura {nrows}x{ncols}" + (f" jitter={jitter}" if jitter else ""))
    P.meta.update(kind="miura", nrows=nrows, ncols=ncols, d=float(d))
    return P


def raise_degree(P: Pattern, slot: int, nextra: int = 2, reach: float = 0.62) -> Pattern:
    """Add `nextra` stub creases at interior vertex #slot, each bisecting the
    currently widest sector. Raises that vertex's degree by nextra and leaves
    every other vertex untouched."""
    pos = dict(P.pos)
    cidx = dict(P.cidx)
    vdata = [list(it) for it in P.vdata]
    v = P.interior[slot]
    base = P.pos[v]
    it = vdata[slot]
    for _ in range(nextra):
        angs = sorted(t[0] for t in it)
        gaps = [(((angs[(m + 1) % len(angs)] - angs[m]) % (2 * np.pi)), angs[m])
                for m in range(len(angs))]
        w, a0 = max(gaps)
        new = float((a0 + w / 2) % (2 * np.pi))
        e = len(cidx)
        stub = ("stub", slot, e)
        pos[stub] = base + reach * np.array([np.cos(new), np.sin(new)])
        cidx[frozenset((v, stub))] = e
        it.append((new, e, np.array([np.cos(new), np.sin(new)])))
        it.sort(key=lambda t: t[0])
    Q = Pattern(pos=pos, interior=list(P.interior), cidx=cidx, vdata=vdata,
                name=f"{P.name} + deg{len(vdata[slot])}@{v}", meta=dict(P.meta))
    Q.meta.setdefault("raised", []).append((v, len(vdata[slot])))
    return Q


def perturb(P: Pattern, sigma: float, seed: int = 0) -> Pattern:
    """Jitter every vertex position. Keeps the pattern a valid flat crease
    pattern (shared creases stay exactly antiparallel) but destroys any
    exact tuning such as Kawasaki."""
    rng = np.random.default_rng(seed)
    pos = {k: v + rng.normal(0.0, sigma, 2) for k, v in P.pos.items()}
    edges = [tuple(k) for k in P.cidx]
    return from_vertices(pos, P.interior, edges, f"{P.name} perturbed {sigma}")


def flat_foldable_quad(nrows: int, ncols: int, t, heights=None, spacing=1.0,
                       name: str = "") -> Pattern:
    """Flat-foldable quadrilateral mesh with straight horizontal fold lines.

    Every horizontal fold line is a straight line across the patch, and column
    i carries its own zig-zag parameter t[i]. Writing h_l for the row heights
    and X_i^0 for a base abscissa,

        y_j     = sum_{l<j} h_l
        X_{i,j} = X_i^0 + t_i * sum_{l<j} (-1)^l h_l

    At vertex (i, j) the two zig-zag creases then point along
    ((-1)^j t_i, +1) and ((-1)^j t_i, -1): a mirror pair about the horizontal,
    so Kawasaki holds at every interior vertex by construction and the mesh is
    rigidly foldable (Tachi, IASS 2009). Miura-ori is the special case of
    constant t and constant h; taking t non-constant gives columns that are
    NOT congruent to each other, which is the point of this builder.
    """
    t = list(t)
    if len(t) != ncols:
        raise ValueError(f"need one t per interior column, got {len(t)} for {ncols}")
    if any(abs(x) < 1e-12 for x in t):
        raise ValueError("t_i must be nonzero (a zero column has collinear creases)")
    h = [1.0] * (nrows + 1) if heights is None else list(heights)
    if len(h) != nrows + 1:
        raise ValueError(f"need {nrows + 1} row heights, got {len(h)}")
    y = np.concatenate([[0.0], np.cumsum(h)])
    alt = np.concatenate([[0.0], np.cumsum([(-1) ** l * h[l] for l in range(len(h))])])

    # Column i wanders by t_i * alt[j] around its base abscissa, so the bases
    # must be separated by more than |t_{i+1} - t_i| * max_j |alt[j]|.
    A = float(np.max(np.abs(alt)))
    X0 = [0.0]
    for i in range(1, ncols):
        X0.append(X0[-1] + spacing + abs(t[i] - t[i - 1]) * A)

    pos = {}
    for i in range(1, ncols + 1):
        for j in range(nrows + 2):
            pos[(i, j)] = np.array([X0[i - 1] + t[i - 1] * alt[j], y[j]])
    for j in range(nrows + 2):
        pos[(0, j)] = np.array([pos[(1, j)][0] - spacing, y[j]])
        pos[(ncols + 1, j)] = np.array([pos[(ncols, j)][0] + spacing, y[j]])
    for j in range(nrows + 2):
        xs = [pos[(i, j)][0] for i in range(ncols + 2)]
        if any(xs[k + 1] <= xs[k] for k in range(len(xs) - 1)):
            raise ValueError(f"row {j} is not monotone in x; increase `spacing`")

    interior = [(i, j) for j in range(1, nrows + 1) for i in range(1, ncols + 1)]
    edges = []
    for (i, j) in interior:
        for nb in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            edges.append(((i, j), nb))
    P = from_vertices(pos, interior, edges,
                      name or f"flat-foldable quad {nrows}x{ncols} t={t}")
    P.meta.update(kind="ff_quad", nrows=nrows, ncols=ncols, t=t, heights=h)
    return P


# Displacements of the 16 lattice points of a 2x2 Miura patch that put the
# pattern exactly on the second-order foldability locus (one scalar condition
# for a 2x2 patch, matching Tachi's one condition per inner facet) while
# leaving it far from flat-foldable and with no straight fold line anywhere.
# Found by minimising min_sigma sigma_min(M_sigma) under those constraints.
_IMPOSTOR_DISP = np.array([
    0.40829439013129143, -0.5087217485581049, 0.08332093353816217, -0.11320442109541183,
    -0.0891853068219739, -0.04331502644538343, -0.4051499580725104, -0.047886995540463474,
    -0.17555101425319608, 0.6298271452856745, 0.0456876795634497, -0.06862519913568824,
    -0.05698422269173242, -0.14336931873346542, -0.2133696666866699, -0.07645674610306807,
    0.0964154854265162, -0.04747773319508809, 0.1951820482668702, -0.03991604710929879,
    0.004881566772363969, 0.29580766493083405, 0.1123249082529907, -0.10044809548199676,
    -0.03636751014345874, 0.10768192089889148, 0.38011279291614575, -0.05450483005104105,
    -0.04847418412162349, 0.20885283816101224, -0.177377038613454, -0.05799879288553374,
])


def second_order_impostor() -> Pattern:
    """A quad mesh whose tangent cone at the flat state is NONEMPTY and which is
    nevertheless NOT rigidly foldable.

    Every interior vertex fails Kawasaki by 26 to 76 degrees, no two creases at
    any vertex are within 9 degrees of collinear, and every sector is below 144
    degrees, so each vertex is individually foldable. The second-order condition
    is satisfied exactly, so `cone.decompose` reports a one-dimensional cell.
    But the flex does not integrate: correcting onto the constraint leaves a
    residual that scales as eps^4 instead of falling to machine precision, and
    every isolated far-field solution of Phi = I has local DOF 0.

    This is the standing counterexample to reading `decompose` as a branch
    count. Second-order foldability is necessary, not sufficient.
    """
    base = miura(2, 2)
    keys = sorted(base.pos.keys(), key=lambda k: (k[1], k[0]))
    pos = {k: base.pos[k] + _IMPOSTOR_DISP[2 * i:2 * i + 2] for i, k in enumerate(keys)}
    P = from_vertices(pos, base.interior, [tuple(k) for k in base.cidx],
                      "second-order impostor 2x2")
    P.meta.update(kind="impostor", note="nonempty tangent cone, not rigidly foldable")
    return P


# A 2x2 quad mesh that is genuinely 1-DOF rigidly foldable and has NO straight
# fold line: every vertex satisfies Kawasaki to 1e-14, Tachi's signed loop
# condition holds to machine precision, and the closest crease pair at any
# vertex is 12 degrees away from collinear. Obtained by solving Kawasaki at all
# four vertices together with prod k = +1 on the single inner facet, while
# pushing every crease pair away from collinear.
_GENERIC_RF_DISP = np.array([
    1.0185207388403277, -0.08539263369791651, 0.06176715669187905, 0.09180761381879117,
    -0.023040399638872992, -0.23457170313300357, 0.24718872720041196, 0.00821791582188276,
    -0.27952516502151, 0.465312920601526, -0.1049978863430043, -0.09600035946940001,
    -0.021119697918330044, 0.13751275227881857, -0.01894723258569206, 0.13986338825985029,
    0.03129436093622803, 0.06819489962371782, 0.08651009859987457, 0.46209795418005917,
    -0.047364013458137275, 0.00012565026869605902, -1.249946082031804, -0.43413472062355357,
    -0.12031460058833703, 0.11118591622070376, -0.2734359059877146, 0.047552884972357656,
    0.9070019329014242, 0.0062150063946387335, 0.20028672283057758, 0.46405313035043355,
])


def rigid_foldable_no_straight_lines() -> Pattern:
    """1-DOF rigidly foldable quad mesh with no straight fold line anywhere.

    This is the class that Miura-like reasoning does not reach: there is no
    collinear crease pair at any vertex, so the "zig-zag pair" that hypothesis
    H3 is phrased in terms of does not exist. Its tangent cone at the flat state
    has exactly one component (Miura 2x2 has four; the extra three come from the
    straight fold lines), and raising any vertex changes the branch count.
    """
    base = miura(2, 2)
    keys = sorted(base.pos.keys(), key=lambda k: (k[1], k[0]))
    pos = {k: base.pos[k] + _GENERIC_RF_DISP[2 * i:2 * i + 2] for i, k in enumerate(keys)}
    P = from_vertices(pos, base.interior, [tuple(k) for k in base.cidx],
                      "rigid-foldable quad, no straight fold line")
    P.meta.update(kind="generic_rf")
    return P


def raise_degree_by_index(P: Pattern, slot: int, order, reach: float = 0.62) -> Pattern:
    """Add one stub crease bisecting sector number i, for each i in `order`.

    Sectors are numbered CCW from the vertex's first crease, so the placement
    is a fixed combinatorial choice that does NOT move when the sector widths
    change. `raise_degree` instead bisects the currently WIDEST sector, which
    jumps discontinuously whenever two sectors are nearly equal: two positions
    a thousandth apart can then get combinatorially different patterns. Every
    sweep over vertex positions must use this function, or the sweep silently
    compares different patterns and reports position dependence that is really
    the placement rule flipping.

    A repeated index is REFUSED. Earlier versions accepted it and recomputed
    the same bisector, producing two coincident creases: a degenerate crease
    pattern rather than two stubs inside one sector. Use `raise_degree_at` with
    explicit fractions, for example [(0, 1/3), (0, 2/3)].
    """
    if len(set(order)) != len(tuple(order)):
        raise ValueError(
            "repeated sector index would place coincident creases; use "
            "raise_degree_at(P, slot, [(sector, fraction), ...]) instead")
    pos, cidx = dict(P.pos), dict(P.cidx)
    vdata = [list(it) for it in P.vdata]
    v = P.interior[slot]
    base = P.pos[v]
    it = vdata[slot]
    orig = sorted(t[0] for t in it)
    nS = len(orig)
    for i in order:
        a0 = orig[i % nS]
        a1 = orig[(i + 1) % nS]
        w = (a1 - a0) % (2 * np.pi)
        new = float((a0 + w / 2) % (2 * np.pi))
        e = len(cidx)
        stub = ("stub", slot, e)
        pos[stub] = base + reach * np.array([np.cos(new), np.sin(new)])
        cidx[frozenset((v, stub))] = e
        it.append((new, e, np.array([np.cos(new), np.sin(new)])))
    it.sort(key=lambda t: t[0])
    Q = Pattern(pos=pos, interior=list(P.interior), cidx=cidx, vdata=vdata,
                name=f"{P.name} + stub{tuple(order)}@{v}", meta=dict(P.meta))
    Q.meta.setdefault("raised_by_index", []).append((v, tuple(order)))
    return Q


def raise_degree_at(P: Pattern, slot: int, places, reach: float = 0.62) -> Pattern:
    """Implant stubs at explicit positions inside chosen sectors.

    `places` is a list of (sector, fraction) with fraction strictly between 0
    and 1: the stub goes at angle a0 + fraction * (sector width), where a0 is
    the start of that sector in the ORIGINAL vertex, numbered CCW from the
    first crease. Two stubs in the same sector are written [(0, 1/3), (0, 2/3)]
    and are genuinely distinct creases.

    This replaces the repeated-index spelling accepted by
    `raise_degree_by_index`, which recomputed the same bisector twice and so
    produced two COINCIDENT creases, a degenerate crease pattern rather than
    two stubs in one sector. Counts measured that way describe the doubled
    crease, not the pattern the notation suggests.
    """
    pos, cidx = dict(P.pos), dict(P.cidx)
    vdata = [list(it) for it in P.vdata]
    v = P.interior[slot]
    base = P.pos[v]
    it = vdata[slot]
    orig = sorted(t[0] for t in it)
    nS = len(orig)
    seen = set()
    for sec, frac in places:
        if not (0 < frac < 1):
            raise ValueError("fraction must be strictly between 0 and 1")
        a0 = orig[sec % nS]
        a1 = orig[(sec + 1) % nS]
        w = (a1 - a0) % (2 * np.pi)
        new = float((a0 + frac * w) % (2 * np.pi))
        if any(abs((new - x) % (2 * np.pi)) < 1e-12 or
               abs((x - new) % (2 * np.pi)) < 1e-12 for x in seen):
            raise ValueError("coincident stub directions")
        seen.add(new)
        e = len(cidx)
        stub = ("stub", slot, e)
        pos[stub] = base + reach * np.array([np.cos(new), np.sin(new)])
        cidx[frozenset((v, stub))] = e
        it.append((new, e, np.array([np.cos(new), np.sin(new)])))
    it.sort(key=lambda t: t[0])
    Q = Pattern(pos=pos, interior=list(P.interior), cidx=cidx, vdata=vdata,
                name=f"{P.name} + stubs{tuple(places)}@{v}", meta=dict(P.meta))
    Q.meta.setdefault("raised_at", []).append((v, tuple(places)))
    return Q
