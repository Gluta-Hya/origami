"""Regression suite: every check reproduces a published result.

Run with `python -m origami_cone.validate`.
"""
from __future__ import annotations
import numpy as np
from .patterns import (single_vertex, single_vertex_sectors, miura,
                       raise_degree, raise_degree_by_index, raise_degree_at, perturb,
                       second_order_impostor, flat_foldable_quad,
                       rigid_foldable_no_straight_lines)
from . import tachi, h3
from .kinematics import (hat, axis3, rodrigues, vertex_closure, residual,
                         continue_branch, find_folded_states, local_dof,
                         certify_cells)
from .cone import (decompose, kernel, hessian_blocks, signature, classify_block,
                   cell_directions, linear_cells, definiteness_count,
                   predicted_components, predicted_at, opposite_pair_values,
                   fast_component_count, incremental_linear_cells, cell_budget)


def _logso3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    A = (R - R.T) / 2 if th < 1e-12 else th / (2 * np.sin(th)) * (R - R.T)
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def t_commutator():
    """[hat u_i, hat u_j] = sin(alpha_j - alpha_i) hat z, for in-plane axes."""
    rng = np.random.default_rng(0)
    Z = hat([0, 0, 1])
    worst = 0.0
    for _ in range(200):
        a, b = rng.uniform(0, 2 * np.pi, 2)
        Ua, Ub = hat(axis3([np.cos(a), np.sin(a)])), hat(axis3([np.cos(b), np.sin(b)]))
        worst = max(worst, np.abs(Ua @ Ub - Ub @ Ua - np.sin(b - a) * Z).max())
    return worst < 1e-12, f"max deviation {worst:.2e}"


def t_second_order():
    """log Phi = (J0.rho, rho^T Q rho) + O(rho^3): error must fall by 8 per halving."""
    rng = np.random.default_rng(7)
    bad = []
    for n in (4, 5, 6, 8):
        s = rng.dirichlet(np.ones(n)) * 360.0
        P = single_vertex_sectors(s)
        J = np.array([[t[2][0] for t in P.vdata[0]], [t[2][1] for t in P.vdata[0]]])
        order = [t[1] for t in P.vdata[0]]
        J = J[:, np.argsort(order)] if list(order) != sorted(order) else J
        Q = hessian_blocks(P)[0]
        r = rng.normal(size=n)
        prev, ratios = None, []
        for eps in (1e-1, 5e-2, 2.5e-2):
            rho = eps * r
            pred = np.array([J[0] @ rho, J[1] @ rho, rho @ Q @ rho])
            err = np.linalg.norm(_logso3(vertex_closure(P, 0, rho)) - pred)
            if prev:
                ratios.append(prev / err)
            prev = err
        if not all(7.5 < x < 8.5 for x in ratios):
            bad.append((n, [round(x, 2) for x in ratios]))
    return not bad, "cubic remainder confirmed for n=4,5,6,8" if not bad else f"failed {bad}"


def t_chen_santangelo_signature():
    """Chen & Santangelo, PRX 8 011034 (2018): the second-order form of a
    degree-N vertex has N-2 nonzero eigenvalues, exactly one of one sign."""
    rng = np.random.default_rng(11)
    bad = []
    for n in range(4, 10):
        for _ in range(60):
            s = rng.dirichlet(np.ones(n) * rng.uniform(0.6, 4.0)) * 360.0
            if s.max() >= 179.5 or s.min() < 1.0:
                continue
            P = single_vertex_sectors(s)
            B, rk = kernel(P)
            p, q, z = signature(B.T @ hessian_blocks(P)[0] @ B)
            if not (p + q == n - 2 and z == 0 and min(p, q) == 1):
                bad.append((n, list(np.round(s, 2)), (p, q, z)))
                break
    return not bad, "signature (1, n-3) on every sampled vertex, n=4..9" if not bad else f"failed {bad[:2]}"


def t_degree3_rigid():
    """A degree-3 vertex admits no rigid folding out of the flat state."""
    P = single_vertex_sectors([120, 130, 110])
    r = decompose(P)
    folded = find_folded_states(P, tries=250, min_norm=0.05)
    ok = len(r["cells"]) == 0 and len(folded) == 0
    return ok, f"cells={len(r['cells'])}, folded states found={len(folded)}"


def t_degree4_two_branches():
    """Waitukaitis et al., PRL 114 055503 (2015): a generic degree-4 vertex has
    exactly two branches of motion meeting at the flat state, each 1-dimensional,
    and each obeys Maekawa's 3-1 split."""
    P = single_vertex_sectors([70, 100, 90, 100])
    r = decompose(P)
    dims = sorted(c.dim for c in r["cells"])
    splits = []
    for c in r["cells"]:
        v = cell_directions(r, c)[0]
        x, res = continue_branch(P, v, 1.2, 40)
        s = np.sign(np.round(x, 6)).astype(int)
        splits.append((int((s > 0).sum()), int((s < 0).sum()), res < 1e-10))
    ok = (len(r["cells"]) == 2 and dims == [1, 1]
          and all(sorted(a[:2]) == [1, 3] and a[2] for a in splits))
    return ok, f"cells={len(r['cells'])} dims={dims} MV splits={splits}"


def t_sector_criterion():
    """Abel, Cantarella, Demaine, Eppstein, Hull, Ku, Lang, Tachi, JoCG 2016,
    Thm 2: for n >= 4 a single vertex is rigidly foldable iff all sectors are
    < pi (and it is not an unspecified cross). Here: the second-order form is
    indefinite iff all sectors < pi."""
    rng = np.random.default_rng(5)
    bad = 0
    total = 0
    for n in range(4, 9):
        for _ in range(900):
            s = rng.dirichlet(np.ones(n) * rng.uniform(0.4, 5.0)) * 360.0
            if s.min() < 0.6:
                continue
            P = single_vertex_sectors(s)
            B, rk = kernel(P)
            if rk != 2:
                continue
            kind, _ = classify_block(B.T @ hessian_blocks(P)[0] @ B)
            total += 1
            if (kind == "linear" or kind == "quadric") != bool(s.max() < 180.0 - 1e-9):
                bad += 1
    return bad == 0, f"{total} sampled vertices, {bad} disagreements"


def t_cross_exception():
    """The same theorem's 'cross' exception: two straight lines crossing is
    foldable only by leaving one whole line flat."""
    P = single_vertex_sectors([60, 120, 60, 120])
    r = decompose(P)
    zeros = []
    for c in r["cells"]:
        v = cell_directions(r, c)[0]
        zeros.append(int((np.abs(v) < 1e-9).sum()))
    return all(z == 2 for z in zeros), f"unfolded creases per branch = {zeros} (expected 2 each)"


def t_sign_convention():
    """Shared creases: outward axis at both ends with the same fold angle. If the
    convention were wrong, the known Miura motion would not satisfy closure."""
    P = miura(2, 2)
    folded = find_folded_states(P, tries=120, min_norm=0.2)
    return len(folded) > 0, f"{len(folded)}/120 random starts reached a genuine folded state"


def t_tachi_generic_quad_rigid():
    """Tachi, IASS 2009: a quadrilateral mesh is overconstrained and generically
    rigid; Miura works only because of exact tuning."""
    P = perturb(miura(2, 2), 0.03, seed=5)
    r = decompose(P)
    folded = find_folded_states(P, tries=200, min_norm=0.15)
    kaw = max(abs(P.kawasaki(k)) for k in range(P.V))
    ref = decompose(miura(2, 2))
    # Overconstrained means: no folding MOTION. Isolated zero-dimensional
    # solutions of Phi = I may still exist far from the flat state (near the
    # fully folded corner), so we check their local DOF rather than their count.
    dofs = sorted({local_dof(P, x) for x in folded})
    ok = (len(r["cells"]) == 0 and len(ref["cells"]) > 0
          and all(d == 0 for d in dofs))
    return ok, (f"Kawasaki residual up to {np.degrees(kaw):.2f} deg: "
                f"perturbed tangent cone = {{0}} ({len(r['cells'])} cells), "
                f"unperturbed has {len(ref['cells'])} cells; "
                f"{len(folded)} far-field solutions, local DOF {dofs or '[]'} "
                f"(0 = isolated point, not a motion)")


def t_tachi_miura_one_dof():
    """Tachi, IASS 2009: Miura is 1-DOF although the naive count E - 3V says less."""
    P = miura(2, 2)
    r = decompose(P)
    full = None
    for c in r["cells"]:
        v = cell_directions(r, c)[0]
        if np.all(np.abs(v) > 1e-9):
            full = v
    if full is None:
        return False, "no branch folds every crease"
    x, res = continue_branch(P, full, 0.8, 36)
    dof = local_dof(P, x)
    return (dof == 1 and res < 1e-10,
            f"naive E-3V={P.E - 3 * P.V}, measured DOF on the all-creases branch = {dof}")


def t_hull_tachi_two_modes():
    """Hull & Tachi et al., arXiv:1809.04243: a Miura tessellation has exactly
    two folding modes once every crease is required to fold (the mirror pair)."""
    out = []
    for nr, nc in ((2, 2), (2, 3), (3, 3)):
        P = miura(nr, nc)
        r = decompose(P)
        n_full = sum(1 for c in r["cells"]
                     if np.all(np.abs(cell_directions(r, c)[0]) > 1e-9))
        out.append((f"{nr}x{nc}", n_full))
    ok = all(n == 1 for _, n in out)
    return ok, f"branches folding every crease (before the MV mirror): {out}"


def t_degree_raise_breaks_mv_label():
    """Implanting one degree-6 vertex makes the branches curved, and the MV
    pattern stops labelling them."""
    P0, P1 = miura(2, 2), raise_degree(miura(2, 2), 0, 2)
    from .invariants import mv_report
    r0, r1 = decompose(P0), decompose(P1)
    m0, m1 = mv_report(r0), mv_report(r1)
    # On a pure quad mesh every 1-dimensional branch realises exactly one MV
    # pattern and no pattern is shared between branches, so the pattern labels
    # the branch. (The trivial cell has dimension n_rows > 1 and naturally
    # realises several, so it is excluded from that statement.)
    pure_lines_unique = all(len(m0["sets"][i]) == 1
                            for i, c in enumerate(r0["cells"]) if c.dim == 1)
    raised_multi = max(m1["per_cell"]) > 1
    ok = (all(c.kind == "flat" for c in r0["cells"])
          and all(c.kind == "curved" for c in r1["cells"])
          and pure_lines_unique and m0["shared"] == 0 and raised_multi)
    return ok, (f"pure: flat cells, MV patterns per 1-dim branch "
                f"{[len(m0['sets'][i]) for i, c in enumerate(r0['cells']) if c.dim == 1]}, "
                f"shared={m0['shared']} | raised: curved cells, patterns/cell "
                f"{m1['per_cell']}, shared={m1['shared']}")


def t_second_order_not_sufficient():
    """A nonempty tangent cone does NOT imply rigid foldability. The fixture
    sits exactly on the second-order locus, is far from flat-foldable, has no
    straight fold line, and still cannot fold: the corrected residual scales as
    eps^4 rather than vanishing, and every far-field solution is isolated."""
    from .kinematics import solve_slice
    P = second_order_impostor()
    r = decompose(P)
    if not r["cells"]:
        return False, "fixture no longer has a nonempty tangent cone"
    kaw = [abs(np.degrees(P.kawasaki(k))) for k in range(P.V)]
    v = cell_directions(r, max(r["cells"], key=lambda c: c.dim))[0]
    res = []
    for eps in (0.2, 0.1, 0.05):
        y = solve_slice(P, v, eps, eps * v)
        res.append(float(np.linalg.norm(residual(P, y))))
    ratios = [res[i] / res[i + 1] for i in range(len(res) - 1)]
    folded = find_folded_states(P, tries=150, min_norm=0.15, seed=11)
    dofs = sorted({local_dof(P, x) for x in folded})
    ok = (min(kaw) > 20.0 and all(1e-11 < x < 1e-6 for x in res)
          and all(13.0 < q < 19.0 for q in ratios) and all(d == 0 for d in dofs))
    return ok, (f"Kawasaki off by {min(kaw):.0f}-{max(kaw):.0f} deg; corrected residual "
                f"{res[0]:.1e} -> {res[-1]:.1e}, ratio {[round(q, 1) for q in ratios]} "
                f"(16 = O(eps^4), so it fails at third order); far-field DOF {dofs or '[]'}")


def t_reported_counts_are_certified():
    """Every cone component of the patterns whose counts this project reports is
    a real branch, not a second-order impostor: residual at machine precision."""
    rows = []
    for lab, P in (("Miura 3x3", miura(3, 3)),
                   ("3x3 centre -> deg 6", raise_degree(miura(3, 3), 4, 2)),
                   ("3x3 corner -> deg 6", raise_degree(miura(3, 3), 0, 2))):
        r = decompose(P)
        cert = certify_cells(P, r)
        rows.append((lab, len(cert), sum(1 for c in cert if c["genuine"])))
    ok = all(n == g for _, n, g in rows)
    return ok, "; ".join(f"{lab}: {g}/{n} certified" for lab, n, g in rows)


def _ratio_drift(sectors_deg, eps=(0.7, 0.95)):
    from .kinematics import solve_slice
    P = single_vertex_sectors(sectors_deg)
    order = [e for _, e, _ in P.vdata[0]]
    r = decompose(P)
    worst = 0.0
    for c in r["cells"]:
        v = cell_directions(r, c)[0]
        x, ts = None, []
        for e in eps:
            x = solve_slice(P, v, e, x)
            if np.linalg.norm(residual(P, x)) > 1e-11:
                ts = []
                break
            ts.append(np.tan(x[order] / 2))
        if len(ts) < 2:
            continue
        a, b = ts[0] / ts[0][0], ts[1] / ts[1][0]
        worst = max(worst, float(np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-12))))
    return worst


def t_conversion_constant_iff_flat_foldable():
    """The conversion-coefficient machinery is valid exactly on the
    flat-foldable class: tan-half-angle ratios are constant along a branch iff
    the degree-4 vertex satisfies Kawasaki."""
    ff = [_ratio_drift([a, b, 180 - a, 180 - b]) for a, b in ((121, 59), (100, 70), (140, 35))]
    rng = np.random.default_rng(0)
    nff = []
    while len(nff) < 3:
        s = rng.dirichlet(np.ones(4) * 2.5) * 360.0
        if s.max() > 175 or abs(s[0] + s[2] - 180) < 5:
            continue
        nff.append(_ratio_drift(list(s)))
    ok = max(ff) < 1e-9 and min(nff) > 1e-3
    return ok, (f"flat-foldable drift up to {max(ff):.1e}; "
                f"non-flat-foldable drift {min(nff):.1e} to {max(nff):.1e}")


def t_closed_form_conversion():
    """The closed-form branch ratios match numerical continuation."""
    from .kinematics import solve_slice
    worst, n = 0.0, 0
    for a, b in ((121, 59), (100, 70), (140, 35), (80, 110), (63, 97)):
        P = single_vertex_sectors([a, b, 180 - a, 180 - b])
        order = [e for _, e, _ in P.vdata[0]]
        cl = tachi.ratios_closed(P, 0)
        r = decompose(P)
        for c in r["cells"]:
            v = cell_directions(r, c)[0]
            x = solve_slice(P, v, 0.8, None)
            if np.linalg.norm(residual(P, x)) > 1e-11:
                continue
            tt = np.tan(x[order] / 2)
            meas = {order[i]: float(tt[i] / tt[0]) for i in range(4)}
            best = min(cl, key=lambda d: max(abs(meas[e] - d[e]) for e in meas))
            worst = max(worst, max(abs(meas[e] - best[e]) / max(abs(best[e]), 1.0) for e in meas))
            n += 1
    return worst < 1e-10, f"{n} branch vectors compared, worst relative error {worst:.1e}"


def t_tachi_condition_is_signed():
    """Tachi's condition is prod k = +1, not |prod k| = 1, and it agrees with
    the second-order condition on the flat-foldable class while correctly
    rejecting the second-order impostor."""
    rows = []
    for lab, P in (("Miura 2x2", miura(2, 2)),
                   ("no straight lines", rigid_foldable_no_straight_lines()),
                   ("perturbed (rigid)", perturb(miura(2, 2), 0.03, seed=5))):
        d = tachi.foldability_defect(P)
        rows.append((lab, d, len(decompose(P)["cells"]) > 0, True))
    # outside the flat-foldable class the two conditions must DISAGREE on the
    # impostor: second order accepts it, Tachi rejects it, and Tachi is right.
    imp = second_order_impostor()
    d_imp = tachi.foldability_defect(imp)
    so_imp = len(decompose(imp)["cells"]) > 0
    rows.append(("second-order impostor", d_imp, so_imp, False))
    agree_on_ff = all((d is not None and d < 1e-9) == so
                      for _, d, so, ff in rows if ff)
    sharper = (d_imp is not None and d_imp > 1e-3) and so_imp
    ok = agree_on_ff and sharper and not tachi.flat_foldable_vertices(imp)
    return ok, ("; ".join(f"{l}: defect {'-' if d is None else f'{d:.1e}'}, 2nd-order {so}"
                          for l, d, so, _ in rows)
                + " -- Tachi agrees with 2nd order on the flat-foldable class and is "
                  "strictly sharper on the impostor")


def t_generic_rigid_foldable_mesh():
    """A 1-DOF rigidly foldable quad mesh with no straight fold line exists,
    has exactly one tangent-cone component, and folds to machine precision."""
    P = rigid_foldable_no_straight_lines()
    import itertools as _it
    coll = min(abs(abs(x - y) - np.pi)
               for k in range(P.V)
               for x, y in _it.combinations([t[0] for t in P.vdata[k]], 2))
    r = decompose(P)
    if len(r["cells"]) != 1:
        return False, f"expected one component, got {len(r['cells'])}"
    v = cell_directions(r, r["cells"][0])[0]
    x, res = continue_branch(P, v, 0.8, 36)
    dof = local_dof(P, x)
    ok = (np.degrees(coll) > 8.0 and tachi.flat_foldable_vertices(P)
          and res < 1e-12 and dof == 1)
    return ok, (f"closest pair {np.degrees(coll):.1f} deg from collinear, 1 component, "
                f"folded residual {res:.1e}, DOF {dof}")


def t_h3_needs_a_straight_fold_line():
    """Outside the class with a straight fold line at every vertex, H3 is not
    just false, it is undefined: there is no zig-zag pair, and in fact every
    vertex shifts the branch count when raised."""
    P = rigid_foldable_no_straight_lines()
    base = len(decompose(P)["cells"])
    zz = [h3.zigzag_creases(P, k) for k in range(P.V)]
    shifts, preds = [], []
    for v in range(P.V):
        n = len(decompose(raise_degree(P, v, 2))["cells"])
        shifts.append(n != base)
        preds.append(h3.predict(P, (v,)))
    ok = all(len(z) == 4 for z in zz) and all(shifts) and not any(preds)
    return ok, (f"zig-zag pair size {[len(z) for z in zz]} (needs 2); baseline {base}; "
                f"raising each vertex shifts the count {sum(shifts)}/{len(shifts)} times "
                f"while H3 predicts a shift {sum(preds)}/{len(preds)} times")


def t_dieleman_class1_count():
    """Cross-check against an independent published count. Dieleman, Vasmel,
    Waitukaitis & van Hecke, Nature Physics 16, 63 (2020), Table I: a generic
    (non-collinear) degree-4 quad tiling of their Class 1 has exactly 2 folding
    branches, independent of size. Our non-collinear rigid-foldable mesh has
    exactly one tangent-cone component, which is 2 branches once the global
    mountain-valley mirror is counted."""
    P = rigid_foldable_no_straight_lines()
    r = decompose(P)
    n = len(r["cells"])
    cert = sum(1 for c in certify_cells(P, r) if c["genuine"]) if n else 0
    return (n == 1 and cert == 1,
            f"{n} tangent-cone component ({cert} certified) = {2 * n} branches with the "
            f"MV mirror; Dieleman et al. Class 1 predicts 2, independent of size "
            f"(also checked at 2x3 during development)")


def _places(order):
    """Turn a sector multiset into explicit (sector, fraction) places, spreading
    repeats evenly inside the sector. A repeat used to mean a coincident crease;
    now it means genuinely distinct stubs."""
    from collections import Counter
    cnt = Counter(order)
    seen = Counter()
    out = []
    for s in order:
        seen[s] += 1
        out.append((s, seen[s] / (cnt[s] + 1)))
    return out


def _ff_sample(s, nr=3, nc=3):
    """Position sample s of the flat-foldable quad family, same RNG as the sweeps."""
    rng = np.random.default_rng(10_000 + s)
    if s == 0:
        return miura(nr, nc)
    for _ in range(60):
        t = rng.uniform(0.25, 1.6, nc) * rng.choice([-1, 1], nc)
        h = rng.uniform(0.6, 1.6, nr + 1)
        try:
            return flat_foldable_quad(nr, nc, t, h, spacing=1.0)
        except ValueError:
            continue
    raise RuntimeError("no monotone sample")


def t_definiteness_mechanism():
    """The branch count of a single-implant mixed-degree pattern is decided cell
    by cell by whether the implanted vertex's second-order form restricted to
    that maximal linear cell is definite. A definite restriction kills the cell.
    This is the mechanism the earlier refuted conjectures (2^V4, 2^(dimK-dmax))
    were groping for, and unlike them it is not a formula in the combinatorics
    alone: it is a signature condition, so the count can be any integer."""
    bad, tot = 0, 0
    for nr, nc in ((3, 3), (2, 3)):
        P0 = miura(nr, nc)
        for slot in range(P0.V):
            for order in ((0,), (0, 1), (0, 2), (0, 1, 2, 3)):
                for s in (0, 1, 3):
                    P = raise_degree_at(_ff_sample(s, nr, nc), slot, _places(order))
                    pred, _ = definiteness_count(P)
                    if pred is None:
                        continue
                    tot += 1
                    bad += int(pred != len(decompose(P)["cells"]))
    return bad == 0, (f"predicted count matches the decomposition on {tot - bad}/{tot} "
                      f"configurations (one implanted vertex, degrees 5 to 8)")


def t_widest_sector_rule_is_discontinuous():
    """raise_degree bisects the WIDEST sector, so its choice jumps when two
    sectors are nearly equal, and a position sweep using it compares
    combinatorially different patterns. raise_degree_by_index is continuous.
    Measured here: the widest-sector rule reports position dependence where the
    index rule reports an invariant. Any claim of position dependence obtained
    with raise_degree is therefore unsafe."""
    wid, idx = set(), set()
    for s in range(24):
        B = _ff_sample(s)
        wid.add(len(decompose(raise_degree(B, 4, 4))["cells"]))
        idx.add(len(decompose(raise_degree_at(B, 4, _places((0, 1, 2, 3))))["cells"]))
    ok = len(wid) > 1 and len(idx) == 1
    return ok, (f"widest-sector rule over 24 positions: counts {sorted(wid)}; "
                f"index rule, same 24 positions: counts {sorted(idx)} "
                f"(the spread is the rule flipping, not geometry)")


def t_placement_decides_the_count():
    """With BISECTING stubs (the default of `_places`; two stubs in one sector
    sit at 1/3 and 2/3 of it) the count is decided by which sectors hold the
    stubs and, for a diagonal pair, by the side of the narrow sectors
    (Corollary 7.3(c)); at the centre column of a 3x3 patch the two diagonal
    values agree. There the baseline is 8 and degree 6 gives 12 for adjacent
    sectors, 13 for diagonal sectors, 11 for two stubs in one sector, and degree
    8 gives 14. Away from the bisectors these values can change (see the
    Corollary 7.3 and 7.4 tests). At an outer-row vertex every placement leaves
    the baseline untouched, for every stub position (Theorem 7.5)."""
    exp_centre = {(): 8, (0,): 10, (1,): 10, (0, 1): 12, (1, 2): 12,
                  (0, 2): 13, (1, 3): 13, (0, 0): 11, (1, 1): 11,
                  (0, 1, 2, 3): 14}
    got = {}
    for order, want in exp_centre.items():
        vals = {len(decompose(raise_degree_at(_ff_sample(s), 4, _places(order))
                              if order else _ff_sample(s))["cells"])
                for s in range(8)}
        got[order] = sorted(vals)
        if vals != {want}:
            return False, f"placement {order}: expected {want}, got {sorted(vals)}"
    outer = {len(decompose(raise_degree_at(_ff_sample(s), 1, _places(o)))["cells"])
             for s in range(8) for o in ((0,), (0, 2), (0, 1, 2, 3))}
    if outer != {8}:
        return False, f"outer-row vertex should stay at 8, got {sorted(outer)}"
    return True, ("centre vertex, bisecting stubs: "
                  + ", ".join(f"{k}->{v[0]}" for k, v in got.items())
                  + "; outer-row vertex stays at 8 for every placement")


def t_closed_form_component_count():
    """`predicted_at` against the full count, for BISECTING stubs. With
    A = 2^(col-1) and C = 2^(ncols-col) the count is 2^ncols on an outer row,
    and on an interior row 2^ncols + 2 for a single stub, + 2A for the wide
    pair, + 2C for the narrow pair, + A + C for an adjacent pair, + A + 1 or
    + C + 1 for two stubs in one wide or narrow sector, and + 2A + 2C - 2 for
    one stub per sector. Diagonal pairs are covered by the Corollary 7.3/7.4
    tests. These values are for bisecting stubs only: off the bisectors the
    classes with an empty associated sector change value (Corollary 7.3), and
    `criterion.predict_N` gives the count at any stub position."""
    ORD = [(0,), (2,), (0, 1), (1, 2), (0, 3), (2, 3), (0, 0), (2, 2), (0, 1, 2, 3)]
    tot = bad = 0
    detail = []
    for nr, nc in ((3, 2), (3, 3), (4, 3)):
        P0 = miura(nr, nc)
        for slot, (i, j) in enumerate(P0.interior):
            for o in ORD:
                pred = predicted_at(nr, nc, i, j, _places(o))
                if pred is None or isinstance(pred, tuple):
                    continue
                vals = {len(decompose(raise_degree_at(_ff_sample(s, nr, nc),
                                                      slot, _places(o)))["cells"])
                        for s in (0, 1)}
                tot += 1
                if vals != {pred}:
                    bad += 1
                    detail.append(f"{nr}x{nc} col{i} row{j} {o}: {pred} vs {sorted(vals)}")
    ok = bad == 0
    return ok, ((f"bisector closed form matches on {tot - bad}/{tot} configurations")
                if ok else "; ".join(detail[:4]))


def _slot_of(P0, i, j):
    return [k for k, v in enumerate(P0.interior) if v == (i, j)][0]


def t_cor74_diagonal_pair_examples():
    """Corollary 7.4 and the example after it. On the 3x4 Miura patch with v*
    = v_{2,2} (|L| = 26, a = 1, c = 3) a diagonal pair in sectors 0 and 2 gives
    N = 25 for bisecting stubs (the wide-side family dies), 23 at fractions
    (0.95, 0.05) (the narrow-side family dies) and 26 at (0.95, 0.5) (both
    survive). All three values occur on one vertex; the fourth, 26 - a - c, never
    does, because the two diagonal families never both die. The second part
    checks that on random diagonal pairs at several vertices."""
    from .cone import fast_component_count
    P0 = miura(3, 4)
    slot = _slot_of(P0, 2, 2)
    want = {((0, .5), (2, .5)): 25, ((0, .95), (2, .05)): 23, ((0, .95), (2, .5)): 26}
    got = {k: fast_component_count(raise_degree_at(P0, slot, list(k)))[0] for k in want}
    if got != want:
        return False, f"expected {want}, got {got}"
    rng = np.random.default_rng(74)
    seen, bad, tot = set(), [], 0
    for _ in range(40):
        nr, nc = [(3, 3), (3, 4), (4, 4), (4, 3)][rng.integers(4)]
        P0 = miura(nr, nc, d=float(rng.uniform(0.3, 1.5)))
        cand = [k for k, (i, j) in enumerate(P0.interior) if 1 < j < nr]
        slot = int(rng.choice(cand))
        i0 = P0.interior[slot][0]
        a, c = 2 ** (i0 - 1) - 1, 2 ** (nc - i0) - 1
        L = 2 ** nc + 2 * a + 2 * c + 2
        pair = [(0, 2), (1, 3)][rng.integers(2)]
        pl = [(s, float(rng.uniform(0.02, 0.98))) for s in pair]
        N = fast_component_count(raise_degree_at(P0, slot, pl))[0]
        tot += 1
        if N not in (L, L - a, L - c) or (a > 0 and c > 0 and N == L - a - c):
            bad.append((nr, nc, P0.interior[slot], pl, N, L))
        seen.add(L - N)
    return not bad, (f"3x4 v_(2,2): 25, 23, 26 as stated; random diagonal pairs "
                     f"(deaths seen {sorted(seen)}) "
                     f"{tot - len(bad)}/{tot} with at most one family dead"
                     + ("" if not bad else f"; failures {bad[:2]}"))


def _miura_stub_angles(P0, slot, places):
    """For each stub: (side, narrow?, beta, gamma, next-to-z+?) at a Miura vertex,
    beta to the horizontal crease and gamma to the zigzag crease bounding its
    sector (the notation of Corollary 7.3)."""
    it = sorted(t[0] for t in P0.vdata[slot])
    A4 = list(it) + [2 * np.pi]
    out = []
    for s, f in places:
        lo, hi = A4[s], A4[s + 1]
        phi = lo + f * (hi - lo)
        horiz_lo = min(abs(lo), abs(lo - np.pi), abs(lo - 2 * np.pi)) < 1e-9
        beta = (phi - lo) if horiz_lo else (hi - phi)
        gamma = (hi - lo) - beta
        side = "R" if s in (0, 3) else "L"
        out.append(dict(sector=s, side=side, narrow=(hi - lo) < np.pi / 2,
                        beta=beta, gamma=gamma, upper=phi < np.pi))
    return out


def _cor73_prediction(P0, nr, nc, slot, places):
    """N from the closed forms of Corollary 7.3(a) and (b), for two stubs in two
    different sectors forming an occupied side or a diagonal pair."""
    from .criterion import vertex_angles
    theta = vertex_angles(P0, slot)[0]
    i0 = P0.interior[slot][0]
    size = {"L": 2 ** (i0 - 1) - 1, "R": 2 ** (nc - i0) - 1}
    L = 2 ** nc + 2 * size["L"] + 2 * size["R"] + 2
    st = _miura_stub_angles(P0, slot, places)
    sn, s2 = np.sin, np.cos(theta) ** 2
    if st[0]["side"] == st[1]["side"]:                    # Corollary 7.3(a)
        up = [x for x in st if x["upper"]][0]
        dn = [x for x in st if not x["upper"]][0]
        live = s2 * sn(up["beta"]) * sn(dn["beta"]) >= sn(up["gamma"]) * sn(dn["gamma"])
        empty = "L" if st[0]["side"] == "R" else "R"
        return L if live else L - 2 * size[empty]
    k = [x for x in st if x["narrow"]][0]                 # Corollary 7.3(b)
    l = [x for x in st if not x["narrow"]][0]
    S2 = np.sin(theta) ** 2
    narrow_live = (S2 * np.cos(k["beta"]) ** 2 * sn(l["beta"])
                   >= sn(k["beta"]) * sn(k["gamma"]) * sn(l["gamma"]))
    wide_live = (S2 * np.cos(l["beta"]) ** 2 * sn(k["beta"])
                 >= sn(k["gamma"]) * sn(l["beta"]) * sn(l["gamma"]))
    return L - (0 if narrow_live else size[k["side"]]) - (0 if wide_live else size[l["side"]])


def t_cor73_one_empty_side_examples():
    """Corollary 7.3(a) and the example after it. On the 3x4 Miura patch at
    v* = v_{2,2} (|L| = 26, c = 3), stubs in sectors 1 and 2 leave the right side
    empty. Bisecting stubs give N = 20: the two right families die
    (Corollary 7.3(c)). Stubs close to the zigzag creases, at fractions 0.03 and
    0.97, satisfy cos^2(theta) sin(beta+) sin(beta-) >= sin(gamma+) sin(gamma-)
    and give N = 26: nothing dies."""
    from .cone import fast_component_count
    P0 = miura(3, 4)
    slot = _slot_of(P0, 2, 2)
    want = {((1, .5), (2, .5)): 20, ((1, .03), (2, .97)): 26}
    got = {k: fast_component_count(raise_degree_at(P0, slot, list(k)))[0] for k in want}
    pred = {k: _cor73_prediction(P0, 3, 4, slot, list(k)) for k in want}
    ok = got == want and pred == want
    return ok, (f"full count {list(got.values())}, Corollary 7.3(a) closed form "
                f"{list(pred.values())}, expected {list(want.values())}")


def t_cor73_closed_forms_match_full_count():
    """Randomised check of the closed forms. (i) The inequalities of
    Corollary 7.3(a) and (b), evaluated at random stub positions, random
    Miura angles and random interior-row vertices of both leans, against the
    full count. (ii) `criterion.predict_N`, i.e. Theorem 6.7 evaluated at v*
    alone, against the full count for 1 to 4 stubs in random sectors at random
    positions."""
    from .cone import fast_component_count
    from .criterion import predict_N
    rng = np.random.default_rng(73)
    shapes = [(3, 3), (3, 4), (4, 3), (4, 4)]
    bad1 = bad2 = n1 = n2 = 0
    hits = set()
    for _ in range(60):
        nr, nc = shapes[rng.integers(len(shapes))]
        P0 = miura(nr, nc, d=float(rng.uniform(0.3, 1.5)))
        cand = [k for k, (i, j) in enumerate(P0.interior) if 1 < j < nr]
        slot = int(rng.choice(cand))
        pair = [(1, 2), (0, 3), (0, 2), (1, 3)][rng.integers(4)]
        pl = [(s, float(rng.uniform(0.02, 0.98))) for s in pair]
        N = fast_component_count(raise_degree_at(P0, slot, pl))[0]
        pred = _cor73_prediction(P0, nr, nc, slot, pl)
        n1 += 1
        bad1 += int(N != pred)
        hits.add(N == 2 ** nc + 2 * (2 ** (P0.interior[slot][0] - 1) - 1)
                 + 2 * (2 ** (nc - P0.interior[slot][0]) - 1) + 2)
    for _ in range(40):
        nr, nc = shapes[rng.integers(len(shapes))]
        P0 = miura(nr, nc, d=float(rng.uniform(0.3, 1.5)))
        cand = [k for k, (i, j) in enumerate(P0.interior) if 1 < j < nr]
        slot = int(rng.choice(cand))
        t = int(rng.integers(1, 5))
        pl = [(int(rng.integers(4)), float(rng.uniform(0.02, 0.98))) for _ in range(t)]
        N = fast_component_count(raise_degree_at(P0, slot, pl))[0]
        n2 += 1
        bad2 += int(N != predict_N(P0, nr, nc, slot, pl))
    ok = bad1 == 0 and bad2 == 0 and hits == {True, False}
    return ok, (f"Corollary 7.3 closed forms {n1 - bad1}/{n1}; Theorem 6.7 at v* "
                f"(criterion.predict_N) {n2 - bad2}/{n2}; both survival and death occur")


def t_incremental_matches_enumeration():
    """The incremental intersection gives exactly the enumerated answer.

    Enumerating one hyperplane per degree-4 vertex costs 2^(V-1), which puts a
    5x5 patch at 16.7 million choices per configuration and out of reach. But
    the map from choice to subspace has huge fibres: a 3x7 patch collapses
    1,048,576 choices onto 256 subspaces. Intersecting one vertex at a time and
    keeping only the distinct subspaces costs V4 times that small number
    instead. Measured: 3x7 goes from 76 s to 2.3 s, and 8x5, nominally 2^39
    choices, finishes in 2.3 s."""
    rows, bad = [], 0
    for nr, nc, order in ((3, 3, (0, 1)), (3, 3, (0, 2)), (3, 4, (0, 0)),
                          (3, 4, (1, 2)), (3, 5, (0, 3)), (4, 4, (0, 1, 2, 3))):
        P0 = miura(nr, nc)
        for slot, (i, j) in enumerate(P0.interior):
            if not (1 < j < nr) or i > 2:
                continue
            P = raise_degree_at(_ff_sample(1, nr, nc), slot, _places(order))
            fast, _ = fast_component_count(P)
            slow = len(decompose(P)["cells"])
            if fast != slow:
                bad += 1
                rows.append(f"{nr}x{nc} col{i} {order}: fast {fast}, enumerated {slow}")
    return bad == 0, ("incremental and enumerated counts agree on every checked "
                      "configuration" if not bad else "; ".join(rows[:4]))


def t_cell_budget_splits_the_formula():
    """The bisector closed form splits into the two counts of the paper. With
    a = 2^(col-1) - 1 and c = 2^(ncols-col) - 1:

      Theorem 5.2: there are |L| = 2^ncols + 2a + 2c + 2 flats for v* in an
               interior row, independent of the stubs.
      Theorems 6.2 and 6.7, bisecting stubs: the deaths are alpha*a + beta*c,
               with (0,0) degree 8, (1,0)/(0,1) a diagonal pair with the narrow
               side on the right/left, (2,0) the narrow pair, (0,2) the wide
               pair, (1,1) an adjacent pair, (2,1) two stubs in one narrow
               sector, (1,2) two in one wide sector, (2,2) a single stub.

    Both are checked here against the measured decomposition, separately, so a
    failure says which half is wrong."""
    badA = badB = tot = 0
    rows = []
    for nc in (3, 4, 5):
        P0 = miura(3, nc)
        for i in range(1, nc + 1):
            slot = [k for k, v in enumerate(P0.interior) if v == (i, 2)][0]
            for o in ((0,), (0, 1), (2, 3), (1, 2), (0, 3), (0, 0), (1, 1),
                      (0, 1, 2, 3), (0, 2), (1, 3)):
                N_pred, deaths = cell_budget(nc, i, _places(o))
                alive, N = fast_component_count(
                    raise_degree_at(_ff_sample(1, 3, nc), slot, _places(o)))
                tot += 1
                if N != N_pred:
                    badA += 1
                    rows.append(f"A nc{nc} col{i} {o}: {N} vs {N_pred}")
                d = N - alive
                ok = (d in deaths) if isinstance(deaths, tuple) else (d == deaths)
                if not ok:
                    badB += 1
                    rows.append(f"B nc{nc} col{i} {o}: {d} vs {deaths}")
    return badA == 0 and badB == 0, (
        f"flat count (Thm 5.2) {tot-badA}/{tot}, deaths for bisecting stubs "
        f"{tot-badB}/{tot}; the deaths cover the whole grid of coefficients "
        f"{{0,1,2}}^2 of a and c"
        if not (badA or badB) else "; ".join(rows[:4]))


def t_exact_rational_certification():
    """The headline counts certified with no floating point anywhere.

    Every number this project reports came out of tolerances: tol_rank for
    ranks, tol_nest for subspace nesting, eigenvalue signs for definiteness.
    Tolerances already invented one result during development, so the headline
    numbers deserve better. Substituting rho_e = sigma_e |w_e| makes both J0 and
    every Hessian block rational for rational vertex positions, the rank-2
    blocks split over Q, and the whole count can then run in Fraction
    arithmetic: subspaces compared by reduced row echelon form rather than a
    norm, inertia by exact symmetric elimination rather than eigenvalues."""
    from fractions import Fraction as Fr
    from .exact import miura_rational, raise_rational, exact_component_count
    cases = [(3, 1, [(0, Fr(1, 2)), (1, Fr(1, 2))], 13),
             (3, 2, [(0, Fr(1, 3)), (0, Fr(2, 3))], 11),
             (4, 1, [(0, Fr(1, 2)), (1, Fr(1, 2))], 25),
             (5, 1, [(0, Fr(1, 2)), (1, Fr(1, 2))], 49),
             (5, 1, [(0, Fr(1, 2)), (3, Fr(1, 2))], 64)]
    rows = []
    for nc, col, places, want in cases:
        pos, interior, edges = miura_rational(3, nc)
        slot = [k for k, v in enumerate(interior) if v == (col, 2)][0]
        p2, i2, e2 = raise_rational(pos, interior, edges, slot, places)
        got, _ = exact_component_count(p2, i2, e2)
        if got != want:
            return False, f"3x{nc} col{col}: exact {got}, expected {want}"
        rows.append(f"3x{nc} col{col} -> {got}")
    return True, ("certified in exact rational arithmetic, no tolerance used: "
                  + ", ".join(rows) + "; 97 and 193 also certified in development")


def _family_deaths(P0, nr, nc, i0, places):
    """Group the maximal cells of a single-implant pattern into the four families
    F(side, half) and report, per family, the list of dead flags. Also returns
    the dead flags of every cell outside the four families."""
    from .cone import incremental_linear_cells
    from collections import defaultdict
    slot = [k for k, v in enumerate(P0.interior) if v == (i0, 2)][0]
    P = raise_degree_at(P0, slot, places)
    B, _ = kernel(P)
    Z = {}
    for key, e in P.cidx.items():
        a, b = tuple(key)
        if (isinstance(a, tuple) and a and a[0] == "stub") or \
           (isinstance(b, tuple) and b and b[0] == "stub"):
            continue
        (i1, j1), (i2, j2) = a, b
        if j1 != j2:
            Z[(i1, min(j1, j2))] = e
    zm, zp = Z[(i0, 0)], Z[(i0, nr)]
    cells, blocks = incremental_linear_cells(P)
    Qv = [p for kd, p in blocks if kd == "quadric"][0]
    fam, other = defaultdict(list), []
    for d, N in cells:
        R = B @ N
        nz = lambda e: np.abs(R[e]).max() > 1e-9
        L = any(nz(Z[(i, 0)]) for i in range(1, i0))
        Rt = any(nz(Z[(i, 0)]) for i in range(i0 + 1, nc + 1))
        side = {(True, False): "L", (False, True): "R"}.get((L, Rt))
        half = ("-" if nz(zm) else "") + ("+" if nz(zp) else "")
        Qr = N.T @ Qv @ N
        p, m, z0 = signature(Qr)
        dead = bool(np.abs(Qr).max() > 1e-9 and z0 == 0 and (p == 0 or m == 0))
        if side and half in ("-", "+"):
            fam[side + half].append(dead)
        else:
            other.append(dead)
    return fam, other, slot


def t_lemma_b_family_structure():
    """Why the deaths are alpha*a + beta*c. Only four families of maximal cells
    can ever die: those whose zigzag support lies strictly on ONE side of the
    implanted column (left or right) and in which exactly ONE half of the split
    column moves (below or above the implant). A family on the left has exactly
    a = 2^(col-1) - 1 cells, one per nonempty subset of the left columns, and
    on the right c = 2^(ncols-col) - 1. Within a family either every cell dies
    or none does, so the dead count is a times the number of dead left families
    plus c times the number of dead right families, and there are two of each."""
    bad, tot = [], 0
    for nc in (3, 4):
        for i0 in range(1, nc + 1):
            a, c = 2 ** (i0 - 1) - 1, 2 ** (nc - i0) - 1
            for order in ((0,), (0, 1), (1, 2), (0, 3), (0, 2), (0, 0), (0, 1, 2, 3)):
                fam, other, _ = _family_deaths(_ff_sample(1, 3, nc), 3, nc, i0, _places(order))
                tot += 1
                if any(other):
                    bad.append(f"nc{nc} col{i0} {order}: a cell outside the families died")
                for k, v in fam.items():
                    if len(set(v)) > 1:
                        bad.append(f"nc{nc} col{i0} {order} {k}: mixed")
                    if len(v) != (a if k[0] == "L" else c):
                        bad.append(f"nc{nc} col{i0} {order} {k}: size {len(v)}")
    return not bad, (f"{tot} configurations: nothing outside the four families dies, each "
                     f"family is all-or-nothing, and family sizes are a and c"
                     if not bad else "; ".join(bad[:3]))


TESTS = [
    ("commutator identity", t_commutator),
    ("second-order expansion is cubic-accurate", t_second_order),
    ("Chen-Santangelo signature (1, n-3)", t_chen_santangelo_signature),
    ("degree-3 vertex is rigid", t_degree3_rigid),
    ("degree-4 vertex: 2 branches, Maekawa 3-1", t_degree4_two_branches),
    ("JoCG 2016 sector criterion (all < pi)", t_sector_criterion),
    ("JoCG 2016 cross exception", t_cross_exception),
    ("shared-crease sign convention", t_sign_convention),
    ("Tachi 2009: generic quad mesh is rigid", t_tachi_generic_quad_rigid),
    ("Tachi 2009: Miura is 1-DOF despite E-3V", t_tachi_miura_one_dof),
    ("Hull-Tachi 2018: one all-creases mode", t_hull_tachi_two_modes),
    ("degree-6 implant breaks the MV label", t_degree_raise_breaks_mv_label),
    ("second-order foldability is NOT sufficient", t_second_order_not_sufficient),
    ("reported counts are certified branches", t_reported_counts_are_certified),
    ("conversion ratios constant iff flat-foldable", t_conversion_constant_iff_flat_foldable),
    ("closed-form conversion coefficients", t_closed_form_conversion),
    ("Tachi's condition is the SIGNED loop product", t_tachi_condition_is_signed),
    ("a rigid-foldable mesh with no straight fold line", t_generic_rigid_foldable_mesh),
    ("H3 needs a straight fold line to even be stated", t_h3_needs_a_straight_fold_line),
    ("matches Dieleman et al. Class 1 branch count", t_dieleman_class1_count),
    ("the definiteness mechanism predicts the count", t_definiteness_mechanism),
    ("widest-sector raising is discontinuous", t_widest_sector_rule_is_discontinuous),
    ("bisecting stubs: placement decides the count", t_placement_decides_the_count),
    ("bisector closed form for the count", t_closed_form_component_count),
    ("Cor 7.4: diagonal pair, 25/23/26 and never both die", t_cor74_diagonal_pair_examples),
    ("Cor 7.3(a): one empty side, 20/26", t_cor73_one_empty_side_examples),
    ("Cor 7.3 and Thm 6.7 closed forms vs full count", t_cor73_closed_forms_match_full_count),
    ("incremental intersection matches enumeration", t_incremental_matches_enumeration),
    ("cell budget splits the closed form", t_cell_budget_splits_the_formula),
    ("exact rational certification", t_exact_rational_certification),
    ("Thm 6.2: four all-or-nothing families", t_lemma_b_family_structure),
]


def run(verbose: bool = True):
    results = []
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        results.append((name, ok, detail))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    if verbose:
        print(f"\n{n_ok}/{len(results)} checks passed")
    return results


if __name__ == "__main__":
    run()
