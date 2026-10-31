# origami_cone

Companion code for the paper *Folding branches of Miura-ori with one vertex of
higher degree: a hinged-stub model and its realisation in paper*. It computes
the second-order tangent cone of the rigid-origami configuration space at the
flat state, for an arbitrary flat crease pattern, and contains every script
behind Appendix A of the paper.

Version 2.4.0. MIT license (see `LICENSE`). History, including claims that were
later corrected, is in `CHANGELOG.md`.

## What it computes

For a crease pattern with interior vertices `v` and fold angles `rho_e`, the
closure condition at each interior vertex is

    Phi_v(rho) = prod_k R(u_k, rho_k) = I

with `u_k` the crease directions in counter-clockwise order. Around `rho = 0`,

    log Phi_v(rho) = ( sum_e rho_e cos alpha_e,
                       sum_e rho_e sin alpha_e,
                       rho^T Q_v rho            ) + O(rho^3)
    (Q_v)_ef = (1/4) sin(alpha_f - alpha_e),   e != f in star(v)

so the second-order cone is

    C2 = { rho : J0 rho = 0 }  ∩  ∩_v { rho^T Q_v rho = 0 }.

The library builds `J0` and the blocks `Q_v`, restricts them to `K = ker J0`,
classifies each block (a degree-4 vertex gives a rank-2 form that factors into
two hyperplanes; a vertex of degree at least 5 gives a quadric), intersects the
hyperplanes vertex by vertex to get the maximal flats, and keeps the flats on
which the remaining quadric is not definite. With one implanted vertex this is
exactly the set of components of C2 (Section 6 of the paper).

## Install

```
pip install .                 # numpy, scipy
pip install ".[symbolic]"     # adds sympy, used only by checks_round3/exact_vertex_symbolic.py
python -m origami_cone.validate
```

The suite runs 31 checks in about three minutes; the expected output is in
`VALIDATION.txt`.

## Quick use

```python
from origami_cone import miura, raise_degree_at, fast_component_count
from origami_cone.criterion import predict_N

P0 = miura(3, 4)                                   # 3 x 4 Miura patch, offset 3/5
slot = P0.interior.index((2, 2))                   # v* = v_{2,2}
places = [(0, 0.95), (2, 0.05)]                    # stubs as (sector, fraction)
P = raise_degree_at(P0, slot, places)
N, L = fast_component_count(P)                     # N = 23 surviving of |L| = 26 flats
print(N, L, predict_N(P0, 3, 4, slot, places))     # Theorem 6.7 at v* alone: 23
```

Sectors are numbered counter-clockwise from the first crease of the vertex; a
stub at fraction `f` of sector `s` makes the angle `f` times the sector width
with the crease that starts the sector. Arbitrary patterns go through
`from_vertices(pos, interior, edges)`. Exact rational counts are in
`origami_cone.exact` (`miura_rational`, `raise_rational`,
`exact_component_count`). Stubs continued to the edge of the sheet, with
crossing vertices, are built by `checks_round2/paper_stubs/extend.py`
(`build_physical`).

## What the paper proves, and where the code stands

Notation: `B = 2^ncols`, `A = 2^(col-1)`, `C = 2^(ncols-col)`, `a = A - 1`,
`c = C - 1`, for v* in column `col`.

* Plain Miura patch: `B` components (Theorem 3.6).
* One implanted vertex in an interior row: `|L| = B + 2A + 2C - 2` maximal
  flats, independent of the stubs (Theorem 5.2). The flats that can die form
  four families of sizes `a, a, c, c`, each all-or-nothing (Theorem 6.2), and
  a family dies iff `t = 1` or `Q1 Q2 < Q12^2` (Theorem 6.7). This is
  `criterion.family_dies` and `criterion.predict_N`.
* Which families die depends on where the stubs are for every class with an
  empty associated sector (Corollary 7.3); a diagonal pair never loses both
  families (Corollary 7.4). The closed forms of `predicted_at`,
  `predicted_components`, `cell_budget` and `predict_pattern` are for
  BISECTING stubs only; for any other position use `criterion.predict_N`.
* One implanted vertex in an outer row: `N = B` for any stubs (Theorem 7.5).
* Every surviving flat carries a folding branch except the flats `M-0` and
  `M0+` when all stubs lie on one side, so `N_br = N - epsilon`
  (Theorem 8.4, Corollary 8.5).
* Stubs continued to the edge of the paper: Theorems 9.3 and 9.4, and the
  face-confined case of Proposition 9.1.

## Appendix A of the paper, script by script

All scripts insert the package root into `sys.path` themselves and can be run
from any folder.

| Appendix A paragraph / statement | script(s) | result |
|---|---|---|
| The closed form, Table 3 (8580 configurations, bisecting stubs) | `paper_verify.py` (output `paper_verify.json`) | all agree |
| The closed form, exact certification (Prop. 11.2) | `checks_round3/exact_certify.py` | 10/10 |
| Maximal flats, types (Thm 5.2) | `checks_round2/model/census.py` | 828/828 |
| Maximal flats, dimensions of the four flats of case (a) | `checks_round2/model/census_dims.py` | 828/828 |
| Death criterion vs direct definiteness, 16,000 cases (Thm 6.7, Lemma 6.4) | `checks_round2/model/crit.py` | all agree |
| Closed forms of Cor. 7.3 vs the criterion, 32,000 cases | `checks_round2/model/check_sine.py`, `checks_round2/model/check_formulas.py` | all agree |
| Criterion at v* alone vs the full count, 700 configurations | `checks_round2/model/vs_pkg.py`; also test `t_cor73_closed_forms_match_full_count` | all agree |
| The 24 entries of Table 2 (the wall, Cor. 7.3(a)) | `checks_round3/wall_table.py` | 24/24 |
| Cor. 7.4 example 25/23/26 and Cor. 7.3 example 20/26 | tests `t_cor74_diagonal_pair_examples`, `t_cor73_one_empty_side_examples` | pass |
| Branches, 435 configurations (Thm 8.4, Cor. 8.5) | `checks_round2/model/branches_all.py`, `checks_round2/model/branches_all2.py` (helpers in `branches.py`) | `N_br = N - epsilon` in all |
| The map Xi (Thm 8.2) | `checks_round2/model/psi_test.py` | closure errors ~1e-15 |
| Exact Miura vertex (Lemma 8.1), numerically | `checks_round2/model/exact_vertex.py` | residuals ~1e-15 |
| Exact Miura vertex (Lemma 8.1), symbolically, both leans | `checks_round3/exact_vertex_symbolic.py` (needs `sympy`) | identity holds |
| Outer rows, 138 configurations (Thm 7.5, Cor. 8.6) | `checks_round2/model/outer.py` | 138/138 |
| Outer rows, stub continuing the boundary zigzag | `checks_round3/outer_degenerate.py` | 64/64 |
| Stubs in paper, one sample for `N` and `N_br` (Thm 9.4) | `checks_round3/paper_stubs_sample.py` (uses `checks_round2/paper_stubs/`) | 100/100 and 100/100 |
| Face-confined stubs, `N = 2B` iff `a < d` (Prop. 9.1) | `checks_round3/face_confined.py` | as stated |
| Incremental intersection and pruning (Prop. 11.1) | `checks_round2/model/prune.py`; test `t_incremental_matches_enumeration` | same flats |
| Figures 1 and 3 | `checks_round2/paper_stubs/fig_patch.py`, `figpaper.py` | |

`checks_round2/paper_stubs/phys_branches.py`, `phys_branches2.py` and
`phys_sweep.py` are the earlier, separate samples of the "Stubs in paper"
check; `checks_round3/paper_stubs_sample.py` replaces them with one sample
that reports both counts.

## Regression suite

`python -m origami_cone.validate` runs 31 checks. Each reproduces a published
result or a statement of the paper, so a change that breaks the theory breaks
a test.

| check | source |
|---|---|
| commutator identity `[hat u_i, hat u_j] = sin(a_j - a_i) hat z` | derivation |
| second-order expansion has a cubic remainder | derivation |
| degree-N form has signature (1, N-3) | Chen & Santangelo, PRX 8, 011034 (2018) |
| degree-3 vertex is rigid | classical |
| degree-4 vertex: 2 branches, Maekawa 3-1 split | Waitukaitis et al., PRL 114, 055503 (2015) |
| single vertex foldable iff all sectors < pi | Abel et al., JoCG 7 (2016), Thm 2 |
| the "cross" exception to that theorem | same |
| shared-crease sign convention | internal consistency |
| generic quad mesh is rigid | Tachi, IASS 2009 |
| Miura is 1-DOF despite `E - 3V` | Tachi, IASS 2009 |
| exactly one all-creases folding mode | Hull & Tachi et al., arXiv:1809.04243 |
| a degree-6 implant destroys the mountain-valley branch label | observation |
| a nonempty tangent cone does not imply rigid foldability | counterexample fixture |
| reported counts are certified branches | `kinematics.certify_cells` |
| conversion ratios constant iff the vertex is flat-foldable | derivation |
| closed-form branch ratios at a flat-foldable degree-4 vertex | derivation |
| Tachi's condition is the SIGNED loop product | Tachi, IASS 2009, made precise |
| a 1-DOF rigid-foldable quad mesh with no straight fold line | constructed |
| H3 is undefined without a straight fold line | `origami_cone.h3` |
| non-collinear quad tiling has 2 branches | Dieleman et al., Nat. Phys. 16, 63 (2020), Table I |
| the definiteness mechanism predicts the count | paper, Section 6 |
| widest-sector raising is discontinuous | why `raise_degree_at` is used |
| bisecting stubs: placement decides the count | paper, Cor. 7.3(c), Thm 7.5 |
| bisector closed form for the count | paper, Thm 7.1 and Table 1 |
| Cor. 7.4: diagonal pair, 25/23/26, never both die | paper, Cor. 7.4 |
| Cor. 7.3(a): one empty side, 20/26 | paper, Cor. 7.3(a) |
| Cor. 7.3 and Thm 6.7 closed forms vs full count, random | paper, Cor. 7.3, Thm 6.7 |
| incremental intersection matches enumeration | paper, Prop. 11.1 |
| flat count and deaths split the closed form | paper, Thms 5.2, 6.2 |
| exact rational certification | paper, Prop. 11.2 |
| four all-or-nothing families of sizes a and c | paper, Thm 6.2 |

## Other modules

* `origami_cone.tachi`: the exact finite condition for flat-foldable quad
  meshes. At a developable degree-4 vertex `tan(rho_i/2)/tan(rho_j/2)` is
  constant along a branch iff the vertex is flat-foldable; around each inner
  face the signed loop product must be `+1`. `tachi.is_rigid_foldable` is
  strictly sharper than the second-order condition.
* `origami_cone.kinematics`: closure residuals, slice solver, branch
  continuation and certification.
* `origami_cone.invariants`: mountain-valley patterns of cells and forcing
  sets (exhaustive, small patterns only).
* `origami_cone.h3`, `origami_cone.sweep`: an early hypothesis and the refuted
  conjectures, kept as null hypotheses and superseded by the paper.

## Known limitations

* The default pipeline is floating point with tolerances (`tol_rank`,
  `tol_nest`). The counts were stable over six orders of magnitude of
  tolerance, and `origami_cone.exact` repeats the count in rational arithmetic
  for rational positions and stubs.
* The tangent cone is not the configuration space.
  `patterns.second_order_impostor()` lies on the second-order locus and does
  not fold. Branches are therefore tested separately (Section 8 of the paper,
  `kinematics.certify_cells`, and the "Branches" scripts above).
* `forcing_sets` is exhaustive over crease subsets.

## Related work

Branch counts for patterns with all vertices of degree 4 are given by
Dieleman et al. (2020), and for triangulated patterns by Chen & Santangelo
(2018). Hayakawa, Ohba and Ohsaki (Mech. Mach. Theory 194, 105572, 2024) solve
the second-order conditions analytically for specific patterns, including a
mixed-degree double-waterbomb pattern. The paper explains how the present
counts relate to these.
