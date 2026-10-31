# Changelog

This file records how the package developed, including claims that were later
corrected or refuted. The current statements are those of the paper and of
the README; where an entry below disagrees with them, the entry is history.

## 2.4.0 (third review round)

* New module `origami_cone.criterion`: the death criterion of Theorem 6.7
  evaluated at the implanted vertex alone (`gram_determinant`, `family_dies`)
  and the count it predicts for an interior-row implant at arbitrary stub
  positions (`predict_N`).
* Regression suite: three tests that encoded refuted claims were replaced.
  * `t_opposite_pair_is_the_position_dependent_case` claimed that only
    diagonal ("opposite") pairs depend on position. False: by Corollary 7.3
    every class with an empty associated sector does.
  * `t_opposite_pair_two_chamber_formula` claimed that a diagonal pair takes
    one of two "chamber" values. False: a diagonal pair takes |L|, |L| - a and
    |L| - c, all on one vertex (Corollary 7.4).
  * `t_chamber_is_the_lean_of_the_implanted_column` claimed that the value is
    decided by the lean of the implanted column. True only for bisecting
    stubs (Corollary 7.3(c)).

  They are replaced by tests of Corollaries 7.3 and 7.4 themselves (the
  examples N = 25, 23, 26 and N = 20, 26 on the 3x4 patch at v_{2,2}) and by
  a randomised check of the closed forms of Corollary 7.3 and of
  `criterion.predict_N` against the full count. The docstrings of
  `t_placement_decides_the_count`, `t_closed_form_component_count`,
  `predicted_at`, `predicted_components`, `opposite_pair_values` and
  `predict_pattern` are now limited to bisecting stubs.
* `checks_round3/`: scripts for the new statements of the third revision
  (degenerate outer-row case, face-confined stubs, one-sample check of
  Theorem 9.3, recomputation of the wall table).
* `checks_round2/paper_stubs/extend.build_physical` takes the horizontal face
  side `a` as a parameter.
* README rewritten in English, with a map from Appendix A of the paper to the
  scripts; the development log moved here and translated. `VALIDATION.txt`
  regenerated (31 checks). MIT license added; `sympy` declared as an optional
  dependency.

## 2.3.0 (second review round)

* `incremental_linear_cells` keeps only the maximal subspaces after each
  vertex (`prune=True`, the default). This is exact and lowers the number of
  stored subspaces, e.g. from 224 to 64 for a 5x5 patch with a degree-8
  implant.
* `checks_round2/model/`: scripts behind Appendix A for the hinged-stub model
  (death criterion, branches, outer rows, the map Xi, pruning).
* `checks_round2/paper_stubs/`: stubs continued straight to the edge of the
  sheet, with degree-4 crossing vertices (`extend.build_physical`), and the
  checks of Theorem 9.3.

## 2.1.0: structure of the deaths

* Matrix factorisation of the Miura second-order forms (contributed by the
  author): the table of q_v(x) over all vertices is a product of an
  ncols x 2 and a 2 x nrows matrix, which gives a short proof of the 2^ncols
  components of a plain Miura patch (Theorem 3.6 of the paper).
* Four families F(side, half) of flats that can die, each all-or-nothing, of
  sizes a and c; measured on 240 configurations with no violation, later
  proved (Theorem 6.2). Test `t_lemma_b_family_structure`.
* A "sector rule" for which families die was stated here and later found to
  be false in general; the paper replaces it by the exact criterion of
  Theorem 6.7.
* The "chambers" of a diagonal pair were described as the lean of the
  implanted column. This holds for bisecting stubs only (see 2.4.0).
* A genuine position dependence was found at degree 7 (two stubs in one
  narrow sector and one more): this is Corollary 7.3(a) and Table 2 of the
  paper.

## 2.0.0: a correction, the flat count, exact certification

* Correction: `raise_degree_by_index` accepted a repeated sector index and
  produced two COINCIDENT creases for it, a degenerate crease pattern rather
  than two stubs in a sector. Every earlier "stacked" count measured that
  doubled crease. Repeated indices are now rejected, and explicit placements
  are given with `raise_degree_at(P, slot, [(sector, fraction), ...])`. Only
  the "stacked" row of the table changed; the corrected rows are B + A + 1 and
  B + C + 1 for two stubs in one wide or narrow sector.
* The count of maximal flats, B + 2A + 2C - 2 for an interior-row implant and
  independent of the stubs, was reduced to an explicit hyperplane arrangement
  built from the factorisation of the Miura forms (scripts `abstract.py`,
  `abstract2.py`, not shipped); later proved (Theorem 5.2).
* `origami_cone.exact`: the whole count in rational arithmetic, after the
  substitution rho_e = sigma_e |w_e|; certified 13, 11, 25, 49, 64, 97, 193,
  144 and 385 with no tolerance.

## 1.2.0: the count splits in two

* `cell_budget`: the closed form split into a count of flats and a count of
  deaths alpha * a + beta * c. Then called "Lemma A" and "Lemma B"; these are
  Theorem 5.2 and Theorems 6.2 and 6.7 of the paper.

## 1.1.0: incremental intersection

* `incremental_linear_cells` and `fast_component_count`: the hyperplanes are
  intersected vertex by vertex, keeping only distinct subspaces, instead of
  enumerating 2^(V-1) choices. A 3x7 patch went from 76 s to 2.3 s, and 5x5,
  3x9, 6x6 and 8x5 became feasible. Same answers as `decompose` everywhere
  checked.

## 1.0.0: the closed form

* `predicted_components`: a closed form for one implanted vertex with
  bisecting stubs, checked on 507 configurations and then predicted blind on
  3x5, 4x4, 3x6 and 3x7. The diagonal pairs were described as having two
  "chamber" values; see 2.1.0 and 2.4.0 for what replaced that description.

## 0.8.0: the mechanism

* The old `raise_degree` split the WIDEST sector, which jumps when two
  sectors are nearly equal; sweeps that used it compared different crease
  patterns. `raise_degree_by_index` (fixed sector numbering) replaced it.
* Observed: with exactly one vertex contributing an irreducible quadric, the
  number of components equals the number of maximal linear flats on which
  that quadric is not definite (`definiteness_count`). This is the mechanism
  proved in the paper.

## 0.7.0 and earlier

* Exploratory runners (`run_a_noncollinear.py`, `run_b_sweep.py`,
  `run_c_formula.py`, not shipped) for non-collinear rigid-foldable quad meshes
  and for position sweeps of mixed-degree patterns.
* Hypothesis H3 (`origami_cone.h3`) on when a single raise changes the count;
  it is phrased through a straight fold line and is superseded by the theorems
  of the paper. Conjectures of the form 2^V4 and 2^(dim K - d_max) were refuted
  and are kept in `sweep.py` as null hypotheses.
