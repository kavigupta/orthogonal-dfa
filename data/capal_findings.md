# CAPAL (ICLR 2026) vs E-L\* on noisy DFA learning

_Generated from `data/capal/*.json` by `orthogonal_dfa.experiments.capal_comparison.generate_report`. Do not edit by hand; rerun the generator after any experiment rerun._

Upstream CAPAL pinned at `57d877f6a083d58852660fac388ff49c052dc2d2`, run at its authors' benchmark-script settings. Both learners model persistent noise, so a membership count is the distinct strings each was told about. The two columns are not the same cost: CAPAL is given a perfect equivalence oracle (the paper's pMAT assumption) whose counterexamples arrive as gold labels, while E-L* has no EQ and manufactures counterexamples out of membership queries. Read `mq` and `eq` together.

## 1. CAPAL's own benchmark suite

CAPAL's 28 shipped `.taf` targets (Simple/Normal/Difficult), at
η ∈ {0.05, 0.10, 0.20, 0.30}. Which cells CAPAL fails, and the size of the
automaton it returns for them:

| target | η | acc | states |
| --- | --- | --- | --- |
| Normal01 | 0.30 | 0.161 | 10/12 |
| Difficult02 | 0.30 | 0.507 | 29/5 |
| Normal03 | 0.30 | 0.998 | 40/7 |

E-L* is admitted on Difficult02, Normal07, Simple01, Simple02, Simple05. The other 23 are excluded by
acceptance imbalance, class-preservation or the covered-accuracy ceiling, and
not run.

## 2. This repo's benchmarks

The modulo-9 and regex oracles from `tests/test_lstar.py`: 8-11 states, larger
than CAPAL's suite, and selected to satisfy E-L*'s preconditions -- which is why
E-L* applies to every target here and to five of CAPAL's 28.

## 3. The wall: full hyperparameter sweep

A full factorial over CAPAL's three real knobs -- `max_same_samples`,
`suffix_pool_len_max`, `alpha` -- across every cell, all four noise levels, and
three seeds (480 runs). For each (cell, η), how many of the
24 configs (knobs × seeds) converge:

| cell | η=0.05 | η=0.1 | η=0.2 | η=0.3 |
| --- | --- | --- | --- | --- |
| parity_mod9_allowed_3_6 | 21/24 | 11/24 | wall (0.92) | wall (0.91) |
| regex_subseq_1010101 | 20/24 | 15/24 | 8/24 | wall (0.93) |
| regex_two_1111 | 22/24 | 15/24 | 1/24 | wall (0.87) |
| regex_alt_1111_or_0000_11 | 19/24 | 13/24 | wall (0.92) | wall (0.78) |
| regex_alt_111_or_000_3sym | 22/24 | 12/24 | 4/24 | wall (0.88) |

**Noise dominates, but not alone.** At η=0.30 every cell fails on all 24 configs and every seed; parity_mod9_allowed_3_6 already walls at η=0.2, regex_alt_1111_or_0000_11 already walls at η=0.2, while the other 3 cells still crack there. Which DFA it is decides where the wall starts; the noise level decides that there is one.

The grid's low corner is upstream's own benchmark setting, so a cell that fails
across it failed with at least the budget CAPAL's authors publish with.

## 4. Matched query budget: CAPAL at E-L*'s own spend

CAPAL with its suffix enumeration uncapped (`enum_depth=8`,
`extra_len_max=16`, `suffix_pool_len_max=24`,
`max_same_samples=2000`) on the η=0.30 cells, three seeds,
stopped once it has issued the membership queries E-L* spent on the same cell.

It converges on none of them, on any seed. Where a cell did spend its budget,
CAPAL was handed exactly the queries E-L* used plus a perfect equivalence oracle
E-L* never gets, and still came back short of it.

A stalled cell ran out of iterations at a fixed point where further rounds issue
no new queries at all -- on regex_alt_111_or_000_3sym the distinct count is
identical at 50 iterations and at 10000 -- so no budget could bind, and the cell
is not a matched-budget measurement. That is a stronger statement than a low
score rather than a weaker one: CAPAL stops improving at a fraction of E-L*'s
spend and cannot use more.

## 5. Why the noise floor bites CAPAL harder (theory)

Both learners use statistical row-equality under persistent noise, but the test
*shape* differs. CAPAL's SAMESTATE compares two noisy rows against each other,
so its noise floor is `p₀ = 2η(1−η)` and observed signal scales by `(1 − 2p₀)`.
E-L* measures each prefix's own accept rate against a data-driven boundary, so
its floor is just `η` and signal scales by `(1 − 2η)`.

| η    | CAPAL signal (1−2p₀) | E-L* signal (1−2η) | ratio |
| ---- | -------------------- | ------------------ | ----- |
| 0.05 | 0.81                 | 0.90               | 1.1×  |
| 0.10 | 0.64                 | 0.80               | 1.25× |
| 0.20 | 0.36                 | 0.60               | 1.7×  |
| 0.30 | 0.16                 | 0.40               | 2.5×  |

At η=0.30 E-L* gets 2.5× more usable signal on the same oracle, and the gap
widens with noise. For the pairs CAPAL merges on modulo-9 (states differing by
±3 mod 9), the maximum true disagreement any suffix can produce is 2/9 ≈ 0.22,
so at η=0.30 the observed disagreement sits only ~0.035 above the 0.42 floor.
Resolving that needs a threshold so tight it over-splits every easy pair -- one
global knob (τ) cannot serve the hard and easy pairs at once. That is the wall,
and it is structural to the pairwise test, which is why §4's matched budget does
not move it.

## 6. Caveats

- The membership columns are not like for like. CAPAL is handed a perfect
  equivalence oracle and E-L* is not, so part of what E-L* pays for in queries
  is work CAPAL is given.
- Sections 3 and 4 cover only this repo's five targets. CAPAL's own 28-target
  suite has never been run at more than one configuration.
- Sections 1 and 2 are single-seed; the sweep and the matched-budget probe use
  three. Per-cell verdicts move under re-measurement: raising CAPAL's budget to
  its authors' settings flipped cells in both directions, one of them from 1.000
  to 0.507. Read single-seed per-cell numbers as indicative.
- The query counts are a snapshot of the learners as they stand. A change to
  E-L*'s suffix screening moved its counts by up to 42x without changing which
  cells it solves.
