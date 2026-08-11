# CAPAL (ICLR 2026) vs E-L\* on noisy DFA learning

_Generated from `data/capal/*.json` by `orthogonal_dfa.experiments.capal_comparison.generate_report`. Do not edit by hand; rerun the generator after any experiment rerun._

Upstream CAPAL pinned at `57d877f6a083d58852660fac388ff49c052dc2d2`, run at its authors' benchmark-script settings. Both learners model persistent noise, so a membership count is the distinct strings each was told about. The two columns are not the same cost: CAPAL is given a perfect equivalence oracle (the paper's pMAT assumption) whose counterexamples arrive as gold labels, while E-L* has no EQ and manufactures counterexamples out of membership queries. Read `mq` and `eq` together.

## 1. The wall: hyperparameter sweep

Every combination of `max_same_samples`, `suffix_pool_len_max` and `alpha`, on
every cell, at all four noise levels, for three seeds (480 runs). For
each (cell, η), how many of the 24 configs (knobs × seeds) converge:

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

## 2. Matched query budget: CAPAL at E-L*'s own spend

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

## 3. Why the noise floor bites CAPAL harder

CAPAL's SAMESTATE compares two noisy rows against each other, so its
disagreement floor is p₀ = 2η(1−η) and true signal is compressed by (1 − 2p₀).
E-L* compares one noisy accept rate against a boundary, so its floor is η and
signal scales by (1 − 2η). At η=0.30 that is 0.16 against 0.40.

CAPAL's test can call a pair different only when the fraction d of suffixes
distinguishing them exceeds τ/(1 − 2p₀), where τ = √(ln(2/α)/2m) over m probe
suffixes (capal.py:674). For modulo-9's ±3 pairs d ≤ 2/9, so at η=0.30 CAPAL
needs m > 3006 at α=1e-3, or m > 1459 at α=0.05. The matched-budget probe ran at
m=2000, α=1e-3 -- under its own threshold, so it shows CAPAL stopping short
rather than a limit no budget could clear. Running modulo at η=0.30 with m=5000,
or at m=2000 with α=0.05, would settle it.

## 4. Caveats

- The membership columns are not like for like. CAPAL is handed a perfect
  equivalence oracle and E-L* is not, so part of what E-L* pays for in queries
  is work CAPAL is given.
- Neither learner is measured on a neutral set. This repo's five targets are
  its own test set, which is why E-L* is in regime on all of them and on only
  five of CAPAL's 28; the rest fail acceptance imbalance, class-preservation or
  the covered-accuracy ceiling. The sweep and the matched-budget probe run only
  on those five, so CAPAL's own suite has never been run at more than one
  configuration.
- The head-to-head experiments are single-seed; the sweep and the probe use
  three. Per-cell verdicts move under re-measurement: raising CAPAL's budget to
  its authors' settings flipped cells in both directions, one of them from 1.000
  to 0.507. Read single-seed per-cell numbers as indicative.
- The query counts are a snapshot of the learners as they stand. A change to
  E-L*'s suffix screening moved its counts by up to 42x without changing which
  cells it solves.
