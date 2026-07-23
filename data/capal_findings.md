# CAPAL (ICLR 2026) vs E-L\* on noisy DFA learning

_Generated from `data/capal/*.json` by `orthogonal_dfa.experiments.capal_comparison.generate_report`. Do not edit by hand; rerun the generator after any experiment rerun._

Upstream CAPAL pinned at `57d877f6a083d58852660fac388ff49c052dc2d2`. Both learners model persistent noise, so `distinct` queries are the honest oracle cost on both sides.

## 1. CAPAL's own benchmark suite

Both learners on CAPAL's 28 shipped `.taf` targets (Simple/Normal/Difficult) at
η ∈ {0.05, 0.10, 0.20, 0.30}. This is CAPAL's home turf.

CAPAL solves **108/112** cells at 100% accuracy. Every failure is
at η=0.30:

| target | η | acc | states |
| --- | --- | --- | --- |
| Normal01 | 0.30 | 0.158 | 10/12 |
| Normal02 | 0.30 | 0.876 | 10/10 |
| Normal04 | 0.30 | 0.983 | 6/5 |
| Simple05 | 0.30 | 0.996 | 19/5 |

E-L* is in its designed regime on only **3/28** targets
(Normal07, Simple01, Simple02); the other 25 are recorded as reasoned
exclusions (acceptance imbalance / class-preservation / covered-accuracy
ceiling), not run. On the shared in-regime cells both are accurate, but the
query cost differs by orders of magnitude:

| target | η | CAPAL acc | conv | CAPAL q | E-L* acc | conv | E-L* q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Normal07 | 0.05 | 1.000 | yes | 919 | 1.000 | yes | 206,288 |
| Normal07 | 0.10 | 1.000 | yes | 905 | 1.000 | yes | 278,012 |
| Normal07 | 0.20 | 1.000 | yes | 2,748 | 1.000 | yes | 428,926 |
| Normal07 | 0.30 | 1.000 | yes | 2,752 | 1.000 | yes | 1,192,004 |
| Simple01 | 0.05 | 1.000 | yes | 434 | 1.000 | yes | 78,100 |
| Simple01 | 0.10 | 1.000 | yes | 435 | 1.000 | yes | 107,651 |
| Simple01 | 0.20 | 1.000 | yes | 442 | 1.000 | yes | 187,633 |
| Simple01 | 0.30 | 1.000 | yes | 468 | 1.000 | yes | 456,223 |
| Simple02 | 0.05 | 1.000 | yes | 434 | 1.000 | yes | 78,287 |
| Simple02 | 0.10 | 1.000 | yes | 435 | 1.000 | yes | 107,728 |
| Simple02 | 0.20 | 1.000 | yes | 442 | 1.000 | yes | 138,275 |
| Simple02 | 0.30 | 1.000 | yes | 468 | 1.000 | yes | 401,894 |

## 2. This repo's benchmarks (head-to-head)

Both learners on the modulo-9 and regex oracles from `tests/test_lstar.py`. On
our turf the picture inverts: E-L* is accurate and noise-robust where it is
in-regime, while CAPAL degrades badly under noise. E-L* pays for it in queries.

| target | η | CAPAL acc | conv | CAPAL q | E-L* acc | conv | E-L* q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parity_mod9_allowed_3_6 | 0.05 | 0.910 | no | 15,517 | 1.000 | yes | 318,521 |
| parity_mod9_allowed_3_6 | 0.10 | 0.876 | no | 9,917 | 1.000 | yes | 462,894 |
| parity_mod9_allowed_3_6 | 0.20 | 0.908 | no | 9,610 | 1.000 | yes | 869,453 |
| parity_mod9_allowed_3_6 | 0.30 | 0.847 | no | 12,240 | 1.000 | yes | 1,771,791 |
| regex_subseq_1010101 | 0.05 | 0.919 | no | 6,154 | 1.000 | yes | 823,884 |
| regex_subseq_1010101 | 0.10 | 0.919 | no | 5,652 | 1.000 | yes | 1,050,071 |
| regex_subseq_1010101 | 0.20 | 0.919 | no | 6,190 | 1.000 | yes | 1,226,669 |
| regex_subseq_1010101 | 0.30 | 0.919 | no | 5,651 | 1.000 | yes | 2,095,356 |
| regex_two_1111 | 0.05 | 0.954 | no | 11,096 | 1.000 | yes | 489,774 |
| regex_two_1111 | 0.10 | 1.000 | yes | 8,241 | 1.000 | yes | 426,561 |
| regex_two_1111 | 0.20 | 0.867 | no | 10,265 | 1.000 | yes | 862,578 |
| regex_two_1111 | 0.30 | 0.867 | no | 6,214 | 1.000 | yes | 2,217,453 |
| regex_alt_1111_or_0000_11 | 0.05 | 1.000 | yes | 17,483 | 0.989 | no | 316,030 |
| regex_alt_1111_or_0000_11 | 0.10 | 1.000 | yes | 21,810 | 0.989 | no | 512,100 |
| regex_alt_1111_or_0000_11 | 0.20 | 0.707 | no | 12,790 | 0.989 | no | 830,754 |
| regex_alt_1111_or_0000_11 | 0.30 | 0.694 | no | 10,741 | 0.989 | no | 7,085,012 |
| regex_alt_111_or_000_3sym | 0.05 | 1.000 | yes | 2,942 | excl | - | - |
| regex_alt_111_or_000_3sym | 0.10 | 1.000 | yes | 6,013 | excl | - | - |
| regex_alt_111_or_000_3sym | 0.20 | 0.543 | no | 19,786 | excl | - | - |
| regex_alt_111_or_000_3sym | 0.30 | 0.433 | no | 9,890 | excl | - | - |

## 3. The wall: full hyperparameter sweep

A full factorial over CAPAL's three real knobs -- `max_same_samples`,
`suffix_pool_len_max`, `alpha` -- across every cell, all four noise levels, and
three seeds (480 runs). For each (cell, η), how many of the
24 configs (knobs × seeds) converge:

| cell | η=0.05 | η=0.1 | η=0.2 | η=0.3 |
| --- | --- | --- | --- | --- |
| parity_mod9_allowed_3_6 | 20/24 | 11/24 | 1/24 | wall (0.91) |
| regex_subseq_1010101 | 17/24 | 11/24 | 3/24 | wall (0.93) |
| regex_two_1111 | 14/24 | 9/24 | 2/24 | wall (0.87) |
| regex_alt_1111_or_0000_11 | 15/24 | 10/24 | 1/24 | wall (0.89) |
| regex_alt_111_or_000_3sym | 24/24 | 12/24 | 3/24 | wall (0.87) |

**The wall is a property of the noise level, not the DFA.** At η=0.30 every
cell fails on all 24 configs; at η≤0.20 every cell -- modulo included --
is crackable by some config and seed, with the crack-rate falling monotonically
with noise. Convergence rate by η, over all configs:

| η | convergence rate |
| --- | --- |
| 0.05 | 0.75 |
| 0.1 | 0.44 |
| 0.2 | 0.08 |
| 0.3 | 0.00 |

The hyperparameters are near-neutral within the swept ranges (each knob value
moves the aggregate rate by <0.05); **η alone drives convergence from 75% to
0%.** The earlier impression that modulo is uniquely hard was an artifact of
sweeping only `max_same_samples`; adding pool/alpha cracks it at η≤0.20.

## 4. Matched query budget: the wall is structural

CAPAL with its suffix enumeration uncapped (`enum_depth=8`,
`extra_len_max=16`, `suffix_pool_len_max=16`,
`max_same_samples=2000`) on the η=0.30 wall cells, three
seeds, versus E-L*'s spend on the same cell:

| cell | CAPAL acc | conv | CAPAL distinct | E-L* acc | E-L* distinct |
| --- | --- | --- | --- | --- | --- |
| parity_mod9_allowed_3_6 | 0.858 | 0/3 | 2,450,379 | 1.000 | 1,771,791 |
| regex_subseq_1010101 | 0.905 | 0/3 | 846,458 | 1.000 | 2,095,356 |
| regex_two_1111 | 0.867 | 0/3 | 793,899 | 1.000 | 2,217,453 |
| regex_alt_1111_or_0000_11 | 0.752 | 0/3 | 1,232,352 | 0.989 | 7,085,012 |
| regex_alt_111_or_000_3sym | 0.658 | 0/3 | 83,353 | excl | - |

CAPAL never converges (0/3 everywhere) even at 0.08–2.45M distinct queries. On
modulo it spends **more** than E-L* and still fails, while E-L* succeeds at
100%; the regex cells plateau below E-L*'s budget and fail. Throwing queries at
CAPAL does not break the wall: the limiter is the pairwise SAMESTATE test shape,
not the label count.

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

## 6. Bottom line

- On CAPAL's own suite CAPAL is broadly applicable and cheap (100% on 108/112
  cells), degrading only at η=0.30. E-L* matches its accuracy but only on the
  narrow slice its preconditions admit, at 2–3 orders of magnitude more queries.
- On this repo's benchmarks E-L* is accurate and noise-robust where in-regime;
  CAPAL fails to converge on modulo and collapses on the regexes at η≥0.20.
- The high-noise wall is a property of the **noise level, not the DFA**: at
  η=0.30 every structured cell fails on every hyperparameter and seed; below
  that everything is crackable. Modulo is not special.
- The wall is **structural, not a budget limit**: forced to E-L*'s query range
  (2.45M on modulo, exceeding E-L*'s 1.77M) CAPAL still stalls. The limiter is
  the pairwise SAMESTATE test's `1 − 2p₀` signal scaling, not the label count.
