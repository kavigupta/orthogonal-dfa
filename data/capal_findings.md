# CAPAL (ICLR 2026) evaluation on this repo's test oracles

Summary of the investigation into whether Chen, Trivedi, Velasquez's CAPAL
learner (github.com/lkwargs/CAPAL) can solve the DFA-learning problems in
`tests/test_lstar.py` at persistent-noise levels η ∈ {0.05, 0.10, 0.20, 0.30}.

Companion artifacts:
- `data/capal_official_sweep.csv` — main noise-sweep numbers
- `data/capal_noise_sweep_combined.csv` — CAPAL / aalpy-L* / ortho-L* head-to-head
- `scripts/run_capal_official.py` — self-contained reproducer for the sweep
- `scripts/capal_modulo_wall_queries.py` — reproducer for §5 (the modulo η=0.30
  wall), reporting states/acc/converged/time/distinct-queries per config
- `orthogonal_dfa/capal_official/` — adapter that runs upstream CAPAL on our
  oracle-creators

## 1. Main sweep — official CAPAL, default config

`fit()` raises `RuntimeError("Maximum iterations reached without convergence")`
on most cells; the table shows the *last hypothesis* the learner produced.
`*` marks non-converged cells.

```
| oracle (target st)         | eta=0.05     | eta=0.10     | eta=0.20     | eta=0.30     |
| parity_mod9_allowed_3_6 (9)| 0.88 (23st*) | 0.82 (18st*) | 0.88 (17st*) | 0.75 (20st*) |
| regex_subseq_1010101 (8)   | 0.81 ( 8st*) | 0.81 ( 7st*) | 0.81 ( 9st*) | 0.81 ( 6st*) |
| regex_two_1111 (9)         | 0.85 (20st*) | 1.00 (12st)  | 0.63 (10st*) | 0.63 ( 9st*) |
| regex_alt_1111_or_0000_11  | 1.00 (15st)  | 1.00 (29st)  | 0.41 (24st*) | 0.39 (11st*) |
| regex_alt_111_or_000_3sym  | 1.00 ( 7st)  | 1.00 (10st)  | 0.22 (22st*) | 0.10 (17st*) |
```

Only 5 of 20 cells converge, all at η ≤ 0.10 and only on the `regex_alt_*`
patterns. At η ≥ 0.20 the algorithm collapses to base-rate accuracy or worse
on every regex oracle. Modulo-9 never converges at any noise level.

## 2. Head-to-head with the repo's ortho-L\*

```
| oracle              | η    | official CAPAL      | ortho-L*         |
| parity_mod9 (9st)   | 0.05 | 0.88 (23st*, 40s)   | 1.00 (9st, 16s)  |
| parity_mod9 (9st)   | 0.10 | 0.82 (18st*, 10s)   | 1.00 (9st, 22s)  |
| parity_mod9 (9st)   | 0.20 | 0.88 (17st*, 9s)    | 1.00 (9st, 44s)  |
| parity_mod9 (9st)   | 0.30 | 0.75 (20st*, 11s)   | 1.00 (9st, 80s)  |
| regex_subseq (8st)  | 0.05 | 0.81 (8st*,  3s)    | 1.00 (8st, 39s)  |
| regex_subseq (8st)  | 0.10 | 0.81 (7st*,  3s)    | 1.00 (8st, 96s)  |
| regex_subseq (8st)  | 0.20 | 0.81 (9st*,  3s)    | 1.00 (8st, 93s)  |
```

CAPAL is **faster** on almost every cell (up to ~38× on regex_subseq) but
**less accurate** everywhere. Ortho-L\* hits the target exactly.

## 3. Why more iterations don't help

At every failing cell we traced, `|S|`, `|E_core|`, and the hypothesis
state-count all lock at a fixed point well before 200 iters:

```
modulo eta=0.05:                     regex_alt_1111 eta=0.20:
  iter  51: states= 23 |S|=77 |E|=98   iter  10: states= 12
  iter 101: states= 23 |S|=77 |E|=98   iter  50: states= 24
  iter 501: states= 23 |S|=77 |E|=98   iter 200: states= 24
  iter 551: states= 23 |S|=77 |E|=98   iter 1000: states= 24
```

`PerfectEQ` keeps returning the same CE; `maybe_promote_core` returns False
because the suffix is already promoted; the SAMESTATE negative cache keeps
prior DIFFERENT decisions locked; nothing new enters `S` or `E_core`. Pure
no-op spinning. Raising `max_iters` above ~200 changes nothing.

## 4. Which knobs *do* help

Sample sweep on three previously-failing cells (`modulo η=0.05`,
`regex_alt_1111 η=0.20`, `regex_subseq η=0.10`):

**Different seeds (10 seeds, default m=60):**
- `modulo η=0.05`: 6/10 seeds converge to 9-state DFA at 100% acc. Default
  seed 0 lands at 88%/23 states — it was an unlucky draw.
- `regex_subseq η=0.10`: 4/10 seeds converge to 100% acc. Seed 0 lands at 91%
  — same story.
- `regex_alt_1111 η=0.20`: **0/10 seeds converge** at m=60. This cell is
  genuinely hard.

**Bigger `max_same_samples` (seed 0):**

```
| cell                       | m=60               | m=120              | m=240              | m=480              |
| modulo η=0.05              | 0.90 / 23st  40.9s | 1.00 /  9st*  0.2s | 1.00 / 21st* 32.0s | 1.00 / 11st*  5.0s |
| regex_alt_1111 η=0.20      | 0.69 / 24st  33.3s | 0.80 / 16st  26.4s | 0.90 / 13st  41.5s | 1.00 / 22st* 77.0s |
| regex_subseq η=0.10        | 0.91 /  7st   2.4s | 0.91 /  8st   5.1s | 1.00 / 11st*  2.1s | 1.00 / 15st* 18.7s |
```

Bumping `m` from 60 to 240–480 unstuck every cell we tested — including the
"nothing works at m=60" regex_alt cell. Multi-seed best-of-N also gets you
there for 2 of 3, in similar wall time.

So most of the "CAPAL fails" cells in the main table are a **default-config
artifact**, not a fundamental limit — the default `m=60` is just too small
for the harder cells.

## 5. Modulo η=0.30: the wall

Even the resource-heavy configurations can't crack this cell. Re-measured on
current upstream CAPAL (github.com/lkwargs/CAPAL @ 57d877f), modulo η=0.30,
seed=0, K_pos=K_neg=10 (matching the repo adapter); every row non-converged.
Reproduce with `scripts/capal_modulo_wall_queries.py`. `distinct MQ` =
`len(mq.cache)`, the number of *distinct* membership queries — the persistent MQ
caches every string, so this is CAPAL's true oracle cost:

```
| config (seed=0)          | states | acc   | conv | time | distinct MQ |
| m=120,  50 iter          | 27     | 0.777 | No   |  11s |      38 618 |
| m=240,  50 iter          | 30     | 0.690 | No   |  29s |      42 147 |
| m=480,  50 iter          | 25     | 0.724 | No   |  96s |      37 845 |
| m=1000, 50 iter          | 24     | 0.783 | No   | 654s |      24 671 |
| m=4000, 15 iter          | 14     | 0.709 | No   | 456s |      11 798 |
| m=4000 + pool_len_max=14 | 10     | 0.911 | No   | 704s |      16 989 |
```

(Target is a 9-state DFA at 100% acc; no config reaches it. Harness validated
against §2's m=60 cell — 20 states, 12 240 queries — an exact match. An earlier
draft of this table reported lower state counts from an ad-hoc run whose
iteration caps weren't recorded and which doesn't reproduce on current upstream;
the numbers above are the reproducible ones.)

Inspection of the learned hypotheses reveals **the structure is wrong, not just
the acceptance bits**: on all-1s the target cycles 0→1→…→8→0 (period 9), but the
learned DFA collapses this into a shorter period (typically 3), conflating states
that differ by 3 modulo 9 — because the SAMESTATE test can't separate their
empirical disagreement rate from the noise floor.

**Oracle-query cost vs ortho-L\*.** Ortho-L\*'s noise is likewise persistent (a
deterministic hash of the string), so distinct-query count is the fair metric on
both sides. Ortho-L\* solves this exact cell with **2 934 112** distinct queries
(9 states, 100% acc) — so CAPAL hits the wall while spending **~70–250× fewer**
oracle labels than ortho-L\* spends to succeed. Two things stand out in the table:

- **Raising `max_same_samples` doesn't add evidence.** Across the 50-iter rows,
  bumping m from 120 → 1000 leaves the distinct-query count flat-to-declining
  (39 k → 25 k), never rising. Under persistent noise, re-sampling the same short
  suffix returns the *same cached* label — SAMESTATE saturates on the ≤511 binary
  strings of length ≤8, so extra samples add **no new information**.
- **Only the suffix *pool* length adds distinct evidence.** `pool_len_max=14`
  lifts distinct queries 11 798 → 16 989 and drops the state count to 10 (closest
  to the true 9) at the best accuracy in the table (0.911) — yet still no
  convergence.

That raises the obvious question — does simply letting CAPAL draw as many
suffixes as ortho-L\* settle it? §10 tests exactly that; it does not.

## 6. Why the noise floor bites CAPAL more than ortho-L\*

The two algorithms both use statistical row-equality under persistent noise,
but the *test shape* is different:

- **CAPAL SAMESTATE** compares two noisy rows, one against the other:
  `D(u, v) = (1/m) · Σ_e [ y(u+e) ≠ y(v+e) ]`
  with noise floor `p₀ = 2η(1−η)`. Observed signal above floor scales by
  **(1 − 2p₀) · true_D**.

- **Ortho-L\*** measures each prefix's own accept rate against a data-driven
  boundary:
  `mean(x) = (1/|V|) · Σ_e y(x+e)`
  with noise floor `η` (only one noisy bit per cell of evidence). Observed
  signal above floor scales by **(1 − 2η) · true_gap**.

Concretely at each noise level:

```
| η    | CAPAL signal (1−2p₀) | ortho-L* signal (1−2η) | ratio |
| 0.05 | 0.81                 | 0.90                   | 1.1×  |
| 0.10 | 0.64                 | 0.80                   | 1.25× |
| 0.20 | 0.36                 | 0.60                   | 1.7×  |
| 0.30 | 0.16                 | 0.40                   | 2.5×  |
| 0.40 | 0.04                 | 0.20                   | 5×    |
```

At η=0.30 ortho-L\* gets **2.5× more usable signal on the same oracle**, and
the gap widens with noise. That's structural to the pairwise-vs-per-prefix
choice — no hyperparameter closes it.

Ortho-L\* also picks a suffix family that maximises the accept-rate gap
between states (`cluster.identify_cluster_around`) and calibrates the
decision boundary from observed cluster means (`cluster.py:36`), where CAPAL
uses a fixed `p₀ + τ` on a random pool.

## 7. Why raising `α` doesn't rescue the hard cell

`α` is the per-comparison Hoeffding significance. `τ = √(ln(2/α) / 2m)`, so
**lower α → bigger τ → more merges** (backwards from what one might guess),
and **higher α → smaller τ → tighter threshold** but at the cost of a
higher false-DIFFERENT rate per call.

Numerically at m=4000:

```
| α       | τ      | threshold p₀ + τ at η=0.30 |
| 1e-3    | 0.089  | 0.51                        |
| 0.01    | 0.071  | 0.49                        |
| 0.05    | 0.062  | 0.48                        |
| 0.1     | 0.056  | 0.48                        |
| 0.5     | 0.039  | 0.46                        |
```

Modulo-9's hardest pair has observed `D ≈ 0.46`. Only α ≈ 0.5 (50%
per-comparison false-DIFFERENT rate) gets the threshold to that observed D
— and at that α the algorithm blows up state count chasing ghosts.

## 8. Why the majority-vote acceptance also drifts

When a class has no CE-derived gold label, acceptance falls back to majority
of `y(member)` over class members. With persistent noise η and k members,
the P(majority wrong) depends on k *and* on the tie-break rule
`sum·2 ≥ k` (which favours accept on ties):

```
| k | P(wrong) if truly accept | P(wrong) if truly reject |
| 1 | η = 0.30                 | η = 0.30                 |
| 2 | η² = 0.09                | 1−(1−η)² = 0.51          |
| 3 | 0.22                     | 0.22                     |
| 5 | 0.16                     | 0.16                     |
```

Mod-9's hypothesis at iter 15 had 21 prefixes over 11 states, so k ≈ 2 on
average and truly-reject classes get flipped **~51%** of the time. Adding
S-members grows k, which fixes this, but happens slowly.

## 9. Predicted minimum for modulo-9 η=0.30 (untested)

To reliably declare all mod-9 hard pairs DIFFERENT (per-pair success ≥95%,
whole-run ≥60%), the Hoeffding requirement is roughly

```
τ ≈ 0.015  →  m ≥ ln(2/α) / (2·0.015²) ≈ 17 000  at α=1e-3
                                        ≈  8 000  at α=0.05
```

Ballpark config that *might* land it, at maybe 40–60% success per seed:

```python
LearnerConfig(
    eta=0.30, max_iters=50,
    alpha=0.05,
    max_same_samples=16000,
    suffix_pool_init=8000,
    suffix_pool_len_max=12,
    tau_cap=0.04,
    discr_search_max_len=10,
    discr_search_random=2000,
)
```

Estimated cost per seed: ~1–2 hours wall. Ortho-L\* on the same cell: 80s.

## 10. Tested: matching ortho-L\*'s query budget doesn't break the wall

§9 predicts a *bigger* `m` might reach the cell. We tested the underlying
question directly — **is the wall just a suffix budget/length limit?** — by
un-capping the two knobs `LearnerConfig` never exposes: `enum_depth`
(systematic-enumeration depth, default 3) and `extra_len_max` (random-fill
suffix length, default 8). With these raised, SAMESTATE probes thousands of
*long* suffixes per pair instead of a handful of short ones.

```
| config (enum/extra/pool), m       | states | acc   | distinct MQ | conv |
| baseline  3/ 8/ 8, m=500  (best)  | 15     | 0.90  |     ~20 000 | No   |
| deep-enum 8/16/16, m=2000 seed 0  | 29     | 0.706 |   4 916 527 | No   |
| deep-enum 8/16/16, m=2000 seed 1  | 27     | 0.879 |   3 257 673 | No   |
| deep-enum 8/16/16, m=2000 seed 2  | 27     | 0.837 |   4 603 687 | No   |
```

Un-capping the suffix length did exactly what it should *mechanically*: the
distinct-query count jumped from ~20 k to **3.3–4.9 M — as many as, or more
than, ortho-L\*'s 2.93 M**. But the wall didn't move: accuracy stayed
0.71–0.88 and the state count got **worse** (27–29 vs baseline's 15–17), still
never converging to the true 9. A pool-length-only sweep (`pool_len_max`
8→32, m=500) tells the same story: best-of-3-seeds accuracy peaks weakly at
pool_len≈16 (0.91) then *declines* at 24–32, never converging.

So short suffixes are **not** the fixable bottleneck. Two effects compound:

1. **Longer suffixes are less discriminating for modulo counting.** For the
   pairs CAPAL merges (states differing by ±3 mod 9), a suffix with `c` ones
   separates them for only 2 of the 9 residues of `c mod 9` — so the *maximum*
   true disagreement is `2/9 ≈ 0.22`, reached only when `c` is uniform (i.e.
   long suffixes). At η=0.30 that gives observed `D = p₀ + (1−2p₀)·0.22 ≈
   0.455`, only 0.035 above the p₀=0.42 noise floor. Short suffixes are worse
   still: their `c` is concentrated near 0, so `true_D` sits below the 2/9
   ceiling.
2. **You can't tighten τ for the hard pair without over-splitting the easy
   ones.** Resolving a 0.035 excess needs `τ < 0.035`, i.e. m ≳ 3 000 distinct
   long suffixes per pair. But a τ that small drops the DIFFERENT threshold
   toward p₀, so ordinary noise fluctuations on *already-correct* pairs also
   cross it — the 27–29-state blow-up. The pairwise test has one global knob
   (τ) and the hard and easy pairs want opposite settings.

This makes §6/§7's thesis concrete: at a **matched** query budget CAPAL still
fails, because the limiter is the *shape* of the pairwise SAMESTATE test, not
the number of labels drawn.

## 11. Query efficiency on the *easy* cells — what ortho-L\* should borrow

§5 shows ortho-L\* winning the hard cell by out-spending CAPAL ~180×. But run
the query comparison on the *easy* cells and the sign flips — ortho-L\* is the
profligate one everywhere. Distinct membership queries, ortho-L\* vs official
CAPAL (default m=60, seed=0; same persistent-noise metric on both sides, and
CAPAL here has a free `PerfectEQ` — see the caveat below). The `old→new` column
brackets the recent ortho-L\* query-reduction work already on `main` (#110
sequential accuracy early-stop, #112/#113 evidence-cache reuse):

```
| cell                       | η    | ortho MQ (old→new) | ortho st | CAPAL st | acc   | conv | CAPAL MQ | ratio |
| regex_alt_1111_or_0000_11  | 0.05 |   436 896 → 447 259 | 10       | 15       | 1.000 | Yes  |   17 483 |   26× |
| regex_alt_1111_or_0000_11  | 0.10 | 1 604 785 → 756 932 | 10       | 29       | 1.000 | Yes  |   21 810 |   35× |
| regex_subseq_1010101       | 0.05 |   887 885 → 293 989 |  8       |  8       | 0.911 | No   |    6 154 |   48× |
| regex_subseq_1010101       | 0.10 |   891 078 → 490 804 |  8       |  7       | 0.911 | No   |    5 652 |   87× |
| parity_mod9                | 0.05 |   501 944 → 405 797 |  9       | 23       | 0.901 | No   |   15 517 |   26× |
| parity_mod9                | 0.10 |   715 766 → 584 259 |  9       | 18       | 0.866 | No   |    9 917 |   59× |
```

Even after `main`'s optimisations (which cut 0–67%, biggest on subseq), ortho-L\*
spends **0.3–0.76 M** distinct queries per cell vs CAPAL's **6–22 k** —
**26–87× more**. On `regex_alt`, where both hit 100%, CAPAL is *strictly*
better: exact **and** ~26–35× cheaper. Ortho-L\*'s edge is robustness (100% on
every cell); it buys that with a 1–2 order-of-magnitude larger oracle budget. Note
distinct ≈ total oracle calls now (e.g. 447 k vs 461 k), so caching is nearly
complete — the remaining cost is genuine *distinct* information, not redundant
re-queries.

**Where do ortho-L\*'s queries go?** Not into the table. Instrumenting the phases
on the merged code (regex_alt / parity, η=0.05):

```
| phase                                  | distinct MQ |
| PrefixSuffixTracker.create (the table) |           0 |
| counterexample-driven synthesis loop   |  all of them |
```

The suffix population is tiny (`n=21`); building the evidence table costs **zero**
new queries. **~100% of the spend is the synthesis loop** — `generate_counterexamples`
and `estimate_agreement_rate`, which draw *fresh* random words each iteration and
run `locate_incorrect_point`, a per-sample binary search that classifies O(log len)
prefixes, each querying the whole suffix family at every DT node. Fresh words →
fresh (prefix+suffix) strings → new distinct queries, with limited reuse across
rounds.

**Why CAPAL is cheaper — two causes, one of them not real:**

1. **Free equivalence oracle (the dominant factor, and a harness artefact).**
   CAPAL here runs with `PerfectEQ`, which finds counterexamples by product-BFS
   against the *true* target — **zero** membership queries. `len(mq.cache)`
   therefore counts only CAPAL's *state-distinction* work; its counterexample
   discovery is free. Ortho-L\* has no such oracle and pays for CE discovery by
   sampling — which, per the breakdown above, is its *entire* bill. Give CAPAL a
   sampling EQ (the repo's `RandomWordEqOracle`) and its count would climb toward
   ortho-L\*'s; this comparison flatters CAPAL.
2. **Compact, reused evidence (a real, portable advantage).** CAPAL's SAMESTATE
   reuses one small suffix set (≤ a few hundred short strings) across *all* prefix
   pairs, cached persistently — so its state-distinction cost stays at ~10 k even
   as prefixes accumulate. Ortho-L\*'s CE/verification sampling generates a fresh,
   never-reused word each time.

**Porting CAPAL's compactness into ortho-L\*.** Two of the four ideas below are
already on `main` and account for the `old→new` reduction; two remain, and both
target the synthesis loop (the 100%-of-cost phase) without touching the noise-robust
per-prefix test that wins §5:

- ✅ **Sequential accuracy estimation** (`main` #110). Replaces the fixed
  2000-sample estimate with an early-stopping sequential binomial test — the bulk
  of the 0–67% reduction above.
- ✅ **Evidence-cache reuse** (`main` #112/#113). Reuses the prefix×suffix mask
  cache and the constant empty-prefix classification, driving total ≈ distinct.
- ⬜ **A shared, cached probe pool.** Draw one fixed pool of counterexample/accuracy
  words and reuse it across synthesis rounds, instead of `us.sample(...)` fresh each
  iteration — caps distinct queries at |pool| × (suffixes per classification)
  regardless of round count. Risk: a fixed pool may stop surfacing counterexamples
  as the hypothesis sharpens (the reason CAPAL needs its `PerfectEQ`), so refill on
  hypothesis change and **measure that accuracy holds**.
- ⬜ **Structural counterexample generation, CAPAL-style.** Instead of blind random
  sampling, walk the current hypothesis DFA to synthesise *targeted* distinguishing
  candidates (short words reaching under-tested transitions), then confirm each with
  the per-prefix test. Fewer, higher-yield probes — closest to CAPAL's real edge and
  it preserves CE-finding power a fixed pool risks losing.

Net: ortho-L\*'s noise robustness lives in the *table* (already free); its query
blow-up lives entirely in *counterexample discovery and verification*. That is
exactly where CAPAL is compact — so the two remaining ports are orthogonal to, and
should not weaken, the high-noise advantage.

## 12. Bottom line

- CAPAL as shipped works cleanly at η ≤ 0.10 on structurally simple
  regex-alternation DFAs. It's fast — up to 38× faster than ortho-L\* on
  cells where both succeed.
- The main "CAPAL fails" cells in the sweep are default-config artefacts:
  bumping `max_same_samples` from 60 → 240–480 or picking a lucky seed
  rescues most of them.
- At η ≥ 0.20 on structured DFAs (modulo-K, regex with rare witnesses),
  CAPAL hits a signal-to-noise wall that no hyperparameter fully overcomes.
  The `1 − 2p₀` scaling in its pairwise test is the culprit.
- Ortho-L\* uses a per-prefix accept-rate test with `1 − 2η` scaling, giving
  it a 2.5×-to-∞ signal advantage in the high-noise regime, plus a
  data-driven decision boundary. It solves every tested cell at 100% acc.
- Modulo-9 at η=0.30 is not reachable with CAPAL at any tested (m, α,
  pool_len_max, seed) combination we tried; theory suggests m ≈ 16k would
  make it *possible* per-seed but at 100–1000× the ortho-L\* wall time.
- The wall is not a query-budget limit. In the §5 configs CAPAL spends ~12–42k
  distinct oracle labels vs ortho-L\*'s 2.93 M. But *forcing* CAPAL to a matched
  budget — deep suffix enumeration pushes it to 3.3–4.9 M distinct queries (§10)
  — still fails and inflates state count to ~28. The bottleneck is the pairwise
  SAMESTATE test shape, not the number of labels.
- On the *easy* cells the efficiency sign flips (§11): even after `main`'s query
  cuts, ortho-L\* spends 0.3–0.76 M distinct queries vs CAPAL's 6–22 k (26–87×
  more), and on `regex_alt` CAPAL is both exact and ~26–35× cheaper. ~100% of
  ortho-L\*'s spend is the counterexample-discovery / accuracy-estimation loop (the
  table itself is free); CAPAL's edge is partly a free `PerfectEQ` and partly genuine
  evidence reuse. Two of the four ports are already landed (#110, #112/#113); the
  remaining two — a shared cached probe pool and targeted CE generation — are the
  concrete next step, orthogonal to the noise-robust test that wins §5.
