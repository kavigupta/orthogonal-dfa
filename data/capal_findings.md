# CAPAL (ICLR 2026) evaluation on this repo's test oracles

Summary of the investigation into whether Chen, Trivedi, Velasquez's CAPAL
learner (github.com/lkwargs/CAPAL) can solve the DFA-learning problems in
`tests/test_lstar.py` at persistent-noise levels η ∈ {0.05, 0.10, 0.20, 0.30}.

Companion artifacts:
- `data/capal_official_sweep.csv` — main noise-sweep numbers
- `data/capal_noise_sweep_combined.csv` — CAPAL / aalpy-L* / ortho-L* head-to-head
- `scripts/run_capal_official.py` — self-contained reproducer for the sweep
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

Even the resource-heavy configurations can't crack this cell:

```
| seed=0                      | states | acc   | conv | time  |
| m=120                       | 11     | 0.642 | No   | 7s    |
| m=240                       | 32     | 0.833 | No   | 102s  |
| m=480                       | 24     | 0.874 | No   | 247s  |
| m=1000, 50 iter (killed)    | ≥23    | ~0.91 | No   | ≥550s |
| m=4000, 15 iter             | 11     | 0.771 | No   | 410s  |
| m=4000 + pool_len_max=14    | 16     | 0.815 | No   | 673s  |
```

Inspection of the m=4000 hypothesis reveals **the structure is wrong, not
just the acceptance bits**: on all-1s the target cycles 0→1→…→8→0 (period
9), but the learned DFA either cycles at period 3 (states {1, 2, 5}) or —
with the longer pool — has a period-3 tail (states {3, 5, 7}) after a short
correct head. The algorithm is conflating states that differ by 3 modulo 9,
because the SAMESTATE test can't separate their empirical disagreement rate
from the noise floor.

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

## 10. Bottom line

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
