# Neural approaches to noisy L*: findings

Two attempts to replace the discrimination-tree + suffix-family machinery with a learned
model. **Neither beats L\*.** This records what was measured, so the negative results are not
re-derived.

Code: `orthogonal_dfa/l_star/neural/`. Scripts: `scripts/train_neural_dfa.py`,
`scripts/bench_surrogate.py`, `scripts/diagnose_surrogate.py`.

Throughout, two different quantities are reported and must not be compared with each other:

* **model fidelity** — the neural model's own balanced accuracy on held-out data.
* **pipeline accuracy** — the extracted DFA against a noiseless oracle, via
  `tests.test_lstar.evaluate_accuracy` (10k uniform length-40 strings).

A pipeline number can be high while the model underneath is poor, because the split/accept
decisions are statistical tests on observed cells and the model only proposes.

Base rates (a constant predictor scores this): subseq 0.813, two_subseq 0.631,
modulo9 0.773, parity 0.500.

---

## Approach 1 — amortised state predictor

`model.py`, `objective.py`, `train.py`. `m(x)` gives a distribution over latent states for
every prefix in one pass, trained on `J_internal` (determinism) + `J_external` (labels).
Nothing backpropagates through 40 soft transition matrices, unlike `utils/pdfa.py`.

**Result:** parity exactly (1.0, 2 states, 25k distinct queries). Every language whose Nerode
partition is finer than its acceptance partition fails at base rate.

**Cause, measured.** `J_external` sees `m` only through a per-state accept scalar, so it is
*exactly invariant* under any refinement of the acceptance partition — zero gradient toward
refining it. `J_internal` is then left to choose among all refinements, where the Nerode
partition beats the bare acceptance partition by **0.015** (0.9853 vs 1.0, the residual
nondeterminism carrying little mass) while language-independent congruences like length-mod-k
win by ~**log 2**. The larger signal wins.

Not an information shortage: the same encoder *without* the state bottleneck reaches 1.0
prefix accuracy on the same 88k labels.

Four bugs that had to be fixed before parity worked at all:

| bug | symptom | fix |
|---|---|---|
| trivial-congruence lock-in | learns "length mod 2", acc 0.50 | warmup: `J_external` alone first |
| Gram penalty `‖G−I/S‖²` | forces state *duplication*, caps `J_int` at 0.97 | off |
| empty prefix is a population of size 1 | one noise-flipped label inverts the whole trajectory | inverse-multiplicity weighting; initial state by self-consistency |
| `accept_logits = 0` | `p` constant in `m`, softmax Jacobian zeroes the gradient exactly | random init |

---

## Approach 2 — table surrogate

`surrogate.py`. Models the observation table directly:

```
M[p][v] = Σ_s m(p)[s] · σ(⟨u_s, φ(v)⟩)
```

Rows and columns are **encoders over strings**, not free per-row embeddings, so the model
generalises to prefixes and suffixes never queried — the difference from ordinary matrix
completion.

### Why the table is the right object

`scripts/profile_query_origins.py` on subseq, 1,277,963 queries:

| queries | share | origin |
|---|---|---|
| 641,680 | **50.2%** | one new suffix column × all 14,584 prefixes ← `_resolve` |
| 180,255 | 14.1% | same, ← `_sample_suffix` |
| 135,000 | 10.6% | new prefix row × all suffixes |
| 320,475 | 25.1% | DT classification |

~75% is literally filling cells of `M`; the rest is the same cells for rows not yet in the
table. The final table is 14,584 × 61 ≈ 890k cells with **only 8 distinct rows**.

### Model fidelity

| claim | measurement |
|---|---|
| it denoises | held-out cells **0.973** vs **0.8** for reading each cell's own noisy label, at 50% density |
| recovers state structure | MI **1.95 / 2.45** bits, purity **0.866** (subseq, 8 states) |
| over-provision `num_states` | `S=32` purity 0.861 vs `S=8` 0.663 — matching the true count is *worse* |

### Pipeline accuracy, 3 seeds

| config | parity | modulo9 | subseq | two_subseq |
|---|---|---|---|---|
| merge path (committed) | 1.0 ×3 | **1.0**/0.888/0.773 | 0.813/**1.0**/0.813 | 0.631 ✗ |
| split-from-blob | 1.0 ×3 | 0.888/0.505/0.888 | 0.813 ✗ | 0.631 ✗ |

14 targets (6 hand-written + 8 generated at 10 and 18 states): **4/14 ≥ 0.97**, median 0.855,
median ~34k queries. L\* spends 869k (modulo) and 1.23M (subseq) distinct queries.

Variance is high enough that single-seed results are meaningless — an early "modulo9 solved at
1.0" did not replicate.

---

## The two robust principles

### 1. Max over columns, never mean

Separating two states typically needs **one specific suffix out of forty**. Any average
divides that signal by 40 and buries it. This one distinction explains four separate failures:

* `merge_by_row` (mean row distance) — over-merges; subseq collapses to 1 state.
* successor assignment by mean distance — most blocks unreachable.
* lagged targets `A[s,k]` marginalised over all length-`k` suffixes — nearly invariant.
* `J_external`'s scalar accept head — exactly invariant.

Measured on subseq: model gap on distinguishing columns **0.510**, elsewhere **0.031**. Each
column separates ~12 of 28 state pairs; 22/28 pairs are covered.

**Not** implicated: averaging *within* a column (over cells at a fixed suffix) is the pooling
that denoises, and is load-bearing.

### 2. Column structure beats quantity

Matched at exactly 36,000 oracle queries, parity, model fidelity:

| layout | balanced |
|---|---|
| 36,000 rows × 1 column | 0.476 |
| 1,500 rows × 24 columns | **1.0000** |

A prefix is placed by its response across 24 suffixes that 1,499 other prefixes already
calibrated. With one column there is nothing to triangulate against.

Caveat: modulo9 scores 0.5000 in *both* arms of this test, which contradicts the plain-GRU
control (0.787) and is unexplained. The likely suspect is that this harness derives the accept
rate through the column encoder as `sigma(<u_s, phi(eps)>)` rather than from a free per-state
scalar -- the same reparameterisation that was independently found to flatten every state's
rate at once. So the parity row is the load-bearing one here.

---

## Data regime

### Short prefixes are required (seeded)

Prefix-length distribution, fixed 36,000-query budget, success = fraction of seeds > 0.9:

| cell | `long` (11–40) | `mixed` (0–40) |
|---|---|---|
| GRU | 1/9 | 8/8 |
| LSTM | 0/4 | 4/4 |

**Not an architecture limitation** — LSTM fails identically. Short prefixes act as a
curriculum into the counting behaviour. This gives `short_prefix_closure` a second
justification beyond transient-state coverage.

Outcomes here are bimodal (≈0.5 or ≈1.0), so the success *rate* is the meaningful statistic,
not the mean.

### Hankel augmentation does nothing

`M[p][v]` depends only on `p+v`, so one query on a length-40 string populates all 41 split
points. Cells 6,000 → 246,000, and **no change** (parity/modulo9 both 0.5 either way; subseq
0.9994 → 0.9936). The 41 splits carry the *same* label — 41 copies of one bit, no new
information. Querying each prefix separately gives 41 *different* labels and is what matters.

### Modulus sweep — terminal supervision, 6,000 strings, length 40

| language | balanced |
|---|---|
| mod 4, accept {0,1} (contiguous) | 0.4733 |
| mod 4, accept {0,2} (alternating) | 0.4626 |
| mod 2 / 3 / 5 / 7, accept {0} | 0.463 / 0.517 / 0.493 / 0.490 |
| mod 9, accept {0} | **0.7672** |

Run to test whether difficulty tracks the *frequency* of the accept function in count-space.
It does not: contiguous vs alternating at fixed modulus is the decisive comparison and shows
no difference, and the trend across `k` is not monotone (mod 7 fails, mod 9 works).

A hypothesis that does fit all seven numbers, untested: over length-40 strings the sum
concentrates near 20±3, and accepting sums in that range number ~8 for mod 2, 6 for mod 3, 3
for mod 5 and 7, but only {18, 27} for mod 9 -- with 27 deep in the tail. So mod 9 reduces to
roughly "is the sum about 18", a single unimodal bump, while the rest need an oscillating
readout.

This entire sweep is in the terminal-supervision regime, which the pipeline never encounters
(it gives ~24 labels per prefix). Treat it as a fact about the harness, not about the
pipeline.

### Architecture controls — 6,000 strings, one label each, length 40

| | plain GRU | + softmax bottleneck | + clamp to [0.2, 0.8] |
|---|---|---|---|
| parity | 0.463 | 0.477 | 0.507 |
| modulo9 | **0.787** | 0.751 | **0.500** |

**The clamp breaks modulo9.** It was added to stop memorisation (predicted rates piling at
0.0/1.0 instead of the noise rates, found by plotting) and it does — at the cost of flattening
the loss landscape by 0.6× through the head. Memorisation should be prevented by cluster
population instead.

Per-step supervision (41 labels per string, one per prefix) gives parity **1.0000** and
modulo9 **0.9985**, versus 0.497/0.574 with terminal supervision.

---

## Rejected variants

Each is retained as a defaulted-off knob with the measurement that rejected it.

| variant | why it fails |
|---|---|
| `counterexample_prefixes` | compares the DFA against the surrogate it was *extracted from* — measures extraction error, not language error. 4/14 → 3/14. |
| `error_boost` | the flag is 48% precise (DFA errs 18.7%, noise flips 20%); per-string noise cannot be averaged, so boosting re-amplifies single fixed labels. Breaks parity at every level. |
| `merge_by_row` | mean over 40 suffixes dilutes a single distinguishing column. |
| `refine_partition` | Moore refinement splitting on *any* successor disagreement; the assignment is a noisy nearest-centroid classification, so blocks shatter. Parity → 14–21 states at 0.49. |
| MI-form `J_internal` | refuses collapse as designed, then selects language-independent congruences (exactly `log 2`, `~log 3`). |
| `resolve_transitions` | assigning a successor to any group it *fails to reject* — failing to reject is not evidence of sameness. Parity 1.0 → 0.500; → 0.875 with a best-match rule. |
| `minify` | merges behaviourally equivalent states, which assumes the transition table is correct. Here it is estimated, so states whose transitions are wrong *in the same way* get merged — turning a transition error into total collapse. `transition_resolver` does not minify either; its DT leaves are separated by real distinguishing suffixes. Replaced by `trim_unreachable`. |

---

## Open

* **Transitions.** With `trim_unreachable` and the max-over-columns assignment, blocks reach
  the true state count (modulo9 8–11 vs true 9) but accuracy does not follow. On modulo9 the
  blocks are 83% residue-pure yet systematically merge `r` with `r+3`. Not power (z ≈ 11.7
  available) and not missing columns (2/9 of random suffixes separate those pairs).
* **Splitting vs merging.** Splitting from one blob is better founded — merging does 496
  pairwise tests corrected only across columns, so chance differences become permanent states
  — but measures worse end to end.
* **Held-out testing for splits.** Proposing and testing on the same cells is circular and
  inflates splits; testing on held-out cells removes that and under-splits. Cross-fitting
  (4 folds) did not resolve it.
* **Dead code.** `merge_by_row`, `merge_by_conformal`, `merge_by_counts`, `refine_partition`
  are no longer on the live path; `q_hat` from `conformal_rate_bound` is computed and feeds no
  decision.
* **Unvalidated upstream heuristics.** Enrichment, suffix proposal and uncertainty sampling
  were each added mid-investigation and never A/B'd.

---

## Corrections

Claims made during this work that later measurement refuted, recorded so they are not
repeated:

* "Training on all prefixes covers transient states for free" — true of *frequency*, false of
  *information*. Short prefixes are tiny noise-dominated populations; the empty prefix is a
  population of size one.
* "Cell cross-entropy buries the 3-of-40 signal" — refuted. The model's gap on distinguishing
  columns is 0.510 vs 0.031 elsewhere. The objective was never the problem; the decision rules
  downstream were.
* "DFA extraction needs an exactly correct partition" — false; L\* makes population decisions
  throughout and tolerates misclassification.
* "The encoder cannot represent sum mod 9" — contradicted by block purity 0.831.
* "Counting languages are hard for gradient descent" (leap complexity, then spectral bias) —
  both refuted. A plain GRU learns modulo9 at length 40 (0.787), and per-step supervision
  gives parity 1.0000. The failures came from a fidelity harness that gave 1 bit per prefix
  where the pipeline gives ~24.
* "mod 4 contiguous vs alternating should differ" — no difference (0.473 vs 0.463).
