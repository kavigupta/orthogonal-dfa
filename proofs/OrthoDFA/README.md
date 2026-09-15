# Machine-checked correctness of the E-L\* clustering algorithm

A Lean 4 + Mathlib formalization of the clustering/gate algorithm's correctness
(the `sample_suffix_family` / certification path, PR #257). Everything below is
proved from Mathlib — **no bespoke axioms, no `sorry`**. Each theorem carries a
`#print axioms` line; all report only Lean's core (`propext`, `Classical.choice`,
`Quot.sound`).

## What is proved

The clustering guarantee is stated in the algorithm's own terms — prefixes, oracle
reads, and membership bits `1[x∈L]`. No DFA, no Myhill–Nerode: the clustering
operates at the ε anchor, where the vote denoises membership directly.

Model: **random classification noise** — `MQ(x) = ℓ(x) ⊕ r(x)`, where `ℓ(x) =
1[x∈L]` is the true label and `r(x) ∼ Bernoulli(η)` is an iid persistent noise bit
(one per string); signal `s = ½ − η`. So a member reads 1 w.p. `1−η = ½+s`, a
non-member w.p. `η = ½−s`. A vote of `k` suffixes averages `k` such reads;
everything downstream derives from `E[r] = η` and independence across strings.

| Theorem | File | Statement |
|---|---|---|
| `wrongDecisive_le` | `Hoeffding.lean` | `k` independent `[0,1]` reads, total mean ≤ `kβ`: the vote sum exceeds `k(β+τ)` with prob ≤ `exp(-2kτ²)`. (Mathlib sub-Gaussian Hoeffding.) |
| `certErr_bound` | `Discharge.lean` | A population's side drifted past tolerance (avg mean ≤ `β+τ`) is admitted with prob ≤ `α`. **Certification kernel.** |
| `misplacedMember_le` | `Discharge.lean` | A member is not decided ACCEPT with prob ≤ `exp(-2k(s-τ)²)`. **Decisiveness/placement.** |
| `misplacedNonmember_le` | `Discharge.lean` | A non-member is not decided REJECT with prob ≤ `exp(-2k(s-τ)²)`. |
| `twoSided` | `Estimate.lean` | Empirical rate over `m` fresh prefixes is within `γ` of the distributional rate except w.p. `2·exp(-2mγ²)`. **Sample → distribution (#257 resampling).** |
| `clustering_correct` | `Top.lean` | Union bound: total failure ≤ `#placement · exp(-2k(s-τ)²) + #populations · α`. **The per-population guarantee of PR #257.** |
| `cleanAdmit_le`, `apLowFNR_le` | `Gate.lean` | The gate accepts an accept-preserving family: a clean side fails admission, or an AP family's empirical FNR exceeds `δfnr`, only w.p. `exp(-2m·…²)`. **Gate model (accept side).** |
| `geometric_miss`, `geometric_miss_triggered` | `Termination.lean` | Over `N` rounds, all miss w.p. ≤ `(1-p)^N`. The *triggered* form allows a round's good-event to depend on the whole accumulated history; only a block-local trigger need be independent. **"Eventually", without the independent-rounds idealization.** |
| `algorithm_correct`, `algorithm_correct_general` | `Algorithm.lean` | Joint product measure over `N` rounds: `P[failure] ≤ (1-p)^N + N·b`. The `_general` form lets `goodErr`/`badErr` depend on the accumulating pool (fresh strings added each round), only the findability trigger block-local. **End-to-end.** |
| `clustering_algorithm_correct` | `Algorithm.lean` | The capstone: same bound with the certification input `hbad` **discharged from `certErr_bound`**. Inputs reduce to the oracle model + findability. |
| `clustering_algorithm_correct_fused` | `Capstone.lean` | Both inputs discharged: `hgood` from `fuse_findability`, `hbad` from `certErr_bound` (independent-rounds instance). |
| `clustering_algorithm_correct_general` | `Capstone.lean` | **Accumulating-pool capstone**: good-pass depends on the whole history, only the findability trigger block-local; `hbad` discharged from `certErr_bound` on each round's fresh reads. Removes the independent-rounds idealization end-to-end. |
| **`clustering_pac`** | `Distributional.lean` | **The distributional clustering theorem (PR #257).** Prefixes come from a *collection* of populations `D : J → Measure S`; candidate suffixes are drawn from `Dsf`; **findability** is the single probability `pAP` that a drawn suffix is accept-preserving on the true noiseless oracle; the seed is `ε = 1`. Conclusion: except w.p. `δ`, the returned family preserves acceptance on `≥ 1 − k·εpop` of **each** population. `good`/`bad` are the *distributional* flip-mass conditions, so the separation is **definitional**, not assumed; the greedy is the defined `leastLossSubset`; read independence is derived from the product. |
| `selection` | `Distributional.lean` | Over draws indexed by `ι` with per-coordinate populations (`ι = J × Fin m`, coordinate `(j,i)` drawn from `D j`), the greedy's least-loss `k`-subset avoids every candidate whose summed distributional flip-mass is `≥ σ`. Candidates carry a suffix assignment `sfx : C → S`, so taking `C = Fin M` (draw indices) keeps the candidate set fixed when suffixes are drawn. |
| `rdAt`, `flipMass`, `rdAt_mean` | `Distributional.lean` | The read at a **drawn** prefix and the `Dj`-flip-mass. `rdAt_mean`: `E[rdAt] = η + (1−2η)·flipMass`. Folding the prefix draw into the read means a **single** Hoeffding level gives the distributional result — no separate sample→distribution step. Modelling a draw as an independent `(prefix, noise)` pair makes read independence *derivable* (`iIndepFun_pi`). |
| `coverage`, `coverage_of_summed_flip` | `Distributional.lean` | Per-suffix flip-mass control ⇒ the family preserves `≥ 1 − #F·ε` of **each** population (union bound; summed control gives per-population control since flip-masses are nonnegative). This is PR #257's "hold every population". |
| `prod_le_of_slice`, `measurableSet_of_countable_slices` | `Distributional.lean` | The plumbing for a **randomly drawn** candidate pool: a uniform slice bound transfers to the product, and with `S` countable/discrete (strings over a finite alphabet) a set of pairs is measurable once its `S`-slices are. |
| `goodCount_le`, `index_event_le` | `Distributional.lean` | The two error sources: findability quantified (the count of accept-preserving draws concentrates) and the per-draw-index separation-failure event, bounded uniformly in the draw. |
| `tail_le`, `geom_le` | `Complexity.lean` | **Sample complexity, explicit.** `c·exp(−2kt²) ≤ ε` once `k ≥ log(c/ε)/(2t²)`; `(1−q)^N ≤ ε` once `N ≥ log(1/ε)/q`. Turns "for `k,m,N` large enough" into explicit thresholds (`k` via `t=s−τ`, `m` via `t=γ`, `N` via `q=(1−reject)pp`). |
| `clustering_pac` | `PAC.lean` | **End-to-end PAC bound.** Composes `algorithm_correct_general` + `geom_le`: if each round is a good pass w.p. ≥ `p` (liveness) and admits a bad family w.p. ≤ `b` (soundness), then for `N ≥ log(2/δ)/p` and `N·b ≤ δ/2` the failure probability is ≤ `δ` — with prob ≥ 1−δ a good family is produced and no bad one admitted. |
| `sepFail`, `sepFail_prob`, `sepFail_measurable`, `not_bad_of_not_mem_sepFail` | `Liveness.lean` | **The separation trigger.** `sepFail` is the reads' "fail to separate the classes" event (`dSepCompl` at the greedy bands, where `ρlo+γ = ρhi-γ`); its complement is the block-local trigger. Measurable (reads are), forces the greedy to avoid `bad` off it, and fires except w.p. `#cands·exp(-2m((½−η)εcov)²)`. These are what let `clustering_end_to_end` discharge the good side. |
| `dSepCompl`, `dSepCompl_prob`, `dSepCompl_measurable`, `avoids_bad_of_not_mem_dSepCompl` | `Liveness.lean` | **Separation event, at the abstract-read level.** The reusable core of the greedy's liveness: probability bound (union bound + concentration), measurability, and that avoiding it forces the least-loss subset to avoid `bad`. `chosen_avoids_bad_whp` is now a three-line corollary. |
| `chosen_accept_preserving`, `chosen_accept_preserving_whp` | `Liveness.lean` | **Liveness core.** The ε-anchored greedy (least-loss `k`-subset) proposes an all-accept-preserving family — deterministically under loss-separability, and w.p. ≥ 1 − #cands·exp(−2mγ²) under *mean-loss separability* + concentration. This *derives* the good-pass instead of assuming it. |
| `liveness_produces_good` | `Liveness.lean` | **Liveness, fused.** Separability ⇒ the round produces a family that is accept-preserving *and* clears the gate (a good pass), except w.p. ≤ #cands·exp(−2mγ²) + qgate (qgate the gate-reject bound from `cleanAdmit_le`/`apLowFNR_le`). |
| `Oracle` (structure) | `Liveness.lean` | The persistent oracle, every field a function of a **single** query string `w : S` (no (prefix,suffix) pair, no concatenation inside it): `label w = 1[w∈L]` the noiseless membership bit, and `noise w` a {0,1} bit iid `Bernoulli(η)` over distinct strings (bit-valuedness asserted, `[0,1]` derived); the membership query is `label ⊕ noise`. `flip v p = ℓ(p·v) ⊕ ℓ(p)` (does `v` flip `p`'s acceptance) is **derived** from the single-string label — so `flip_bit` (it is a {0,1} bit) is derived too, not asserted. Strings are Mathlib's own theory — `[Mul S]` concatenation, `[IsRightCancelMul S]` right-cancellation (`mul_left_injective`), free monoid `FreeMonoid`/`List` one instance, `S` abstract. The read `= flip v (pref i) ⊕ noise(pref i · v)` — prefixes an injective enumeration `pref : ℕ → S` — and its mean/independence (via `iIndepFun.precomp`, distinctness from `mul_left_injective ∘ pref`)/range are **derived**, not fields. |
| `greedy_picks_good` | `Liveness.lean` | **Clean-premise liveness.** Takes one `Oracle` over a right-cancellative string type + an injective prefix enumeration + language/findability — no `trigger`/`himplies`, no asserted read law. Loss means derived from the oracle; the greedy's least-loss `k`-subset avoids the bad set (flip ≥ ε_cov) except w.p. `#cands·exp(−2m·((½−η)ε_cov)²)`. |
| `chosen_avoids_bad`, `chosen_avoids_bad_whp` | `Liveness.lean` | **Coverage-free liveness.** The greedy avoids the *bad* set (flip of D-mass ≥ ε_cov) w.p. ≥ 1 − #cands·exp(−2mγ²). `good`/`bad` are definitional w.r.t. the target ε_cov (D-mass of the flip), so the mean-loss bounds hold *by definition* and follow from a large D-pool — no coverage assumption. Borderline (D-negligible) flips may be chosen, which is fine. |
| `read_disagreement_mean` | `Liveness.lean` | **Derives**, from random classification noise `MQ = ℓ ⊕ r`, `r ∼ Bernoulli(η)` iid, that the per-prefix disagreement `flip ⊕ r` has mean `η + flip·(1−2η)` — pure linearity from `E[r]=η`. A flip shifts expected disagreement by exactly `1−2η = 2s`; baseline `η` is common. No longer assumed. |
| `denoised_loss_eq_flip` | `Liveness.lean` | **Loss ↔ flip-mass.** Mean loss over `m` prefixes `= m·η + (1−2η)·(flip count)` — the sum of `read_disagreement_mean`. Discharges the mean-separability hypotheses (`ρlo = η`, `ρhi = η + (1−2η)·ε_cov`, `γ = (½−η)·ε_cov`). |

The rate is held **per population**, never pooled — `clustering_correct` sums one
term per population and per placement check, exactly the fix #257 makes.

## Where the Lean departs from the code (deliberate; none a soundness gap)

1. **Vote fpr/fnr — conservative bound, not the exact binomial.** A vote is `k`
   iid reads of *one* string, so its count is exactly `Bin(k, β±s)` and the code's
   fpr/fnr *are* exact binomial tails — exactly valid, nothing approximated. The
   Lean lemmas (`wrongDecisive_le`, `misplaced*_le`) instead bound these by the
   sub-Gaussian `exp(-2k·margin²)`. This is a deliberate choice to keep the trusted
   base at Lean core: Mathlib v4.33.1 has no "sum of iid Bernoulli = `Bin(k,p)`"
   (its binomial-RV API is stubbed with `proof_wanted`), so an exact tail would need
   either that missing lemma or an added `HasLaw` assumption. The bound is a valid
   overestimate — the code's real fpr/fnr is `≤` it — so every guarantee here holds
   a fortiori for the code.

2. **Certification — Chernoff threshold, not exact-binomial.** Unlike a vote, a
   certification count spans `n` *different* prefixes of mixed true class, so it is
   a sum of *non-identical* Bernoullis — **not** a binomial. The code tests it
   against a `Bin(n, β+τ)` reference; that test is valid, but justifying the
   non-iid tail's domination by the *exact* binomial needs Hoeffding's 1956 theorem
   (absent from Mathlib). `certErr_bound` uses the Chernoff reference instead (MGF
   domination via AM–GM, in Mathlib) — same `≤ α` guarantee, admits marginally less
   readily. This is a genuine mathematical subtlety, not just a Mathlib-maturity gap
   like (1).

   The two layers are now wired into a single theorem with no abstract
   `trigger`/`himplies`/`p`/`b` premises left. The good side is derived from the
   oracle: the block-local trigger is the reads' separation event `(sepFail)ᶜ`
   (`sepFail_measurable`), it forces a good round (`not_bad_of_not_mem_sepFail`,
   from the selection lemma), and it fires with probability
   `p = 1 − C·exp(-2m((½−η)εcov)²)` (`sepFail_prob`, from `denoised_loss` ∘
   `read_disagreement_mean`). The bad side is `certErr_bound`. `clustering_pac`
   then drives the run-level failure below `δ`. The one soundness input still taken
   as data is the certification read family `Xc` with its drifted mean `≤ β+τ` —
   i.e. the oracle's certification reads of a drifted side (oracle model + target),
   not an abstraction like the eliminated `trigger`/`p`.

4. **Structural / DFA layer — deferred.** This proves the *clustering algorithm*
   correct (families placed and certified). It does **not** yet prove the E-L\*
   *learner* outputs a DFA equal to the target; that needs Myhill–Nerode and the
   transition-resolution construction, the planned next layer.

## Building

```
cd proofs/OrthoDFA
export PATH="$HOME/.elan/bin:$PATH"
lake build
```

Mathlib is prebuilt in `.lake`; the build replays it and checks the five theorems
above in a couple of seconds. The `#print axioms` lines print during the build.
