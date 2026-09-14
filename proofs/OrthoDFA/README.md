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

Model (`Hoeffding.lean`): the oracle gives, per query string, an independent read;
a member of `L` reads 1 with mean `β+s`, a non-member with `β-s`; reads lie in
`[0,1]`. A vote of `k` suffixes averages `k` such reads.

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

3. **Findability fusion — constructed (`fuse_findability`, `Capstone.lean`).**
   In `clustering_algorithm_correct_fused` both per-round inputs are discharged:
   `hbad` from `certErr_bound`, and `hgood` from `fuse_findability`, which gives a
   good pass probability `≥ (1-reject)·pp`. The proposal↔reads coupling is handled
   by the honest model of the persistent oracle — every read is pre-drawn, so
   reads are independent of the proposal and the round's space is a genuine product
   `μ_prop × μ_reads`; the fusion is then Fubini, not a bespoke kernel. The
   remaining inputs are the oracle model, the findability rate `pp`, and the
   gate-accept factor `1-reject` (itself `cleanAdmit_le`/`apLowFNR_le`, supplied as
   `haccept`).

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
