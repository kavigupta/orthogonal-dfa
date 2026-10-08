# Machine-checked correctness of the E-L\* clustering algorithm

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over a population,
  the returned family decides a prefix wrongly at most `εcov + slack` of the time on the uniform
  pool and at most `1/2 + slack` on any other, and leaves it undecided at most
  `2·indecisionLimit + slack` of the time on each.
- `OrthoDFA/Stage.lean` — `RoundOutcome`: for any tree and DFA a round ends with, if the DFA/DT
  agreement check fails on a share `d` of sampler strings, the harvest sampler finds at least one
  string on at least a `d/L` share of attempts, harvests no single string `t` on more than
  `∑_{i ≤ |t|} min(i+1, L)/L · D(the draw starts as t does)` of them, and reads no single string
  on more than that.
- `OrthoDFA/Round.lean` — `RoundProgress`: through a suffix-free vote family whose node reads of
  states read undecided less than `uHi` flip with chance at most `φ`, a round ends, but for chance
  `(L+1)(N+1)·φ/ε + passReadBound·φ`, with:
  1. the DFA/DT check failing on at most `ε` of sampler strings;
  2. a population the next gate must act on;
  3. the pass halving the indecision limit;
  4. its bisection population forcing the next gate;
  5. a state its family reads badly;
  6. a visited string's extension read off the hypothesis's edge more often than not; or
  7. the pass running out of probes before a patience streak.

  `Pass.lean` states the pass, with its bisection population, and `Automaton.lean` the target.
- `OrthoDFA/StartAtK.lean` — the round walked from position `k` along learned edges only.
  `RoundAtK`: with the round's reads fixed, whatever probes the pass draws, the round's decision
  is right but for `exp(−2·nw·δ²) + 2·exp(−2·ng·δ²)` over its two batches. That decision is one of:
  - a walk source, where the walk is blocked on at least `θw − δ` of draws;
  - a boundary source, where the check is blocked on at least `θc − δ`;
  - a pass, where the check disagrees on at most `ε + δ`;
  - a refusal, where every disagreeing probe, rerun, splits a leaf, adds a member, or stops at a
    string the cut cannot place.

  `WalkYield`: the walk source yields its blocked share less the draws that reached an unlearned
  edge through a wrong earlier one. `SourceSpread`: no string takes more of a source than the
  draws sharing its first `k` letters.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, that a round
either passes the DFA/DT agreement gate or leaves a harvest that yields and is spread, and that a
round ends in one of the seven outcomes above. It does not prove that the E-L\* learner outputs the
target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
