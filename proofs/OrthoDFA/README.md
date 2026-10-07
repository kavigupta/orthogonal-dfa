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
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, and that a round
either passes the DFA/DT agreement gate or leaves a harvest that yields and is spread. It does not
prove that the E-L\* learner outputs the target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
