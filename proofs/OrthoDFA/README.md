# Machine-checked correctness of the E-L\* clustering algorithm

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over a population,
  the returned family decides a prefix wrongly at most `εcov + slack` of the time on the uniform
  pool and at most `1/2 + slack` on any other, and leaves it undecided at most
  `2·indecisionLimit + slack` of the time on each.
- `OrthoDFA/Stage.lean` — the L\* stage as `TransitionResolver` runs it, and `PassDichotomy`: when
  the counterexample pass stops for patience, each of its last `patience` probes was correct, met
  a string the cut cannot place (which the pass harvested), or found a disagreement the split
  evidence would not split on (whose shorter string it harvested).
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, and says what the
counterexample pass has done when it stops. It does not prove that the E-L\* learner outputs the
target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
