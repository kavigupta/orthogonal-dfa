# Machine-checked correctness of the E-L\* clustering algorithm

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — the target as a DFA, the quality of a suffix family, and
  `ClusteringQualityGuarantee`: the returned family is as good as its populations force.
- `OrthoDFA/Verify.lean` — names the proofs of both claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct. It does not prove
that the E-L\* learner outputs the target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
