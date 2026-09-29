# Machine-checked correctness of the E-L\* clustering algorithm

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — the target as a DFA, the quality of a suffix family, and
  `ClusteringQualityGuarantee`: the returned family is as good as its populations force.
- `OrthoDFA/ReturnAccuracy.lean` — `ReturnAccuracy`: whatever the L\* stage does, the learner
  rarely returns a hypothesis the merge check passed that is inaccurate once denoised.
- `OrthoDFA/Check.lean` — the merge check, and `CheckGuarantee`: it rarely fails a state whose
  label is pure, and rarely passes one with a large minority.
- `OrthoDFA/Termination.lean` — `Termination`: if each round's clustering, stage and check meet
  their specs, at most `2·|Q|` rounds fail the check.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, that the learner
returns only accurate hypotheses, and that it stops once the L\* stage meets its spec. It does not
prove that the L\* stage meets it.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
