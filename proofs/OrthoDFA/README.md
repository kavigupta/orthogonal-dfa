# Machine-checked correctness of the E-L\* learner

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over each
  population, the returned family decides a prefix wrongly at most `εcov + slack` of the time
  and leaves it undecided at most `2·indecisionLimit + slack` of the time.
- `OrthoDFA/ReturnAccuracy.lean` — `ReturnAccuracy`: whatever the L\* stage does, the learner
  rarely returns a hypothesis the merge check passed that is inaccurate once denoised.
- `OrthoDFA/Check.lean` — the merge check, and `CheckGuarantee`: it rarely fails a state whose
  label is pure, and rarely passes one with a large minority.
- `OrthoDFA/Stage.lean` — the L\* stage as `TransitionResolver` runs it.
- `OrthoDFA/Termination.lean` — `Termination`: if each round's clustering, stage and check meet
  their specs, few rounds fail to return.
- `OrthoDFA/Learner.lean` — the learner, and `LearnerCorrect`.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

`LearnerCorrect` is the end-to-end claim; the others are its parts. It holds for any L\* stage
that reads the noise at few strings and labels what the family cuts well (`mislabelledWellCut`).
Nothing here proves that the L\* stage meets that.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
