# Machine-checked correctness of the E-L\* clustering algorithm

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over a population,
  the returned family decides a prefix wrongly at most `εcov + slack` of the time on the uniform
  pool and at most `1/2 + slack` on any other, and leaves it undecided at most
  `2·indecisionLimit + slack` of the time on each.
- `OrthoDFA/FamilyRead.lean` — the suffix family's read of a string: accept at `X(w) ≥ kh`,
  reject at `X(w) ≤ kl`, undecided between, where `X(w)` counts the members `v` whose query
  `w·v` answers 1.  `FamilyReadBound`: with the language a DFA's, the family suffix-free
  and the band passing `BandPasses`, the reads are independent across strings and every DFA
  state has one distribution its strings read with, which is accept at most `ε` of the time, or
  reject at most `ε`, or undecided at least a third with accept or reject at most `ε₂`; and is on
  its rarer decided side at most `ε`, or at most `κ` times as often as it is undecided.  `BandPasses` is the
  check the band selection rule `evidence_margin_for_population_size` makes, at
  `ε = cross_limit`, `ε₂ = MINORITY_READ_LIMIT` and `κ = MINORITY_UNDECIDED_RATIO`.
- `OrthoDFA/TripartiteSearch.lean` — `tripartiteSearch`, a bisection for where `agrees` turns from
  agreeing to disagreeing that steps past undecided positions, and `TripartiteSearchCorrect`: each
  result says what `agrees` is where it points, the queries stay between the endpoints and number
  at most `2 + 3⌈log₂(hi − lo)⌉`, and with nothing undecided it finds the turn exactly.
  `orthogonal_dfa/l_star/tripartite_search.py` is the same function.
- `OrthoDFA/RateTest.lean` — `rateSide`, the binomial test of a count of hits against a rate, and
  `RateTestSound`: over independent draws it says above with chance at most its failure chance when
  the true rate is at most the tested one, and below likewise when it is at least.
  `statistics.rate_side` is the same test.
- `OrthoDFA/ProbeDistribution.lean` — how probes are drawn (`PrefixDistribution`).
- `OrthoDFA/RoundEnd.lean` — what it is for a round of the learner to end well, shared by every
  level: consistent, or a harvest. A harvest is a sampler; more than half of what it finds is at
  bad states, it finds no string more often than the probes share its first `k` letters (spread),
  and it finds `y` in all (yields).
- `OrthoDFA/Hypothesis.lean` — the learner's hypothesis: a tree sifting strings to its leaves, and
  edges between the leaves.
- `OrthoDFA/Level2Probe.lean` and `OrthoDFA/Level2.lean` — level 2 of indecisive L\*,
  `indecisiveLStarLevel2`. Its probe walks to where the walk stops, sifts the successor of an
  unlearned edge it stops at, and finds where the sift and the walk part with `tripartiteSearch`;
  a probe shorter than `walkStart` counts as agreeing. A harvest is the site whose undecided
  strings reached `harvestSize`; its sampler (`harvestLaw`) draws a fresh probe, walks it under the
  final hypothesis, and finds the probe's undecided strings at the site.
- `OrthoDFA/Level2Claim.lean` — `IdealRoundCorrect`: with every state's reads all undecided or
  all on its side, level 2 on `P` probes ends well, a harvest's sampler yielding `y`, but for
  chance at most `(P + 1)(1 − ε)^n + P (2 + (|Q| + 2)|Σ|) y`, `n` the clean probes in a row that
  end it.
- `OrthoDFA/Level3.lean` — level 3, `indecisiveLStarLevel3`: level 2's probes with reads drawn at
  random, its counts kept over the stretch since the hypothesis last changed and tested after
  every probe. A probe's undecided strings at an edge are counted once. A harvest is the site whose
  test fired.
- `OrthoDFA/Level3Claim.lean` — `RandomRoundCorrect`: with every string's read drawn once,
  independently, undecided with its state's chance and never on its state's wrong side, level 3
  on `C.probeBudget` probes fails to end well, a harvest's sampler yielding `y`, with chance at
  most `roundRisk`. The sampler (`harvestLaw`) draws a fresh probe, walks it under the final
  hypothesis and finds the probe's undecided strings at the site, as the tests counted them.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
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
