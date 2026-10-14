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
- `OrthoDFA/DTree.lean` — the discrimination tree, and `DTree.siftWithPath`, sifting a string
  through it under a three-valued cut. `SiftWithPathCorrect`: the reads are the string followed by
  each midfix on its way, no more than the tree is deep; a decided sift ends at a leaf, along the
  path its reads decide, and an undecided one at its only undecided read.
  `MidfixTree.sift_with_path` is the same function. It also holds `sift`, the same sift under a
  read, and `splitAt`, which splits a leaf. `SiftAndSplitCorrect`: `sift` lands where
  `siftWithPath` does, strings at different leaves read decidedly apart where their paths part,
  and a split replaces its leaf by two children, reads its midfix there, and changes only the sifts
  that landed on it. `MidfixTree.split` is the same split, naming the leaf by its state.
- `OrthoDFA/HypothesisDFA.lean` — a hypothesis DFA: a tree and learned edges between its leaves,
  and the walk a string takes along them. `WalkCorrect`: the walk starts at its leaf, follows one
  learned edge per letter, stops short only at an unlearned one, and stays on the leaves when the
  edges do; `walkEnd` is its last leaf when it follows every letter. `Disagrees` is where the walk
  and the tree part.
- `OrthoDFA/RateTest.lean` — `rateSide`, the binomial test of a count of hits against a rate, and
  `RateTestSound`: over independent draws it says above with chance at most its failure chance when
  the true rate is at most the tested one, and below likewise when it is at least.
  `statistics.rate_side` is the same test.
- `OrthoDFA/SeqTest.lean` — `seqRate`, a sequential likelihood-ratio test of a count of hits:
  against a rate `p₀` it fires once `(p₁/p₀)^h ((1 − p₁)/(1 − p₀))^(n − h) ≥ 1/a`, `h` hits in `n`
  draws. `SeqRateSound`: over independent draws it fires at any look with chance at most `a` when
  the true rate is on the far side of `p₀` from `p₁` (Ville's inequality). `stretchBound`: where
  `p₀ < p₁ < p₀'`, one of the tests of `p₀` and of `p₀'` against `p₁` fires within that many
  draws, whatever the count.
- `OrthoDFA/ProbeDistribution.lean` — how probes are drawn (`PrefixDistribution`).
- `OrthoDFA/RoundEnd.lean` — what it is for a round of the learner to end well, shared by every
  level: consistent, or a harvest. A harvest is a sampler; more than half of what it finds is at
  bad states, it finds no string more often than the probes share its first `k` letters (spread),
  and it finds `y` in all (yields).
- `OrthoDFA/Level2Probe.lean` and `OrthoDFA/Level2.lean` — level 2 of indecisive L\*,
  `indecisiveLStarLevel2`. Its probe walks to where the walk stops, sifts the successor of an
  unlearned edge it stops at, and finds where the sift and the walk part with `tripartiteSearch`
  with no recovery, stopping at the first undecided read; a probe shorter than `walkStart` counts
  as agreeing. A disagreement at decided reads splits the edge's leaf at once (Kearns–Vazirani:
  the probe's prefix and the edge's witness part at the edge's letter and the midfix where their
  successors' leaves part), and an unlearned edge is learned. The undecided reads probes stop at,
  and the disagreements, since the hypothesis last changed are counted, and after each probe the
  sequential test (`seqRate`) at level `testLevel` checks them, each against `alternativeRate`:
  the round harvests once the undecided reads' test of `harvestRate` fires, and ends consistent
  once the disagreements' test of `disagreementRate` fires. A decided disagreement changes the
  hypothesis, so the disagreements counted are the searches that stopped at an undecided midpoint,
  as at level 3. With `harvestRate < alternativeRate < disagreementRate`, one of the tests fires
  by `n = stretchLength` probes under one hypothesis, at most `stretchBound`. A harvest's sampler
  (`harvestLaw`) draws a fresh probe, walks it under the final hypothesis, and finds the undecided
  read it stops at.
- `OrthoDFA/Level2Claim.lean` — `IdealRoundCorrect`: with every state's reads all undecided or
  all on its side, `0 < harvestRate < alternativeRate < disagreementRate < 1`, `0 < a < 1` and
  `stretches · n ≤ P`, level 2 ends well, consistent with its hypothesis and tree parting on at
  most `disagreementRate` of the probes or a harvest whose sampler yields `harvestRate`, but for
  chance at most `2 S a`, `S` the stretches and `a = testLevel`: each sequential test errs at most
  `a` a stretch.
- `OrthoDFA/Level3.lean` — level 3, `indecisiveLStarLevel3`: level 2's probes with reads drawn at
  random, the search stopping at an undecided midpoint, as at level 2. A clean disagreement
  splits the edge's leaf at once and an unlearned edge is learned. Its counts are kept over the
  stretch since the hypothesis last changed and tested after every probe, the start, the middles
  and success each by `seqRate` against an alternative (`startAlternative`, `middleAlternative`,
  `disagreementAlternative`), an edge by a threshold. A probe's undecided strings at an edge are
  counted once. A harvest is the site whose test fired.
- `OrthoDFA/Level3Claim.lean` — `RandomRoundCorrect`: with every string's read drawn once,
  independently, undecided with its state's chance and never on its state's wrong side, level 3
  on `C.probeBudget` probes fails to end well, a harvest's sampler yielding `y`, with chance at
  most `roundRisk`. The sampler (`harvestLaw`) draws a fresh probe, walks it under the final
  hypothesis and finds the probe's undecided strings at the site, as the tests counted them. A
  stretch pays each sequential test's level once, and each edge's threshold once.
- `OrthoDFA/Level4Claim.lean` — `EpsRoundCorrect`: level 3's algorithm, unchanged, with each read
  on its state's wrong side with chance at most `εw`, fails to end well with chance at most
  `epsRoundRisk`: `roundRisk`'s terms, plus `wrongRisk`, the chance that the probes that can read
  wrong a prefix followed by a string of length at most `|Q|` exceed `Gw` of the probes, plus
  `probeBudget · Gw`, the chance that some probe of the round is one. A probe that cannot splits
  and learns only on real targets, so the tree stays within `|Q| + 1` leaves.
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
