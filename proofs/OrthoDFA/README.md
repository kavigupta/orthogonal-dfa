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
  edges do; `walkEnd` is its last leaf when it follows every letter. `HypothesisDFA.split` splits a
  leaf and re-sifts the witnesses of the edges into it (`hypothesis_dfa.HypothesisDFA.split_leaf`).
  `SplitAndLearnCorrect`: splitting a leaf, or learning an edge from a witness, keeps every learned
  edge witnessed (`EdgesOK`).
- `OrthoDFA/Probe.lean` — `probe`: a string walked along the edges from its first `k` letters'
  leaf and sifted, and where the two part, found by `tripartiteSearch` without recovery
  (`counterexample_probe.locate`). `ProbeCorrect`: with the edges witnessed, a wrong edge is one the
  probe crossed with a string whose successor sifts off its target, an unlearned edge is the one it
  stopped at, the at most one undecided string it reports reads undecided and is a prefix of the
  probe followed by a midfix, and a probe on which the walk and the sift part (`Disagrees`) never
  comes out clean.
- `OrthoDFA/RateTest.lean` — `rateSide`, the binomial test of a count of hits against a rate, and
  `RateTestSound`: over independent draws it says above with chance at most its failure chance when
  the true rate is at most the tested one, and below likewise when it is at least.
  `statistics.rate_side` is the same test.
- `OrthoDFA/SeqTest.lean` — `seqRate`, the sequential likelihood-ratio test of a count of hits
  against a rate, and `SeqRateSound` (Ville's inequality): over independent draws, testing at every
  look fires at some look with chance at most `a` when the true rate is on the far side of `p₀`
  from `p₁`. `statistics.seq_rate` is the same test.
- `OrthoDFA/RoundEnd.lean` — what it is for a round of the learner to end well, shared by every
  level: consistent, or a harvest. A harvest is a sampler; more than half of what it finds is at
  bad states, it finds no string more often than the probes share its first `k` letters (spread),
  and it finds `y` in all (yields).
- `OrthoDFA/Level2.lean` — level 2 of indecisive L\*, `indecisiveLStarLevel2`. A disagreement at
  decided reads splits the edge's leaf at once (Kearns–Vazirani: the probe's prefix and the edge's
  witness part at the edge's letter and the midfix where their successors' leaves part), and an
  unlearned edge is learned. The undecided reads probes stop at, and the searches that stopped at
  an undecided midpoint (the disagreements), are counted since the hypothesis last changed. After
  each probe a sequential test (`seqRate`) checks each count against the midpoint of the two
  rates (`alternativeRate`): the round
  harvests once the undecided reads' test of `harvestRate` fires, and ends consistent once the
  disagreements' test of `disagreementRate` fires. Under the `j`-th hypothesis both run at level
  `δ / (2 (j + 1) (j + 2))`, so each test's levels sum to at most `δ / 2`. With
  `harvestRate < disagreementRate`, one of the tests fires within `stretchLength` probes under one
  hypothesis, at most `12 ε / (ε − y)² · log(1/a) + 1` (`stretchBound_mid_le`). A harvest's sampler
  (`harvestLaw`) draws a fresh probe, walks it under the final hypothesis, and finds the undecided
  read it stops at.
- `OrthoDFA/Level2Claim.lean` — `Level2.Correct`: with every state's reads all undecided or all on
  its side (`ReadsByState`), `0 < y < ε < 1` (`harvestRate`, `disagreementRate`), `0 < δ < 1` and
  at least `12 S ε / (ε − y)² · log(2 S (S + 1) / δ) + S` probes,
  `S = |Q|((|Q| + 2)|Σ| + 1) + 2|Σ| + 1` the most hypotheses, level 2 ends well, consistent with
  its hypothesis and tree parting on at most `ε` of the probes or a harvest whose sampler yields
  `y`, but for chance at most `δ`.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Utilities

`OrthoDFA/Utils/` states general lemmas the proofs use, about probability rather than the
algorithm. They are proved under `OrthoDFA/Proofs/` and named in `Verify.lean` like the claims, but
need not be read to trust the claims above.

- `OrthoDFA/Utils/AdaptiveReads.lean` — `AdaptiveReadsBound`: reads drawn once per string,
  independently across strings, each bad with chance at most `ε`, and queried in rounds, each
  round choosing its strings from what earlier rounds read; some query of the first `n` rounds
  reads bad with chance at most `ε` times the most strings they query while none has.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct. It does not prove
that the E-L\* learner outputs the target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
