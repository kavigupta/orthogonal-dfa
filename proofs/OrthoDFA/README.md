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
- `OrthoDFA/ProbeDistribution.lean` — how probes are drawn (`PrefixDistribution`).
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
- `OrthoDFA/Level3.lean` — level 3, `indecisiveLStarLevel3`: level 2's probes with reads drawn at
  random, the search stopping at an undecided midpoint, as at level 2. A clean disagreement
  splits the edge's leaf at once and an unlearned edge is learned. Its counts are kept over the
  stretch since the hypothesis last changed and tested after every probe: the ends (probes whose
  start or end sift stops undecided) and the middles each by `seqRate` against an alternative
  (twice `endsRate`, twice `middleRate`), and success, counting the searches and the ends together,
  by `seqRate` against half `disagreementRate`. Its settings are the yield `y = harvestRate` and
  the disagreement rate `ε = disagreementRate`; the tests' rates follow from them:
  `endsRate = y`, `middleRate = 1/4` and `searchShare = ε/6`. Under the `j`-th hypothesis each test
  runs at level `δ / (8 (j + 1) (j + 2))`. A harvest is the site whose test fired, `ends` or
  `middles`.
- `OrthoDFA/Level3Claim.lean` — `RandomRoundCorrect`: with every string's read drawn once,
  independently, undecided with its state's chance and never on its state's wrong side, every
  length-`walkStart` prefix at most `p₀` of the probes, `48 (L + 1)(|Q| + 1) θ ≤ 1`,
  `2G ≤ y ≤ ε/48`, `ε < 1`, `0 ≤ θ`,
  `0 < δ`, and at least `36 S / ε · log(16 S (S + 1) / δ) + S` probes, level 3 fails to end well,
  consistent with its hypothesis and tree parting on at most `ε` of the probes (`EndsWell` over
  `searchSet ∪ endsSet`, the probes that search or stop undecided at their start or end sift) or a
  harvest whose sampler yields `y`, a state being bad where it is undecided more than `1.5θ` of
  the time, with chance at most `δ`. Here `G = 9θ(|Q| + 1) + 2 p₀ log(8K / δ)`, with
  `K = (1 + (|Q| + 1)³|Σ|)^|Q| (|Q| + 3)^((|Q| + 2)|Σ|)` counting the trees level 3 can grow with an
  edge map into each one's leaves. But for chance `δ/8` each, the probes stopping undecided at
  their start or end at a good state are at most `9θ(|Q| + 1) + p₀ log(8K / δ)`, and those
  stopping at an undecided middle at a good state at most `3θ(L + 1)(|Q| + 1)` of the searches
  plus `p₀ log(8K / δ)`: a search reads only decided sifts, so it is independent of each string
  it might stop at. Only the ends tie `θ` to `ε`, and without `L`. The sampler (`harvestLaw`) draws a fresh probe,
  walks it under the final hypothesis and finds the probe's undecided string at the site, as the
  tests counted it.
- `OrthoDFA/Level4Claim.lean` — `EpsRoundCorrect`: level 3's algorithm, unchanged, with each read
  on its state's wrong side with chance at most `εw`, fails to end well with chance at most `δ`
  under level 3's premises, `G`'s `log(8K/δ)` becoming `log(16K/δ)`, and
  `εw · probeBudget · ((L + 1)|Q| + (|Q| + 1)²|Σ|) ≤ δ/8`. A step that reads no string wrong
  splits and learns only on real targets. A probe's step reads at most `(L + 1)|Q|` strings and a
  split re-sifts at most `(|Q| + 1)²|Σ|`, so by `AdaptiveReadsBound` the premise keeps every string
  the round reads right but for chance `δ/8`. The good states' bounds take `δ/16` each and the
  stretches `3δ/4`.
- `OrthoDFA/Level5Claim.lean` — `KappaRoundCorrect`: level 3's algorithm, unchanged, with each read
  on its state's wrong side with chance at most `εw + κ U q` (`KappaReadOracle`), fails to end well
  with chance at most `δ` under level 4's premises with the last replaced by
  `εw (N(L + 1)|Q| + |Q|(|Q| + 1)²|Σ|) + κ (5S log(8S(S + 1)/δ) + 1 + |Q|(|Q| + 1)²|Σ|) ≤ δ/8`,
  `N` the budget. A string is read wrong at most `κ` times as often as undecided, beyond `εw`; a
  probe's sifts stop at their first undecided read, and the tests count it: under the `j`-th
  hypothesis, until a test fires, the probes stopping undecided number at most `5 log(1/a_j)`,
  whatever the stretch's length. So by `RaceReadsBound` over the strings each step reads, the round reads some string
  wrong with chance at most `δ/8`.
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

- `OrthoDFA/Utils/RaceReads.lean` — `RaceReadsBound`: reads drawn once per string, independently
  across strings, each bad with chance at most `ε` plus `κ` times its chance of reading in `V`, and
  queried in rounds, each fixed by what it and the earlier rounds read; some query of the first `n`
  rounds reads bad with chance at most `ε b + κ m`, where the queries, before any bad read, number at
  most `b`, `m` of them reading in `V`.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct. It does not prove
that the E-L\* learner outputs the target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
