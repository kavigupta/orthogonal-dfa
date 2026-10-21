# Machine-checked correctness of the E-L\* clustering algorithm

## The theorems

Every theorem below is proved in Lean with no `sorry`, using only `propext`, `Classical.choice`
and `Quot.sound` (`OrthoDFA/Verify.lean` prints this).

The headline is `LearnerLoopDFA` (`OrthoDFA/LoopDFA.lean`, proved in `Proofs/LoopDFA.lean`): the
loop outputs a DFA, and `LearnerLoopDFASimple` states the same at explicit settings (below). The
whole run reads one noise, fixed per string as the Python's is. Each round runs level 3's algorithm
(`indecisiveLStarLevel3`), guaranteed by level 5's claim (`Level5RoundCorrect`), reading the
clustering's family. One-step consistency at `L − 1` is not enough for this (`parity_toggle`,
#438).

- `ClusteringGuarantee`: with probability at least `1 − δ` the clustering loop stops, and the
  family it returns cuts all but `εcov` of the uniform pool as the noiseless oracle does, at a
  polynomial number of queries. The samplers draw without replacement, and a draw lands on what a
  state already holds at most `(|J| + 3)·npref` times the largest atom, so each sampler rejects
  more than `rejCount` of its draws only on a binomial tail (`SuccessiveDrawsTails`); a tame
  population need only keep every atom under `rateScale·(1/2 − η₀)/(2¹⁸(|J| + 3)·npref)`, and under
  `δ·min(εcov, indecisionLimit)²/(2³³|J|(3 + 2⌊log₂ npref⌋))` for the hash to split it evenly.
- `ClusteringQualityGuarantee`: averaged over a population, the returned family decides a prefix
  wrongly, or leaves it undecided, only a bounded share of the time.
- `FamilyReadBound`: the suffix family's reads are independent across strings, every DFA state
  has one distribution its strings read with, and the band's check bounds each.
- `TripartiteSearchCorrect`: the search the probes use to find where the walk and the sift part
  finds what it reports, queries at most `2 + 3⌈log₂(hi − lo)⌉` positions (`2 + ⌈log₂(hi − lo)⌉`
  without recovery), and finds the turn exactly where nothing is undecided.
- `Level2RoundCorrect`, `Level3RoundCorrect`, `Level4RoundCorrect` and `Level5RoundCorrect`: one L\* round
  ends well but for chance `δ`, with ideal reads, random indecision, rare wrong reads, and wrong
  reads scaling with indecision.
- `FamilyKappaOracle` (`OrthoDFA/FamilyOracle.lean`): the clustering's family reads are a
  `KappaReadOracle`, each state's side the one it reads more often, `εw` and `κ` the band's.
- `LearnerLoopDFA` and `LearnerLoopDFASimple`: the loop outputs a DFA. See below.

## The learner's DFA

`LearnerLoopDFA` asks for what the Python's `satisfies_preconditions` checks of the target, at its
length `L`: `coveredStates M L c` are the states at least `c` (`min_coverage`) of the strings of
length `L` visit at some position from `⌈L/2⌉` to `L`, and `rerootAccuracy M L S q` is the share
of those strings whose run from `q` stays in `S` and is classified as the target classifies them
(`covered_accuracy_ceiling` at the start `q`). Where some covered `q` has accuracy at least
`1 − γ`, the loop, its probes uniform over the lengths `k = ⌈L/2⌉` to `L₁ = L + |Q| + 3` and
walked from `k`, stops within `|Q| + 1` rounds at a consistent round, and its re-rooted DFA
(`rerootedDFA`: the final hypothesis started at the leaf that scores best on `n` held-out strings)
misclassifies at most `γ + ξ` of the strings of length `L`, but for chance `δ`, once `L ≥ L₀`. A run
that reaches an unlearned edge counts as wrong. The start leaf (`pickLeaf`) is picked as follows: after
the loop, `n = ⌈32 log(16(|Q|+1)(|Q|+2)/δ)/ξ²⌉` strings uniform of length `L` are drawn and read
by the last round's family; each leaf scores the strings its run classifies as the family decides
them, undecided reads scoring none, and the first leaf of highest score is the start. With
`N = L₁ + 1 − k`, each round runs level 3 at disagreement rate `ε = c/(4(|Σ|+1)N)` and yield
`ε/48`, drawing at most `B₃ = 36 S/ε · log(16 S(S+1)/δ') + S + 1` probes,
`S = |Q|((|Q|+2)|Σ|+1) + 2|Σ| + 1`. `LearnerLoopDFASimple` states the same at explicit settings,
every hypothesis an inequality; they follow from `(L₁ + 1)⁶ (E + E₂) ≤ |Σ|^k`, `E` and `E₂` free
of `L`, logarithmic in `1/δ`, `1/c`, `1/ξ`, `1/(1/2 − η₀)`, `1/pAP`, `|Q|` and `|Σ|`. Bounding
`log (L₁ + 1)` by its tangent at `T = 2(log (E + E₂) + 12)/log |Σ| + |Q| + 4`, that holds once
`L ≥ L₀` (`dfa_pow`, `Proofs/LoopDFAArith.lean`), where

    L₀ = 2(log (E + E₂) + 6 log T)/log |Σ| + 144(log T − 2)/(log |Σ| (T log |Σ| − 12)).

At `|Q| = 3`, `|Σ| = 2`, `η₀ = 0.1`, `c = δ = 0.01`, `pAP = 1/3`, `ξ = 0.1`, `L₀ ≈ 418.8`; the
least `L` with `(L₁ + 1)⁶ (E + E₂) ≤ |Σ|^k` is 419, and the least meeting `LearnerLoopDFASimple`'s
hypotheses 369.

The probes over a window of lengths are the algorithm's change. Every prefix length is itself a
probe, so a probe and its extension by a letter that both agree, read nothing on a rarer side
past a midfix and nothing undecided at their sifts, cross a learned edge between their states'
true leaves (`edge_of_pair`, `Proofs/DFABridge.lean`). A covered state is visited `c` of the time
summed over the window, so an edge out of its true leaf that is wrong or unlearned leaves
`c/((|Σ|+1)N)` of the probes disagreeing (`edge_of_occupied`), `disagreeSet` being the probes
that search, cross a wrong edge, stop at an unlearned one, stop undecided at their start or end
sift (level 3's own success test bounds these at `ε`), or read a string followed by a midfix of
the tree on its state's rarer side (`rarerSet`). The last is bounded by Markov: each string reads
wrong with chance at most the band's far side, and on the event that no step reads wrong the tree
is in `classSet |Q|`, its midfixes at most `|Q|` long, so the probes reading wrong past a midfix
weigh more than `ε` with chance at most `(|Q|+1)|Σ|^|Q|` times the far side over `ε`, which the far
side's choice makes `δ'/8`. The clustering, run at `εcov = min(ε, ξ/(8N))`, reads every covered
state on its true side (`sideOf_eq_of_mass`). Every covered state's edges are then true and it
accepts as the target does, so the hypothesis from the covered start's true leaf tracks the
re-rooted target on every run that stays covered (`classify_start_le`).

The pick compares each leaf against the family's read, not the oracle's: the oracle's noise rates
on and off the language may differ, and agreement with such a read weighs a leaf's errors by
`1 − 2η` of the string's true label, which can rank a worse leaf first. The family's read of a
string of length `L` is wrong at most `(1025/1024) N εcov` of the time and undecided at most
`(3/8) N εcov`, with `N εcov ≤ ξ/8`, so a leaf's
mean score is within that of its accuracy. Reads persist per string, so the held-out strings'
reads are fresh only where no held-out string repeats or begins with the first `k` letters of a
prefix the last round's clustering read or of one of its probes: level 3 reads only short strings
beginning with some probe's first `k` letters (`level3_congr_zone'`, `Proofs/LoopSelect.lean`), and
the earlier rounds read other suffixes. That collision has chance at most
`n(6(|Q|+2)RB + B₃ + n)/|Σ|^k`, which `L ≥ L₀` makes at most `δ/(4(|Q|+1))` for each round that
can be last. Off it, given the loop, each leaf's score is within `nξ/8` of its mean given the
strings, and that within `nξ/8` of `n` times its mean over the strings, but for chance
`4(|Q|+2)e^(−nξ²/32)` (`scoreBad_le`), the tree having at most `|Q| + 2` leaves, so the picked leaf
misjudges at most `γ + ξ/2 + (3/8 + 1025/512)Nεcov ≤ γ + ξ` (`pickLeaf_le`). `δ` splits half to
the loop, a quarter to the scores and a quarter to collisions.

## The loop

`indecisiveLStarLoop` (`OrthoDFA/LearnerLoop.lean`) clusters a family over the populations so far,
runs level 3 with it, and adds a harvest's sampler, scaled to a probability, as a population of
every later round. The whole run, every round's clustering and level 3, reads one noise. Each
round's clustering draws its prefixes, certification prefixes and suffixes off every one the
earlier rounds' clusterings read (`offLaw`), so no two rounds read the same string. Level 3 runs at
disagreement rate `ε` and draws its probes from `D` conditioned on their first `k ≥ 1` letters
avoiding those of every prefix the round's clustering read. The band's far side is
`δ'/(8W)`, `W = B₃ w|Q| + |Q|(|Q|+1)²|Σ| + (|Q|+1)|Σ|^|Q|/ε` with `w = min(L₁+1, 3 + ⌈log₂ L₁⌉)`,
which meets level 5's premise `εw(B₃ w|Q| + |Q|(|Q|+1)²|Σ|) ≤ δ'/8` (`κ = 0` for the family) and
the Markov bound above. A
round's claim (`RoundSpec`, `Proofs/LoopTermination.lean`) is level 3's ending, transferred to `D`,
and its tree in `classSet |Q|`; it fails with chance at most `δ' + δ'/8 + δ'/8`: level 5's claim,
the event that some step reads a string wrong (`pi_wrong_le'`), and the Markov bound
(`level3Fail_le`, `Proofs/LoopRenoise.lean`). A harvest's sampler yields at least `ε/96` and finds a
string at most as often as the probes share its length-`k` prefix, so a population puts at most
`96 prefixMax / ε` on a string, within `cap`, which also bounds the suffixes' atoms; its strings are
at most `L₁ + |Q|` long, the tree's midfixes at most `|Q|`. Against `cap` the clustering's atom
conditions hold, every population tame: its samplers' rejections, `NP·cap` under the rate scale
times the signal; the hash's balance and the quality's deviation, `log₂ NP·cap` under `δ` times
the rates squared; and the tables' own mass, `NP·cap` under the slack. The earlier rounds'
reads, at most `6(K+1)NP` prefixes and `NS + 1` suffixes a round, hold at most `m` of any
population and `ms` of the suffixes. `NS` and `NP` are bounded by
`poolCount_le` and `prefCount_le_poly` (`counts_le`, `Proofs/LoopSimple.lean`).

The rounds' draws are independent, and given the earlier rounds a round's clustering draws, the
hash's sides included, are drawn as `runMeasure` draws them at their populations conditioned off
what they read
(`Proofs/LoopLaw.lean`, `Proofs/LoopIndep.lean`): a population's draws are made from uniforms
(`drawFrom`), and a round's population and reads are fixed by finite data, its last hypothesis's
tree and edge graph, its reads of short strings and its clustering's prefixes and suffixes
(`Proofs/LoopCodes.lean`). After `T` probes level 3's tree has at most `T + 2` leaves and midfixes at
most `T` long, whatever it reads, and its run reads only a probe's prefix past its first `k` letters
followed by at most `T` letters (`runProbes_congr_zone`, `Proofs/LoopRuns.lean`). So a round's
outcomes are measurable: at fixed draws each is fixed by the oracle's answers at finitely many
strings (`Proofs/LoopMeasurable.lean`).

On one noise, what the earlier rounds hand on is fixed by the answers at the strings they read, and
a round's bad event by the answers at the strings it reads; when no two rounds read the same prefix
or suffix the two sets are disjoint, so the two are independent (`persist_stage_le`,
`Proofs/LoopPersist.lean`, `Proofs/LoopCollide.lean`). A uniform below `1` draws a string of
positive mass off the earlier rounds' reads (`drawFrom_offLaw`, `Proofs/LoopOff.lean`), so with
chance one every suffix read is `Ls` long, every prefix shorter, and no round reads a prefix or a
suffix an earlier one read (`loop_coll_null`, `Proofs/LoopColl.lean`). The clustering's guarantee
holds of the conditioned laws, whose atoms are at most twice the unconditioned and whose `pAP`
drops by at most what was removed, and a population's average under the conditioned law is within
the removed mass `m` of its own (`integral_le_offLaw`), which the claim's slack pays
(`loop_cluster_round_le_persist`, `Proofs/LoopE1.lean`). Level 3 reads `z·v` with `v` a family
suffix and `z`'s first `k` letters a probe's, so its avoiding probes never read a string the
clustering read. Re-noising the clustering's strings with a second, independent noise
(`Oracle.renoise`, `Proofs/Renoise.lean`) leaves a fresh family on which level 5's guarantee holds
at the conditioned probes, whose prefix mass grows by at most `1/(1 − m)` for `m ≤ |keys| ·
prefixMax` the mass removed (`Proofs/LoopAvoid.lean`); the actual run is the same run, and its
ending transfers back to `D` at disagreement `ε + m`, yield `(1 − m) y`, and at most `m` more of
the sampler at any set of states (`looseEndsWell_of_endsWell`, `Proofs/LoopTransfer.lean`).

The loop stops because each harvest finds a new state the later families rarely leave undecided
(`loop_agrees_of_specs`, `Proofs/LoopTermination.lean`). Level 5's harvest puts at most
`A·T + sl` of its law on the states read undecided at most `T` of the time,
`A = (|Q|+1)(6/y + 16(2 + ⌈log₂ L₁⌉))` and `sl` the concentration slack, plus `m` after the
transfer. Every later family is held by the gate to `c = 2ι + slack` undecided on the harvest's
population, so a state holding more than `c/T` of it is read undecided at most `T` of the time by
every later family (`good_of_heavy`). At `T = 7/(16A)`, with `ι ≤ ε/(4608|Q|(|Q|+1))`, the slack and
`m/y` at most `1/16` each, and `(2 + ⌈log₂ L₁⌉)ε < 3/8`, more than `|Q|c/T` of the population lies
on the states read undecided more than `T` of the time, so more than `c/T` on one of them. That
state is never such again, so one of the first `|Q| + 1` rounds does not harvest.

## The round, by refinement

Each level proves one L\* round correct and states its ending through `OrthoDFA/RoundEnd.lean`:
the round ends well when it is consistent, the hypothesis and the tree parting on at most `ε` of
the probes, or ends in a harvest whose sampler is spread, yields, and puts at most
`slope·T + slack` of its law on the states read undecided at most `T` of the time. Each level says
what a state's chance of reading undecided is.

1. #417, `Level2RoundCorrect`: ideal reads; a state reads undecided where all its strings do.
2. #419, `Level3RoundCorrect`: indecision random, decided reads always right.
3. #420, `Level4RoundCorrect`: rare wrong reads at every state.
4. #450, `Level5RoundCorrect`: wrong reads at most `εw` plus `κ` times the state's indecision, the
   round the loop runs.

## Open

- **The shipped Python uses fixed settings**, not the ones the claims choose.
- **The rarer reads past a midfix** are bounded by a union over every word of at most `|Q|` letters,
  `(|Q|+1)|Σ|^|Q|` of them, which enters only the band's far side, through its log.
- Each one-round theorem is per round, with the family, the seed and the band fixed, and treats the
  noise as fresh; the loop's theorem reads one noise across rounds (above).
- **The model's loop differs from the Python's.** The round draws its probes avoiding the first `k`
  letters of the prefixes the round's clustering read, where the Python draws them from `D`. Each
  round's clustering draws its prefixes, certification prefixes and suffixes off every one the
  earlier rounds' clusterings read, where the Python does not avoid them. The uniform pool is drawn
  afresh each round, where the Python keeps one pool across rounds. The hash's sides are fair coins
  drawn afresh each round, where the Python draws its key once per run, and a harvest's table is
  drawn from its law, where the Python first puts the harvested strings in it.

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim. It
  imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over a population, the
  returned family decides a prefix wrongly at most `εcov + slack` of the time on the uniform pool
  and at most `1/2 + slack` on any other, and leaves it undecided at most `2·indecisionLimit +
  slack` of the time on each.
- `OrthoDFA/FamilyRead.lean` — the suffix family's read of a string: accept at `X(w) ≥ kh`, reject
  at `X(w) ≤ kl`, undecided between, where `X(w)` counts the members `v` whose query `w·v` answers
  1.  `FamilyReadBound`: with the language a DFA's, the family suffix-free and the band passing
  `BandPasses`, the reads are independent across strings and every DFA state has one distribution
  its strings read with, which is accept at most `ε` of the time, or reject at most `ε`, or
  undecided at least a third with accept or reject at most `ε₂`; and is on its rarer decided side at
  most `ε`, or at most `κ` times as often as it is undecided. `BandPasses` is the check the band
  selection rule `evidence_margin_for_population_size` makes, at `ε = cross_limit`, `ε₂ =
  MINORITY_READ_LIMIT` and `κ = MINORITY_UNDECIDED_RATIO`.
- `OrthoDFA/TripartiteSearch.lean` — `tripartiteSearch`, a bisection for where `agrees` turns from
  agreeing to disagreeing that steps past undecided positions.
  `orthogonal_dfa/l_star/tripartite_search.py` is the same function.
- `OrthoDFA/DTree.lean` — the discrimination tree, and `DTree.siftWithPath`, sifting a string
  through it under a three-valued cut. `MidfixTree.sift_with_path` is the same function. It also
  holds `sift`, the same sift under a read, and `splitAt`, which splits a leaf. `MidfixTree.split`
  is the same split, naming the leaf by its state.
- `OrthoDFA/HypothesisDFA.lean` — a hypothesis DFA: a tree and learned edges between its leaves,
  and the walk a string takes along them; `walkEnd` is its last leaf when it follows every letter.
  `HypothesisDFA.split` splits a leaf and re-sifts the witnesses of the edges into it
  (`hypothesis_dfa.HypothesisDFA.split_leaf`).
- `OrthoDFA/Probe.lean` — `probe`: a string walked along the edges from its first `k` letters'
  leaf and sifted, and where the two part, found by `tripartiteSearch` without recovery
  (`counterexample_probe.locate`). `ProbeCorrect`: with the edges witnessed, a wrong edge is one the
  probe crossed with a string whose successor sifts off its target, an unlearned edge is the one it
  stopped at, the at most one undecided string it reports reads undecided and is a prefix of the
  probe followed by a midfix, and a probe on which the walk and the sift part (`Disagrees`) never
  comes out clean.
- `OrthoDFA/RateTest.lean` — `rateSide`, the binomial test of a count of hits against a rate.
  `statistics.rate_side` is the same test.
- `OrthoDFA/SeqTest.lean` — `seqRate`, the sequential likelihood-ratio test of a count of hits
  against a rate. `statistics.seq_rate` is the same test.
- `OrthoDFA/ProbeDistribution.lean` — how probes are drawn (`PrefixDistribution`).
- `OrthoDFA/RoundEnd.lean` — what it is for a round of the learner to end well, shared by every
  level: consistent, or a harvest. A harvest is a sampler; it puts at most `slope·T + slack` of
  what it finds on the states read undecided at most `T` of the time, it finds no string more
  often than the probes share its first `k` letters (spread), and it finds `y` in all (yields).
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
- `OrthoDFA/Level2Claim.lean` — `Level2RoundCorrect`: with every state's reads all undecided or all on
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
- `OrthoDFA/Level3Claim.lean` — `Level3RoundCorrect`: with every string's read drawn once,
  independently, undecided with its state's chance and never on its state's wrong side, every
  length-`walkStart` prefix at most `p₀` of the probes, `0 < y ≤ ε/48`, `ε < 1`, `0 < δ`, and at
  least `36 S / ε · log(16 S (S + 1) / δ) + S` probes, level 3 fails to end well, consistent with
  its hypothesis and tree parting on at most `ε` of the probes (`EndsWell` over
  `searchSet ∪ endsSet`, the probes that search or stop undecided at their start or end sift) or a
  harvest whose sampler yields `y` and puts at most `A·T + η/y` of its law on the states read
  undecided at most `T` of the time, with chance at most `δ`. Here
  `A = (|Q| + 1)(6/y + 16(2 + ⌈log₂ L⌉))` and `η = p₀ log(8(|Q| + 1)K/δ)`, with
  `K = (1 + (|Q| + 1)³|Σ|)^|Q| (|Q| + 3)^((|Q| + 2)|Σ|)` counting the trees level 3 can grow with an
  edge map into each one's leaves. A probe's start and end sifts read at most `3(|Q| + 1)`
  strings, and a search queries at most `2 + ⌈log₂ L⌉` positions, each sift reading at most
  `|Q| + 1` strings. Making a string undecided can only stop the search sooner, so the search
  reaching a position and a string read there undecided have chance at most the product of theirs.
  At each state's rate `t` and every tree, but for chance `δ/8` at each site over the states, the
  probes stopping undecided at their start or end at the states read undecided at most `t` of the
  time are at most `6(|Q| + 1)t + η`, and those stopping at an undecided middle there at most
  `2(2 + ⌈log₂ L⌉)(|Q| + 1)t` of the searches plus `η` (`harvest_noise`); a middles harvest that
  ends well finds at least an eighth of the searches. The sampler (`harvestLaw`) draws a fresh
  probe, walks it under the final hypothesis and finds the probe's undecided string at the site, as
  the tests counted it.
- `OrthoDFA/Level4Claim.lean` — `Level4RoundCorrect`: level 3's algorithm, unchanged, with each read
  on its state's wrong side with chance at most `εw`, fails to end well with chance at most `δ`
  under level 3's premises, `η`'s `log(8(|Q| + 1)K/δ)` becoming `log(16(|Q| + 1)K/δ)`, and
  `εw (N w|Q| + |Q|(|Q| + 1)²|Σ|) ≤ δ/8`, `N` the budget and `w = min(L + 1, 3 + ⌈log₂ L⌉)`. A
  step that reads no string wrong splits and learns only on real targets. A probe sifts at most
  `w` of its prefixes, its search's queries and the successor of its walk's end, each sift reading
  at most `|Q|` strings, and at most `|Q|` splits re-sift `(|Q| + 1)²|Σ|` each; so by
  `RaceReadsBound` over the strings each step reads the premise keeps every string the round reads
  right but for chance `δ/8`. It is level 5 at `κ = 0`. The sampler's bounds take `δ/16` at each
  site and the stretches `3δ/4`.
- `OrthoDFA/Level5Claim.lean` — `Level5RoundCorrect`: level 3's algorithm, unchanged, with each read
  on its state's wrong side with chance at most `εw + κ U q` (`KappaReadOracle`), fails to end well
  with chance at most `δ` under level 4's premises with the last replaced by
  `εw (N w|Q| + |Q|(|Q| + 1)²|Σ|) + κ ((16/5)S log(8S(S + 1)/δ) + 1 + |Q|(|Q| + 1)²|Σ|) ≤ δ/8`.
  A string is read wrong at most `κ` times as often as undecided, beyond `εw`; a
  probe's sifts stop at their first undecided read, and the tests count it: under the `j`-th
  hypothesis, until a test fires, the probes stopping undecided number at most `(16/5) log(1/a_j)`,
  whatever the stretch's length. So by `RaceReadsBound` over the strings each step reads, the round reads some string
  wrong with chance at most `δ/8`.
- `OrthoDFA/AllowanceClaim.lean` — `Level3AllowanceCorrect`: level 3's algorithm, its reads
  independent and undecided at their state's rate but with no premise on wrong reads, fails to end
  well with chance at most `7δ/8` plus the chance that a step grows its tree past `|Q| + f + 2`
  leaves (`GrowsRun`), under level 5's premises with `S`, `K`, `A` and `η` taken at `|Q| + f`
  states. Level 3's stretches use right reads only to keep the tree in its class; the proof runs
  level 5's frame on the target padded with `f` unreachable states, with the leaf count as the
  invariant.
- `OrthoDFA/FamilyOracle.lean` — `FamilyKappaOracle`: a suffix family's reads (a DFA target, a
  suffix-free family and a band passing `BandPasses`) are level 5's `KappaReadOracle`, each state's
  side the one it reads more often (`readSide`), `εw` the band's `ε` and `κ` its `κ`, since the
  rarer side's `max ε (κ U)` is at most `ε + κ U`.
- `OrthoDFA/SuffixFamily.lean` — `SuffixFamily`: the oracle, DFA, family and band, with its reads
  and each state's read distribution.
- `OrthoDFA/LearnerLoop.lean` — the loop, `indecisiveLStarLoop`: each round clusters, runs level 3
  reading the family at the gate's cut, and hands its harvest's sampler on as a population.
- `OrthoDFA/LoopDFA.lean` — `LearnerLoopDFA` and `LearnerLoopDFASimple`, `pickLeaf` and `rerootedDFA`.
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
- `OrthoDFA/Utils/RaceTail.lean` — `RaceTailBound`: as `RaceReadsBound`, counting the fresh-bad
  rounds, those where a string no earlier round queried reads bad. Each round is fixed by what it
  and the earlier rounds read while fewer than `f` fresh-bad rounds precede it. If the new strings
  number at most `b` and those in `V` at most `m` over those rounds, each round costing at most `w`,
  then at least `f` rounds are fresh-bad with chance at most `exp(t'(ε b + κ m))/(1 + t)^f` for
  `t w ≤ 1 − e^{−t' w}`. The proof: `(1 + t)^c e^{−t'(ε q + κ v)}` at the counts so far has mean at
  most `1`.

- `OrthoDFA/Utils/BitSum.lean` — `BitSumTails` (Hoeffding, 1956): a sum of independent bits at
  any rates lands at or below a count at least two under its mean no more often than a binomial
  with as many trials or more at a rate the mean clears, and likewise above.

- `OrthoDFA/Utils/SuccessiveDraws.lean` — `SuccessiveDrawsTails`: over independent draws, where
  `T` with any fewer than `n` more strings holds at most `t` of the law, the draws among the first
  `n` that land in `T` or repeat an earlier one reach a count no more often than `Bin(n, t)` does,
  and the distinct draws outside `T` that land in `P` reach a count no more often than
  `Bin(n, D(P))`, and stay at or below one no more often than `Bin(n, D(P) − t)`, does.

- `OrthoDFA/Utils/CoveredStart.lean` — `CoveredStartCorrect`: where every state of a set `S` has its
  edges learned and true and is read on its true side, the hypothesis from a state's true leaf
  (`Random.leafOf`) misjudges at most `γ` of the uniform strings of length `L` when re-rooting the
  target at that state classifies `1 - γ` of them with its run staying in `S`
  (`rerootAccuracy`). Also `uniformStrings`.

- `OrthoDFA/Utils/TripartiteSearch.lean` — `TripartiteSearchCorrect`: each result says what
  `agrees` is where it points, the queries stay between the endpoints and number at most
  `2 + 3⌈log₂(hi − lo)⌉`, or `2 + ⌈log₂(hi − lo)⌉` without recovery, and with nothing undecided it
  finds the turn exactly.

- `OrthoDFA/Utils/DTree.lean` — `SiftWithPathCorrect`: the reads are the string followed by each
  midfix on its way, no more than the tree is deep; a decided sift ends at a leaf, along the path
  its reads decide, and an undecided one at its only undecided read. `SiftAndSplitCorrect`: `sift`
  lands where `siftWithPath` does, strings at different leaves read decidedly apart where their
  paths part, and a split replaces its leaf by two children, reads its midfix there, and changes
  only the sifts that landed on it.

- `OrthoDFA/Utils/HypothesisDFA.lean` — `WalkCorrect`: the walk starts at its leaf, follows one
  learned edge per letter, stops short only at an unlearned one, and stays on the leaves when the
  edges do. `SplitAndLearnCorrect`: splitting a leaf, or learning an edge from a witness, keeps
  every learned edge witnessed (`EdgesOK`).

- `OrthoDFA/Utils/RateTest.lean` — `RateTestSound`: over independent draws `rateSide` says above
  with chance at most its failure chance when the true rate is at most the tested one, and below
  likewise when it is at least.

- `OrthoDFA/Utils/SeqTest.lean` — `SeqRateSound` (Ville's inequality): over independent draws,
  testing with `seqRate` at every look fires at some look with chance at most `a` when the true
  rate is on the far side of `p₀` from `p₁`.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, one L\* round
correct at each level above, and the loop's DFA correct from a start it picks, under the Python's
covered-start precondition, for a model of the loop that differs from the Python as listed above.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
