# Machine-checked correctness of the E-L\* clustering algorithm

## The theorems

Every theorem below is proved in Lean with no `sorry`, using only `propext`, `Classical.choice`
and `Quot.sound` (`OrthoDFA/Verify.lean` prints this).

- `ClusteringGuarantee`: with probability at least `1 − δ` the clustering loop stops, and the
  family it returns cuts all but `εcov` of the uniform pool as the noiseless oracle does, at a
  polynomial number of queries.
- `ClusteringQualityGuarantee`: averaged over a population, the returned family decides a prefix
  wrongly, or leaves it undecided, only a bounded share of the time.
- `FamilyReadBound`: the suffix family's reads are independent across strings, every DFA state
  has one law its strings read with, and the band's check bounds each law.
- `IdealRoundCorrect`: with every read fixed by its state, one L\* round ends well but for
  chance `(P + 1)(1 − ε)^n`.
- `TallyRoundCorrect`: with a `TriOracle` reading every string independently with its state's
  law, settings chosen by the proof make the tally loop's round over at most `K · G` probes from a
  `PrefixDistribution` fail to end well with chance at most `2δ`, once the laws' `κ` and `ε`, the
  goodness level `θ` and the probes' prefix mass are small enough.
- `TallyRoundFamily`: the same, read by a `SuffixFamily` (a DFA target, a suffix-free family and
  a band passing `BandPasses`), at `fam.oracle`.

## The round, by refinement

Each level proves one L\* round correct and states its ending through `OrthoDFA/RoundEnd.lean`:
the round ends well when it is consistent, the hypothesis and the tree parting on at most `ε` of
the probes, or ends in a harvest more than half of whose strings are at bad states and whose
sampler is spread and yields. What makes a state bad is each level's own.

The probes' law is a `PrefixDistribution` (`OrthoDFA/RoundEnd.lean`): `D`, with lengths at most
`L`, a first anchor `start ≤ L`, and `q`, the most `D` puts on starting with any one string of
length from `start` to `L`. The harvest's sampler replays the final hypothesis on a fresh draw from
an anchor drawn uniformly from `start, …, L` (here `tallyReplay`: the strings a probe walked from
the anchor first reads undecided). No argument needs more of `start` than `start ≤ L`, which keeps
the anchors nonempty. #415's `preconditions.start_length`, `⌈L/2⌉`, is one choice: there the
anchors are the positions from which `covered_states` counts a visit.
- `Spread`: it takes no string `t` more often than
  `B(t) = ∑_{start ≤ i ≤ |t|} (min(i, L) + 1 − start)/(L + 1 − start) · D(the draw's first i
  letters are t's)`. Every string a probe walked from `e` reads extends a prefix of the draw at
  least `e` long, so this holds for any hypothesis (`Proofs/TallyReplay.lean`, by
  `spread_of_prefix` in `Proofs/RoundEnd.lean`).
- `Yields y`: it finds something on at least `y` of its draws and anchors. A tree's root reads its
  string alone, so the replay finds the anchor's prefix whenever that reads undecided. Over the
  independent reads (`anchor_reads`, `Proofs/Yield.lean`, a Chernoff bound) that share is at
  least `0.75θ · coverage(not good) / (L + 1 − start)`, half its least mean, but for chance
  `exp(−1.5θ · coverage(not good) / (16q))`, `coverage` being the share of draws visiting a
  read-state that is not good at a position from `start` to `L`.

1. #417, `IdealRoundCorrect`: ideal reads; a state is bad where all its strings read undecided.
2. #419: indecision random, decided reads always right.
3. #420: rare wrong reads at every state.
4. This, `TallyRoundCorrect` and `TallyRoundFamily`: the tally loop as `sifting.read` runs it, read by
   the suffix family; a state is bad where it reads undecided more than `1.5θ` of the time.

## The statements

`OrthoDFA/TallyRound.lean` and `OrthoDFA/TallyFamily.lean` hold them, with everything they need
to be read. With `n = |Q| + S`, `A = |Σ|`, `L = P.L`, `K = 2^18 (n + 1)² A / εd³`,
`G = 2 log(64 (2^(n+3) A + 8K²)/δ)` and a read-state bad where it reads undecided more than `1.5θ`
of the time:

    TallyRoundCorrect: ∀ M (O : TriOracle μ M) θ εd (P : PrefixDistribution α) δ S k,
      0 < δ ≤ 1 → 0 < εd ≤ 1 → 1 ≤ L → 0 ≤ θ →
      2^13 (L + 1)(n + 1) θ ≤ εd →
      0 < O.κ → 2^14 (n + 1)² A² (L + 1)(n + 3) O.κ ≤ εd →
      0 ≤ O.ε → 2^13 (L + 1)(n + 1)² A (O.ε (n + 1) + tallyLog σ S δ · prefixMax P.D k) ≤ εd →
      11 P.q log(1/δ) ≤ θ · coverage P.D P.start P.L M bad →
      ∃ C : TallyCfg, C.k = k ∧ C.εd = εd ∧ C.budget ≤ K · G ∧
        P[the round over C.budget probes from P.D does not end well] ≤ 2δ

    TallyRoundFamily: TallyRoundCorrect at fam.oracle : TriOracle μ fam.M.

The proof chooses `Lmax = n + 3`, `θpt = 1/4`, `φpt = 3εd/4`, `θs = θe = 3εd/16`, `φe = 1`,
`a = e^{−G}`, `n₀ = ⌈1000 G/εd²⌉`, `m = ⌈3100 G/εd²⌉`, `exc = 6(L + 1)(1 + 12θ Lmax) G` and the
budget the records cap asks for (`Proofs/TallyCorrect.lean`).

At `|Q| = 10`, `|Σ| = 2`, `S = 2`, `δ = 0.01` and `εd = 1`, the conditions ask for `θ ≤ 3e-7`,
`κ ≤ 2e-10`, `ε ≤ 9e-10` and `prefixMax ≤ 1.3e-11`, so `k ≥ 37` for uniform binary probes, with a
budget of at most `8.4e9` probes (`1.1e13` at `εd = 0.1`). At `L = 30` they cannot all hold:
the prefix mass needs `k > L`, and the yield condition, with `q = 2^-15`, needs `θ ≥ 1.5e-3/c`.
At `L = 60` (`q = 2^-30`) they hold for `k ≥ 37` and `θ ≤ 1.5e-7` once the bad read-states'
coverage `c ≥ 0.3`. The constants are loose; the internal closed form below admits far more.

## The internal bound

`Proofs/TallyRiskDefs.lean` states what the claim is derived from; none of it needs to be read to
trust the claim.

    TallyRoundRisk: ∀ M (O : TriOracle μ M) θ (P : PrefixDistribution α) C κs, 2 · O.κ ≤ κs →
      ∫⁻ ω, P[the round over C.budget probes from P.D does not end well] ∂μ
        ≤ roundRisk σ O.κ O.ε θ P.D P.L C κs + exp(−1.5θ · coverage(not O.Good θ) / (16 P.q))

A read-state `q` is good (`O.Good θ q`) where it reads undecided at most `1.5θ` of the time.
`roundRisk σ κ ε θ D L C κs` is the least, over the proof's free parameters `kn : Knobs` meeting
`kn.Admissible σ κ ε θ D L C κs`, of `roundBound σ κ ε θ D L C kn`, the sum of:

* `recordsTail`, records that are not true exceeding `κs` times the undecided strings plus `ρ`:
  `K · E · exp(−(ρ − 2ε(L + 1)(n + 1)) / (2p₀(1 + 4(κ(L + 1) + ε(L + 1)(n + 1)))))`;
* `edgeTail`, an edge's good undecided first reads past `θg` of its first reads:
  `K · E · (n + 2)|Σ| · exp(−φe(θe/2 − 2θg)/2 / (2p₀(L + 1)(1 + 6θ(n + 1))))`;
* `startTail`, the good undecided start reads past `θgs`: `K · exp(θ₁(e^{l p₀} − 1)/p₀ − lθgs)`;
* `middlesTail`, searches stopping at a good undecided middle past `θgpt`:
  `K · E · exp(−θgpt φpt/2 / (2p₀(1 + 4(L + 1)θ₁)))`;
* `fakeRun`, the records that are not true racing the tests' allowances;
* `(S + |Q| + 1) · subOpen`, a sub-round keeping its tree too long;
* `testLevels`, `(2^(Lmax+1)|Σ| + budget + 3 budget²) a`, the harvests' and success's tests,

where here `K = classCount` bounds the class of trees, `E = edgeMapCount` their edge maps,
`θ₁ = 1.5θ(n + 1)`, and `p₀ = prefixMax D k`, the largest mass of a length-`k` prefix.
`TallyRiskBound` (same file) is `roundRisk ≤ δ` under explicit conditions on `κ`, `ε`, `θ` and the
settings; `TallyRoundCorrect` substitutes its chosen settings into those conditions. At
`|Q| = 10`, `|Σ| = 2`, `L = 30`, `δ = 0.01`, `TallyRiskBound`'s conditions hold at hand-picked
settings (`S = 2`, `εd = θpt = θs = 1/4`, `budget = 3.5e8`) for `κ ≤ 5e-8`, `θ ≤ 6e-6`,
`ε ≤ 1e-8` and `p₀ = 2^-30`.

## Open

- **The shipped Python uses fixed settings**, not the ones `TallyRoundCorrect` chooses.

- **The middles field is per search.** A search makes up to `(L + 1)(n + 1)` first reads, and in a
  tree of depth `n + 1` at good read-states it stops undecided with chance near `1 − e^{−1.5θ R}`
  for `R` its reads, so the bound is the worst case, not slack.
- **`prefixMax` and `k`.** Every field concentrates over the probes' length-`k` prefixes, with a
  union over the class and its edge maps. The planned replacement (a): the tests count each string
  once, at its first read in the round, and a harvest is judged over those new strings; each new
  string's read is then fresh when charged, so the good read-states' undecided new strings at a
  place are at most binomial in the new strings read there, with no noise field, no union and no
  `prefixMax`.
- **Reads per probe.** The edge test counts first reads (b1: a probe's first reads are fresh), so
  its allowance is `θe ≥ 12θ` a read; `fakeRun` charges every probe the most first reads it can
  make, `(L + 1) Lmax`, against it. A bound `R̄` on the mean first reads a probe makes would put
  `R̄` there instead, and is not proved.
- **`W` and `κ`.** `fakeRun` holds `ρ` to about `(S + 1)m/W`, and with it `κ` far below the
  shipped band's. `FakeRace` races the records that are not true against the undecided strings the
  tests let through, counted with their multiplicity; with each string counted once, a heavy
  string's undecided read is one count while its minority twin is a record at every probe through
  it.
- Each theorem is per round, with the family, the seed and the band fixed, and treats the noise as
  fresh. Reads made in earlier rounds, which pin bits a later round reads again, are not modelled.

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
- `OrthoDFA/RoundEnd.lean` — `EndsWell`, the ending every level of the round claims.
- `OrthoDFA/IdealRound.lean` — the round with ideal reads, and `IdealRoundCorrect`.
- `OrthoDFA/TallyLoop.lean`, `OrthoDFA/TallyRound.lean` — the tally loop and `TallyRoundCorrect`.
  `Proofs/TallyRiskDefs.lean` holds the internal bound (`TallyRoundRisk`, `TallyRiskBound`), and
  `Proofs/TallyCorrect.lean` derives the claim from it at the chosen settings.
  `Proofs/TallyParts.lean` states its parts, as sub-rounds. Probes walk from `k` as `sifting.read`
  does; a clean disagreement records its prefix and target at its edge. Each place keeps its count
  over the round: the start's undecided read strings, and each edge's first reads and those of them
  read undecided; the start's undecided rate above `θs`, or an edge's undecided first reads
  exceeding `θe` of its first reads by `exc` of them, harvests; the stretch's rate of searches
  stopping at an undecided middle above `θpt` harvests those middles; the stretch's disagreement
  rate settling below `εd` succeeds. Otherwise one edge with `m` records at a target it does not
  point at is redirected there, or its leaf splits where its own target also has `m`; a split drops
  every record. A stretch starts afresh at every change of hypothesis. Each law reads its rarer
  decided side at most `κ` times as often as it is undecided. A sub-round runs at one tree of the
  class (genuine splits and at most `S` that are not) until a split, genuine (real) or not (fake),
  or the round ends. A split that is not genuine needs `m` records that are not true made in its
  sub-round, so while the round's such records stay below `(S + 1) m` its trees stay in the class.
  `SubRound` bounds a sub-round's chance of keeping its tree over `subT` probes with fewer than `B`
  records that are not true; `FakeRace` bounds the round's records that are not true reaching `(S +
  1) m`, given `TallyE`; `HarvestGood` bounds a harvest that is not mostly at read-states that are
  not good. Proved in `Proofs/TallyBudget.lean`: `round_budget` composes sub-rounds, and
  `tally_round_of` gives `TallyRoundRisk` from `SubRound`, `FakeRace`, `HarvestGood` and `SuccessSound`.
  `HarvestGood` is proved in `Proofs/TallyHarvest.lean`: at each edge the good read-states'
  undecided reads less `2θg` of its reads exponentiate to a supermartingale while the tree is in the
  class, and at the start and over a stretch's undecided middles they are at most binomial.
  `SubRound` is proved in `Proofs/TallyKeep.lean`: a hypothesis recording less than `θr` of the time
  and keeping over `nEnd` probes has its middle-stopping searches rarer than `θpt'` of its searches,
  or its disagreements too many for success; each new hypothesis learns or redirects an edge,
  lowering a potential of at most `2 Lmax |Σ|`; and every record raises its edge and target towards
  the `m` at which it is fixed, a true one at one of `|Q| |Σ|` edges and targets, so with fewer than
  `B` that are not true the sub-round holds at most `|Q| |Σ| m + B - 1` records and its probes at
  hypotheses recording at least `θr` of the time are one binomial over `nRec`. `FakeRace` is proved
  in `Proofs/TallyRace.lean`: `TallyE.spurious` bounds a probe's record not being true by `κ` times
  its undecided strings plus `ρ`, the edges' and the start's tests hold those strings over the round
  to `θe` of an edge's first reads past `Xe` once the edge is charged `φe` of the probes (an edge
  charged less holds them to that) and `θs'` of the probes past `Xs`, so the records that are not
  true less `ν/η` times the strings' excess exponentiate to a supermartingale up to `ν (θe (L + 1)
  Lmax + θs' + 2 Lmax |Σ| φe) + (e^η - 1) ρ` a probe. `Proofs/QueryTree.lean`,
  `Proofs/TallyQuery.lean` and `Proofs/TallyFirsts.lean` give the fields' per-probe means: a probe
  is a query tree over independent reads (`run_probeQ`), and a string's first read in it is fresh,
  so a record not being true, which needs a read on its state's rarer side (`untrue_minority`), has
  chance at most `κ` times the probe's undecided first reads, which the tests count (`und_twins`),
  plus `ε` times its first reads (`spurious_mean`); and an edge's good undecided first reads average
  at most `1.5θ` of its first reads (`edge_first_mean`).
- `OrthoDFA/SuffixFamily.lean`, `OrthoDFA/TallyFamily.lean` — `SuffixFamily`, its reads and laws,
  `SuffixFamily.oracle`, and `TallyRoundFamily`. Proved in `Proofs/TallyCorrect.lean` from
  `TallyRoundCorrect`'s proof, the family's reads being independent with its laws (`Proofs/SuffixFamily.lean`,
  from `FamilyReadBound`).
- `Proofs/TallyRiskBound.lean` — `TallyRiskBound`: explicit knobs, Hoeffding's lower binomial tail
  for the termination tests, and `tallyLog` bounding the unions' counts.
- `Proofs/TallyRisk.lean` — `TallyRoundRisk` at each admissible choice of knobs: `TallyRoundGiven`
  (`Proofs/TallyParts.lean`, proved in `Proofs/TallyBudget.lean`) with its noise event's chance
  bounded by `tallyE_le` (`Proofs/TallyEvent.lean`), which unions the fields' tails:
  `spurious_le` (`Proofs/TallySpurNoise.lean`), `goodEdge_le` (`Proofs/TallyEdgeRead.lean`),
  `goodStart_le` (`Proofs/TallyNoise.lean`) and `goodPT_le` (`Proofs/TallyPTNoise.lean`), each by
  `block_tail` (`Proofs/TallyEdgeNoise.lean`): per-probe quantities read only through strings
  sharing the probe's length-`k` prefix are independent across prefixes.
- `OrthoDFA/Stage.lean`, `StartAtK.lean`, `Loop.lean`, `Pass.lean`, `Automaton.lean` — trees, their
  reads, walks and bisection, which the round's statements build on.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, and one L\* round
correct at each level above. It does not prove that the E-L\* learner outputs the target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
