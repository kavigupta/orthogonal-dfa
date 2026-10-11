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
- `TallyRound`: with a `ReadOracle` reading every string independently with its state's law in a
  `ReadModel`, the tally loop's round over `C.budget` probes from `D` fails to end well with chance
  at most `roundRisk`.
- `TallyRiskBound`: `roundRisk ≤ δ` under explicit conditions on the read model and the loop's
  settings.
- `TallyRoundFamily`: the same, read by a `SuffixFamily` (a DFA target, a suffix-free family and
  a band passing `BandPasses`), at the family's read model.

## The round, by refinement

Each level proves one L\* round correct and states its ending through `OrthoDFA/RoundEnd.lean`:
the round ends well when it is consistent, the hypothesis and the tree parting on at most `ε` of
the probes, or ends in a harvest more than half of whose strings are at bad states and whose
sampler is spread and yields. What makes a state bad is each level's own.

The harvest's sampler replays the final hypothesis on a fresh draw from an anchor drawn uniformly
from `⌈L/2⌉, …, L`, the positions from which #415's `covered_states` counts a visit (here
`tallyReplay`: the strings a probe walked from the anchor first reads undecided).
- `Spread`: it takes no string `t` more often than
  `B(t) = ∑_{⌈L/2⌉ ≤ i ≤ |t|} (min(i, L) + 1 − ⌈L/2⌉)/(L + 1 − ⌈L/2⌉) · D(the draw's first i
  letters are t's)`. Every string a probe walked from `e` reads extends a prefix of the draw at
  least `e` long, so this holds for any hypothesis (`Proofs/TallyReplay.lean`, by
  `spread_of_prefix` in `Proofs/RoundEnd.lean`).
- `Yields y`: it finds something on at least `y` of its draws and anchors. A tree's root reads its
  string alone, so the replay finds the anchor's prefix whenever that reads undecided. Over the
  independent reads (`anchor_reads`, `Proofs/Yield.lean`) that share is at least `1.5θ` times
  `coverage(not good) / (L + 1 − ⌈L/2⌉)` less `t`, but for chance `exp(−2t²(L + 1 − ⌈L/2⌉)/q)`,
  `coverage` being the share of draws visiting a read-state that is not good at a position from
  `⌈L/2⌉` on, and `q` the most a draw puts on starting with any one string past `⌈L/2⌉`.

1. #417, `IdealRoundCorrect`: ideal reads; a state is bad where all its strings read undecided.
2. #419: indecision random, decided reads always right.
3. #420: rare wrong reads at every state.
4. This, `TallyRound` and `TallyRoundFamily`: the tally loop as `sifting.read` runs it, read by
   the suffix family; a state is bad where it reads undecided more than `1.5θ` of the time.

## The statements

`OrthoDFA/TallyRound.lean` and `OrthoDFA/TallyFamily.lean` hold them, with everything they need
to be read:

    TallyRound: ∀ G (R : ReadOracle μ G) D L, (∀ᵐ x ∂D, |x| ≤ L) → ∀ C κs t q, 2κ ≤ κs →
      0 ≤ t → 0 < q → (no string past ⌈L/2⌉ starts more than q of D) →
      ∫⁻ ω, P[the round over C.budget probes from D does not end well] ∂μ
        ≤ roundRisk G D L C κs + exp(−2t²(L + 1 − ⌈L/2⌉)/q)

    TallyRoundFamily: ∀ (fam : SuffixFamily μ α σ) D L, (∀ᵐ x ∂D, |x| ≤ L) → ∀ C θ κs t q,
      2 · fam.κ ≤ κs → (as above) →
      ∫⁻ ω, P[...] ∂μ ≤ roundRisk (fam.readModel θ) D L C κs + exp(−2t²(L + 1 − ⌈L/2⌉)/q)

`roundRisk G D L C κs` is the least, over the proof's free parameters `kn : Knobs` meeting
`kn.Admissible G D L C κs`, of `roundBound G D L C kn`, the sum of:

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

where `n = |Q| + S`, `K = classCount` bounds the class of trees, `E = edgeMapCount` their edge
maps, `θ₁ = 1.5θ(n + 1)`, and `p₀ = prefixMax D k`, the largest mass of a length-`k` prefix.
`Knobs.Admissible` lists the relations, among them `0 < κ`, `2ε(L + 1)(n + 1) ≤ ρ`, `θg ≥ 3θ`,
`θe ≥ 4θg`, `θgpt ≥ 3(L + 1)(n + 1)θ`, `2θgpt ≤ 1`, `exc > 4(L + 1)(1 + 4θg Lmax) log(1/a)`,
`(e^η − 1)κs(L + 1) ≤ 1 − e^{−ν(L+1)}`, the tests' binomial levels, and the budget covering
`(S + |Q| + 1)` sub-rounds. A probe shorter than `k` agrees and reads nothing.

## The closed form

`TallyRiskBound` picks the knobs as explicit functions of the primitives and makes each of the
four noise tails and `fakeRun` at most `δ/8`, and every test and termination term at most `a`.
With `n = |Q| + S`, `p₀ = prefixMax D k`, `lg = log(1/a)` and
`Λ = 4(n + 2)|Σ| log((n + 3)|Σ|/δ) ≥ log(8 K E (n + 2)|Σ| / δ)`, its conditions are:

* the read model: `0 < κ`, `2κ ≤ κs`, `4κs(L + 1) ≤ 1`, `0 ≤ ε`, `4ε(L + 1)(n + 1) ≤ 1`, `0 ≤ θ`;
* the tests' rates: `24θ ≤ θe`, `96(L + 1)(n + 1)θ ≤ θpt ≤ 1/4`, `24(n + 1)θ + 8p₀Λ ≤ θs ≤ 1`,
  `φpt ≤ 3εd/4`, `εd ≤ 1`;
* the prefix mass: `32(L + 1)p₀Λ ≤ φe θe` and `256 p₀Λ ≤ θpt φpt`;
* the tests' counts: `n₀ ≥ 32 lg/θs²`, `32 lg/θpt²`, `2 lg/φpt²`, and
  `4(L + 1)(1 + 12θ Lmax) lg < exc ≤ 8(L + 1)(1 + 12θ Lmax) lg`;
* the level: `4(2^(Lmax+1)|Σ| + 7 budget²) a ≤ δ`;
* the budget: `budget ≥ (n + 1)((4 Lmax|Σ| + 1)(8 lg/(εd² θpt²) + 4n₀/(3εd) + 2)
  + 8(|Q||Σ| + S + 1)m/εd + 32 lg/εd² + 1)`;
* the records cap: `(S + 1)m` at least `4 log(8/δ)`, `16κs(16(L + 1)(1 + 12θ Lmax) Lmax|Σ| lg + n₀)`,
  `16κs budget (θe(L + 1)Lmax + 2θs + 2 Lmax|Σ| φe)` and `16 budget (ε(L + 1)(n + 1) + 3p₀Λ)`.

Since the budget grows with `m`, the last two give, roughly,
`κs ≲ (S + 1)εd / (128(n + 1)(|Q||Σ| + S + 1)(L + 1) Lmax θe)` and
`p₀ ≲ (S + 1)εd / (384(n + 1)(|Q||Σ| + S + 1)Λ)`.

At `|Q| = 10`, `|Σ| = 2`, `L = 30`, `δ = 0.01`, every condition holds at `S = 2`, `Lmax = 15`,
`εd = θpt = θs = 1/4`, `φpt = 3/16`, `θe = 0.1`, `φe = 0.8`, `n₀ = 24215`, `m = 17826`,
`budget = 3.5e8`, `a = 2.9e-21`, for any read model with `κ ≤ 5e-8`, `θ ≤ 6e-6`, `ε ≤ 1e-8`, and
`p₀ = 2^-30`, which uniform length-30 binary probes give at `k = 30`. The shipped band's
`κ = 4e-3` and `θ = 0.004` are far outside this; its `ε = 1e-10` is inside.

## Open

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
- `OrthoDFA/TallyLoop.lean`, `OrthoDFA/TallyRound.lean` — the tally loop, `TallyRound` and
  `TallyRiskBound`;
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
  `tally_round_of` gives `TallyRound` from `SubRound`, `FakeRace`, `HarvestGood` and `SuccessSound`.
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
  `SuffixFamily.readModel`, and `TallyRoundFamily`. Proved in `Proofs/TallyFamily.lean` from
  `TallyRound`, the family's reads being independent with its laws (`Proofs/SuffixFamily.lean`,
  from `FamilyReadBound`).
- `Proofs/TallyRiskBound.lean` — `TallyRiskBound`: explicit knobs, Hoeffding's lower binomial tail
  for the termination tests, and `tallyLog` bounding the unions' counts.
- `Proofs/TallyRisk.lean` — `TallyRound` at each admissible choice of knobs: `TallyRoundGiven`
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
