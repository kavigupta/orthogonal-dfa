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
  goodness level `θ` and the probes' prefix mass are small enough. A harvest's replay yields in
  proportion to how often the draws' prefixes read undecided.
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
  string alone, so the replay finds the anchor's prefix whenever that reads undecided. That share
  has mean `V / (L + 1 − start)`, `V = undecidedCount` being the mean count of positions from
  `start` to `L` at which a draw's prefix reads undecided. It is a sum of independent reads with
  weights at most `q / (L + 1 − start)`, so Chernoff's bound (`anchor_reads`, `Proofs/Yield.lean`)
  holds it at least `(V − 2√(q V log(1/δ))) / (L + 1 − start)` but for chance `δ`. Every state
  that is not good reads undecided more than `1.5θ` of the time, so `V ≥ 1.5θ · coverage(not
  good)`, but the yield does not depend on `θ`.

1. #417, `IdealRoundCorrect`: ideal reads; a state is bad where all its strings read undecided.
2. #419: indecision random, decided reads always right.
3. #420: rare wrong reads at every state.
4. This, `TallyRoundCorrect` and `TallyRoundFamily`: the tally loop as `sifting.read` runs it, read by
   the suffix family; a state is bad where it reads undecided more than `1.5θ` of the time.

## The statements

`OrthoDFA/TallyRound.lean` and `OrthoDFA/TallyFamily.lean` hold them, with everything they need
to be read. With `n = |Q| + S`, `A = |Σ|`, `L = P.L`, `Z = |Q| A + S + 1`,
`ρ = (S + 1) εd / (256 (n + 1) Z)`, `K = (n + 1)(2(n + 3)A + 1)(10/εd³ + 270/εd² + 1400/εd)
+ 8(n + 3) A (L + 1)`, `X = 4 (2^(n+4) A + 7K²)/δ`, `G = log X + 2 log(2 log X)`,
`V = undecidedCount P M O.dist` and a read-state bad where it reads undecided more than `1.5θ`
of the time:

    TallyRoundCorrect: ∀ M (O : TriOracle μ M) θ εd (P : PrefixDistribution α) δ S k,
      0 < δ ≤ 1 → 0 < εd ≤ 1 → 1 ≤ L → 0 ≤ θ →
      384 (L + 1)(n + 1) θ ≤ 1 →
      0 < O.κ → 4 O.κ (8 (n + 3)(L + 1) √(A ρ / 3) + 1) ≤ ρ →
      0 ≤ O.ε → (L + 1)(n + 1) O.ε + 3 · tallyLog σ S δ · prefixMax P.D k ≤ ρ →
      ∃ C : TallyCfg, C.k = k ∧ C.εd = εd ∧ C.budget ≤ K · G ∧
        P[the round over C.budget probes from P.D does not end well, its replay yielding
          (V − 2√(P.q V log(1/δ))) / (L + 1 − P.start)] ≤ 2δ

    TallyRoundFamily: TallyRoundCorrect at fam.oracle : TriOracle μ fam.M.

`16ρ` is about the share of the probes at which the budget affords a record that is not true. The
proof (`Proofs/TallyCorrect.lean`) chooses, with `N = n + 3`, `L₁ = L + 1` and `s = √(A ρ/3)`:
`Lmax = N`, `θpt = 1/4`, `θs = 1/2`, `φpt = 3εd/4`, `θe = 8s + 1/(L₁ N)`, `φe = 4s L₁/A`,
`a = e^{−G}`, `n₀ = ⌈(512 + 32/(9εd²)) G⌉`, `exc = 6 L₁ G`, `m` the records the caps ask for and
the budget `TallyRiskBound` asks for. `θe φe ≥ 32 L₁ ρ/3` holds the edge field's prefix mass to the
same cap as the records', and `θe L₁ N = 2 N A φe` then minimises the edges' share of `κ`'s cap.
`θpt` is at its largest, since `θ` and the budget both scale with it.

At `|Q| = 10`, `|Σ| = 2`, `S = 2`, `δ = 0.01`, with `ε` and the prefix mass each taking the
whole of their shared cap (half each halves both):

| | `θ ≤` | `κ ≤` | `ε ≤` | `prefixMax ≤` | uniform binary `k ≥` | budget `≤ K G` |
|---|---|---|---|---|---|---|
| `L = 30`, `εd = 1` | 6.5e-6 | 4.9e-7 | 9.7e-8 | 1.5e-8 | 27 | 6.0e7 |
| `L = 30`, `εd = 0.1` | 6.5e-6 | 1.4e-7 | 9.7e-9 | 1.5e-9 | 30 | 2.1e9 |
| `L = 60`, `εd = 1` | 3.3e-6 | 2.6e-7 | 4.9e-8 | 1.5e-8 | 27 | 6.0e7 |
| `L = 60`, `εd = 0.1` | 3.3e-6 | 7.6e-8 | 4.9e-9 | 1.5e-9 | 30 | 2.1e9 |

The chosen budget is within 5% of `K G`. Against the earlier constants (`2^13`, `2^14`, `2^18`) this
is 21× more `θ` at `εd = 1` (213× at 0.1), about 2500× (7000×) more `κ`, 110× more `ε`, 1100× more
prefix mass (`k ≥ 27` against 37), and a 140× (5000×) smaller budget. Every condition holds at
`L = 30` for `k` from 27 to 30 at `εd = 1`; at `εd = 0.1` only `k = 30 = L` with `ε` near 0 is
left. The yield is positive once `V > 4 q log(1/δ)`: at `L = 30` (`start = 15`, `q = 2^-15`)
that is a mean undecided rate above `3.5e-5` over the 16 anchors, and at `L = 60` above `5.5e-10`.

## The internal bound

`Proofs/TallyRiskDefs.lean` states what the claim is derived from; none of it needs to be read to
trust the claim.

    TallyRoundRisk: ∀ M (O : TriOracle μ M) θ (P : PrefixDistribution α) C κs δ,
      2 · O.κ ≤ κs → 0 < δ ≤ 1 →
      ∫⁻ ω, P[the round over C.budget probes from P.D does not end well, its replay yielding
          (V − 2√(P.q V log(1/δ))) / (L + 1 − P.start)] ∂μ
        ≤ roundRisk σ O.κ O.ε θ P.D P.L C κs + δ

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
`|Q| = 10`, `|Σ| = 2`, `L = 30`, `δ = 0.01` and `εd = 1/4`, the chosen settings admit
`κ ≤ 2.3e-7`, `θ ≤ 6.5e-6`, `ε ≤ 2.4e-8` and `p₀ ≤ 3.6e-9` at a budget of `3.9e8`, against
`κ ≤ 5e-8`, `θ ≤ 6e-6`, `ε ≤ 1e-8` and `p₀ = 2^-30` at `3.5e8` for the earlier hand-picked
settings.

## Open

- **The shipped Python uses fixed settings**, not the ones `TallyRoundCorrect` chooses.

- **The middles field is per search.** A search makes up to `(L + 1)(n + 1)` first reads, and in a
  tree of depth `n + 1` at good read-states it stops undecided with chance near `1 − e^{−1.5θ R}`
  for `R` its reads, so the bound is the worst case, not slack. `pt_mean_le` unions over every
  position a search could stop at; the bisection sifts at most `2 + 3⌈log₂ L⌉` of them, which
  would raise `θ`'s cap by `(L + 1)/(2 + 3⌈log₂ L⌉)` (1.8× at `L = 30`, 3× at `L = 60`), but
  needs an adaptive, first-read count of the middles like `edge_first_mean`'s.
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
