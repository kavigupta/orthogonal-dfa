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
- `TallyRound`: the tally loop's round, its reads one draw of accept, reject or undecided per
  string with its state's law, ends well but for the chance that the noise event `TallyE` fails
  and the bounds of its parts.
- `TallyRoundFamily`: the same round read by the suffix family, with `TallyE` replaced by its
  tails, from a DFA target, a suffix-free family, `BandPasses` and parameter conditions.

## The round, by refinement

Each level proves one L\* round correct and states its ending through `OrthoDFA/RoundEnd.lean`:
the round ends well when it is consistent, the hypothesis and the tree parting on at most `ε` of
the probes, or ends in a harvest more than half of whose strings are at bad states. What makes a
state bad is each level's own.

1. #417, `IdealRoundCorrect`: ideal reads; a state is bad where all its strings read undecided.
2. #419: indecision random, decided reads always right.
3. #420: rare wrong reads at every state.
4. This, `TallyRound` and `TallyRoundFamily`: the tally loop as `sifting.read` runs it, read by
   the suffix family; a state is bad where it reads undecided more than `1.5θ` of the time.

## `TallyRoundFamily`

For any DFA target `M` with `|Q|` states, suffix-free family `F`, band passing `BandPasses` at
`ε, ε₂, κ`, and parameters meeting the conditions, there is a read model whose laws are the
family's, at `θ` and `κs`, under which the round over `T` probes fails to end well with chance
at most

* the records field's tail,
  `K · E · exp(−(ρ − 2ε(L + 1)(n + 1)) / (2p₀(1 + 4(κ(L + 1) + ε(L + 1)(n + 1)))))`;
* the edges' tail, `K · E · (n + 2)|Σ| · exp(−φe(θe/2 − 2θg)/2 / (2p₀(L + 1)(1 + 6θ(n + 1))))`;
* the start's tail, `K · exp(θ₁(e^{l p₀} − 1)/p₀ − lθgs)`;
* the middles' tail, `K · E · exp(−θgpt φpt/2 / (2p₀(1 + 4(L + 1)θ₁)))`;
* `fakeRun`, the records that are not true racing the tests' allowances;
* `(S + |Q| + 1) · subOpen`, a sub-round keeping its tree too long;
* `(2^(Lmax+1)|Σ| + T + 3T²) a`, the harvests' and success's tests,

where `n = |Q| + S`, `K = (1 + (n + 1)³|Σ|)ⁿ` bounds the class of trees, `E = (n + 3)^((n + 2)|Σ|)`
their edge maps, `θ₁ = 1.5θ(n + 1)`, and `p₀` the largest mass of a probe's length-`k` prefix.
The conditions are, besides the tests' binomial levels, `0 < κ ≤ κs/2`,
`2ε(L + 1)(n + 1) ≤ ρ`, `θg ≥ 3θ`, `θe ≥ 4θg`, `θgpt ≥ 3(L + 1)(n + 1)θ`, `2θgpt ≤ 1`,
`exc > 4(L + 1)(1 + 4θg Lmax) log(1/a)`, `k ≤ |x| ≤ L`, and `(e^η − 1)κs(L + 1) ≤ 1 − e^{−ν(L+1)}`.

## Where the conditions hold

They hold together only far from the shipped parameters. At `|Q| = 10`, `S = 5`, `|Σ| = 2`,
with the termination tests at `m = 30`, `εd = 0.1` (`W ≈ 8e8` probes, `a ≈ 5e-22`):

- the records field needs `p₀ ≲ 7e-13`, its slack `ρ` being held to about `1/W` by `fakeRun`; with
  uniform letters that is `k ≥ 40`, and so `L ≥ 40`. At `L = 30` the conditions cannot hold.
- the middles field needs `2θgpt` well under the middles test's rate, so `θ ≲ 1e-7`. At
  `θ = 0.004` it needs `θgpt ≥ 5.95`.
- `fakeRun` needs `κ ≲ 8e-7`; the shipped band has `κ = 4e-3`.
- the band's cross limit needs `ε ≲ 1e-14`; the shipped one is `1e-10`.

One assignment meeting every condition, the binomial ones checked numerically: `L = 60`, `k = 54`,
`θ = 1e-7`, `κ = 7.7e-7`, `ε = 1e-14`, `ρ = 3.5e-10`, `p₀ = 5.8e-17`.

## Open

- **The middles field is per search.** A search makes up to `(L + 1)(n + 1)` first reads, and in
  a tree of depth `n + 1` at good read-states it stops undecided with chance near
  `1 − e^{−1.5θ R}` for `R` its reads, so the bound is the worst case, not slack.
- **`p₀` and `k`.** Every field concentrates over the probes' length-`k` prefixes, with a union over
  the class and its edge maps. The planned replacement (a): the tests count each string once, at
  its first read in the round, and a harvest is judged over those new strings; each new string's
  read is then fresh when charged, so the good read-states' undecided new strings at a place are
  at most binomial in the new strings read there, with no noise field, no union and no `p₀`.
- **Reads per probe.** The edge test counts first reads (b1: a probe's first reads are fresh), so
  its allowance is `θe ≥ 12θ` a read; `fakeRun` charges every probe the most first reads it can
  make, `(L + 1) Lmax`, against it. A bound `R̄` on the mean first reads a probe makes would put
  `R̄` there instead, and is not proved.
- **`W`, `κ` and `ε`.** `fakeRun` holds `ρ` to about `1/W`, and with it `κ` and `ε` far below the
  shipped band's. `FakeRace` races the records that are not true against the undecided strings
  the tests let through, counted with their multiplicity; with each string counted once, a heavy
  string's undecided read is one count while its minority twin is a record at every probe
  through it.
- Each theorem is per round, with the family, the seed and the band fixed, and treats the noise
  as fresh. Reads made in earlier rounds, which pin bits a later round reads again, are not
  modelled.

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
  its rarer decided side at most `ε`, or at most `κ` times as often as it is undecided.
  `BandPasses` is the check the band selection rule `evidence_margin_for_population_size` makes, at
  `ε = cross_limit`, `ε₂ = MINORITY_READ_LIMIT` and `κ = MINORITY_UNDECIDED_RATIO`.
- `OrthoDFA/RoundEnd.lean` — `EndsWell`, the ending every level of the round claims.
- `OrthoDFA/IdealRound.lean` — the round with ideal reads, and `IdealRoundCorrect`.
- `OrthoDFA/TallyLoop.lean`, `OrthoDFA/TallyRound.lean` — the tally loop and `TallyRound`, as
  sub-rounds. Probes walk from `k` as `sifting.read` does; a clean disagreement records its prefix
  and target at its edge. Each place keeps its count over the round: the start's undecided read
  strings, and each edge's first reads and those of them read undecided; the start's undecided
  rate above `θs`, or an edge's undecided first reads exceeding `θe` of its first reads by `exc`
  of them, harvests; the stretch's rate of searches stopping at an undecided middle above `θpt`
  harvests those middles; the stretch's disagreement rate settling below `εd` succeeds. Otherwise one edge with `m` records
  at a target it does not point at is redirected there, or its leaf splits where its own target
  also has `m`; a split drops every record. A stretch starts afresh at every change of hypothesis.
  Each law reads its rarer decided side at most `κ` times as often as it is undecided. A sub-round
  runs at one tree of the class (genuine splits and at most `S` that are not) until a split,
  genuine (real) or not (fake), or the round ends. A split that is not genuine needs `m` records
  that are not true made in its sub-round, so while the round's such records stay below
  `(S + 1) m` its trees stay in the class. `SubRound` bounds a sub-round's chance of keeping its
  tree over `subT` probes with fewer than `B` records that are not true; `FakeRace` bounds the
  round's records that are not true reaching `(S + 1) m`, given `TallyE`; `HarvestGood` bounds a
  harvest that is not mostly at read-states that are not good. Proved in
  `Proofs/TallyBudget.lean`: `round_budget` composes sub-rounds, and `tally_round_of` gives
  `TallyRound` from `SubRound`, `FakeRace`, `HarvestGood` and `SuccessSound`. `HarvestGood` is
  proved in `Proofs/TallyHarvest.lean`: at each edge the good read-states' undecided reads less
  `2θg` of its reads exponentiate to a supermartingale while the tree is in the class, and at the
  start and over a stretch's undecided middles they are at most binomial. `SubRound` is proved in
  `Proofs/TallyKeep.lean`: a hypothesis recording less than `θr` of the time and keeping over
  `nEnd` probes has its middle-stopping searches rarer than `θpt'` of its searches, or its
  disagreements too many for success; each new hypothesis learns or redirects an edge, lowering a
  potential of at most `2 Lmax |Σ|`; and every record raises its edge and target towards the `m`
  at which it is fixed, a true one at one of `|Q| |Σ|` edges and targets, so with fewer than `B`
  that are not true the sub-round holds at most `|Q| |Σ| m + B - 1` records and its probes at
  hypotheses recording at least `θr` of the time are one binomial over `nRec`. `FakeRace` is
  proved in `Proofs/TallyRace.lean`: `TallyE.spurious` bounds a probe's record not being true by
  `κ` times its undecided strings plus `ρ`, the edges' and the start's tests hold those strings
  over the round to `θe` of an edge's first reads past `Xe` once the edge is charged `φe` of the
  probes (an edge charged less holds them to that) and `θs'` of the probes past `Xs`, so the
  records that are not true less `ν/η` times the strings' excess exponentiate to a supermartingale
  up to `ν (θe (L + 1) Lmax + θs' + 2 Lmax |Σ| φe) + (e^η - 1) ρ` a probe.
  `Proofs/QueryTree.lean`, `Proofs/TallyQuery.lean` and `Proofs/TallyFirsts.lean` give the
  fields' per-probe means: a probe is a query tree over independent reads (`run_probeQ`), and a
  string's first read in it is fresh, so a record not being true, which needs a read on its
  state's rarer side (`untrue_minority`), has chance at most `κ` times the probe's undecided first
  reads, which the tests count (`und_twins`), plus `ε` times its first reads (`spurious_mean`);
  and an edge's good undecided first reads average at most `1.5θ` of its first reads
  (`edge_first_mean`).
- `OrthoDFA/TallyFamily.lean` — `TallyRoundFamily`, above. Proved in `Proofs/TallyFamily.lean`
  from `FamilyReadBound` (`readModel_of_family`), `TallyRound` and `tallyE_le`
  (`Proofs/TallyEvent.lean`), which unions the fields' tails: `spurious_le`
  (`Proofs/TallySpurNoise.lean`), `goodEdge_le` (`Proofs/TallyEdgeRead.lean`), `goodStart_le`
  (`Proofs/TallyNoise.lean`) and `goodPT_le` (`Proofs/TallyPTNoise.lean`), each by `block_tail`
  (`Proofs/TallyEdgeNoise.lean`): per-probe quantities read only through strings sharing the
  probe's length-`k` prefix are independent across prefixes.
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
