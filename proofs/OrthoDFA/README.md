# Machine-checked correctness of the E-L\* clustering algorithm

## The theorems

Every theorem below is proved in Lean with no `sorry`, using only `propext`, `Classical.choice`
and `Quot.sound` (`OrthoDFA/Verify.lean` prints this).

The headline is `LearnerLoopDFA` (`OrthoDFA/LoopDFA.lean`, proved in `Proofs/LoopDFA.lean`): the
loop outputs a DFA, and `LearnerLoopDFASimple` states the same at explicit settings (below). The
whole run reads one noise, fixed per string as the Python's is. One-step consistency at `L − 1` is not
enough for this (`parity_toggle`, #438).

- `ClusteringGuarantee`: with probability at least `1 − δ` the clustering loop stops, and the
  family it returns cuts all but `εcov` of the uniform pool as the noiseless oracle does, at a
  polynomial number of queries.
- `ClusteringQualityGuarantee`: averaged over a population, the returned family decides a prefix
  wrongly, or leaves it undecided, only a bounded share of the time.
- `FamilyReadBound`: the suffix family's reads are independent across strings, every DFA state
  has one distribution its strings read with, and the band's check bounds each.
- `TripartiteSearchCorrect`: the search the probes use to find where the walk and the sift part
  finds what it reports, queries at most `2 + 3⌈log₂(hi − lo)⌉` positions, and finds the turn
  exactly where nothing is undecided.
- `Level2.Correct`: with every read fixed by its state, one L\* round ends well but for chance
  `δ`, over enough probes.
- `Level5Correct`: with a `TriOracle` reading every string independently with its state's
  distribution, settings chosen by the proof make `indecisiveLStarLevel5`, the counterexample
  search, over at most `K · G` probes from a `PrefixDistribution` fail to end well with chance at
  most `δ`, once the reads' `κ` and `ε`, the goodness level `θ` and the probes' prefix mass are
  small enough. A harvest's sampler yields at least `3εd/16`.
- `Level5FamilyCorrect`: the same, read by a `SuffixFamily` (a DFA target, a suffix-free family
  and a band passing `BandPasses`), at `fam.oracle`.
- `LearnerLoopDFA` and `LearnerLoopDFASimple`: the loop outputs a DFA. See below.

## The learner's DFA

`LearnerLoopDFA` asks for what the Python's `satisfies_preconditions` checks of the target, at its
length `L`: `coveredStates M L c` are the states at least `c` (`min_coverage`) of the strings of
length `L` visit at some position from `⌈L/2⌉` to `L`, and `rerootAccuracy M L S q` is the share
of those strings whose run from `q` stays in `S` and is classified as the target classifies them
(`covered_accuracy_ceiling` at the start `q`). Where some covered `q` has accuracy at least
`1 − η`, the loop, its probes uniform over the lengths `k = ⌈L/2⌉` to `L₁ = L + |Q| + 3` and
walked from `k`, stops within `|Q| + 1` rounds at a consistent round, and the leaf it then picks
classifies all but `η + ξ` of the strings of length `L` as the target does, but for chance `δ`,
once `L ≥ L₀`. A run that reaches an unlearned edge counts as wrong. The pick (`pickLeaf`): after
the loop, `n = ⌈32 log(16(|Q|+1)(|Q|+2)/δ)/ξ²⌉` strings uniform of length `L` are drawn and read
by the last round's family; each leaf scores the strings its run classifies as the family decides
them, undecided reads scoring none, and the first leaf of highest score is the start. With
`N = L₁ + 1 − k`, level 5 runs at `εd = c/(34(|Σ|+1)²(|Q|+2)N(L₁+1))` and draws at most `Kp · G`
probes a round; `L₀` is `|Q| + 4 + 4(log (E + E₂) + 100)/log |Σ|`, `E` and `E₂` free of `L`,
logarithmic in `1/δ`, `1/c`, `1/ξ`, `1/(1/2 − η₀)`, `1/pAP`, `|Q|` and `|Σ|`. At `|Q| = 3`,
`|Σ| = 2`, `η₀ = 0.1`, `c = δ = 0.01`, `pAP = 1/3` and `ξ = 0.1`, `L₀ = 1618`, `n = 33196` and the
budget is `4.2e16` probes a round (`εd = 4.9e-12`). `LearnerLoopDFASimple` states the same at explicit settings,
every hypothesis an inequality.

The probes over a window of lengths are the algorithm's change. Every prefix length is itself a
probe, so a probe and its extension by a letter that both agree, reading nothing on a rarer side
and nothing undecided, cross a learned edge between their states' true leaves (`edge_of_pair`,
`Proofs/DFABridge.lean`). A covered state is visited `c` of the time summed over the window, so an
edge out of its true leaf that is wrong or unlearned leaves `c/((|Σ|+1)N)` of the probes checked
(`edge_of_occupied`). Level 5's consistent ending answers for its checked probes
(`checkedProbes`): those that search or could learn an edge, which the success test counts or
which reset the hypothesis, those reading a prefix's extension by one of the tree's midfixes on its
state's rarer side, which the run certifies (`rarer_le`), those reading one undecided once
`undecidedMass` is at most `ū`, by Markov, and those whose start or end sift stops undecided, which
level 5's own start and edge tests certify (`sift_le`). So the bridge needs no average over the
window: the clustering, run at `εcov = min(εd/(48(|Q|+1)²), ξ/(8N))`, keeps level 5's good states
undecided rarely against its gates and reads every covered state on its true side
(`sideOf_eq_of_mass`). Every covered state's edges are then true and it accepts as
the target does, so the hypothesis from the covered start's true leaf tracks the re-rooted target
on every run that stays covered (`classify_start_le`).

The pick compares each leaf against the family's read, not the oracle's: the oracle's noise rates
on and off the language may differ, and agreement with such a read weighs a leaf's errors by
`1 − 2η` of the string's true label, which can rank a worse leaf first. The family's read of a
string of length `L` is wrong, or undecided, at most `N εcov ≤ ξ/8` of the time, so a leaf's
mean score is within that of its accuracy. Reads persist per string, so the held-out strings'
reads are fresh only where no held-out string repeats or begins with the first `k` letters of a
prefix the last round's clustering read or of one of its probes: level 5 reads only short strings
beginning with some probe's first `k` letters (`level5_congr_zone`, `Proofs/LoopSelect.lean`), and
the earlier rounds read other suffixes. That collision has chance at most
`n(6(|Q|+2)RB + Kp·G + n)/|Σ|^k`, which `L ≥ L₀` makes at most `δ/(4(|Q|+1))` for each round that
can be last. Off it, given the loop, each leaf's score is within `nξ/8` of its mean given the
strings, and that within `nξ/8` of `n` times its mean over the strings, but for chance
`4(|Q|+2)e^(−nξ²/32)` (`scoreBad_le`), so the picked leaf misjudges at most `η + ξ/2 + 3Nεcov ≤ η
+ ξ` (`pickLeaf_le`). `δ` splits half to the loop, a quarter to the scores and a quarter to
collisions.

## The loop

`indecisiveLStarLoop` (`OrthoDFA/LearnerLoop.lean`) clusters a family over the populations so far,
runs level 5 with it, and adds a harvest's sampler, scaled to a probability, as a population of
every later round. The whole run, every round's clustering and level 5, reads one noise. Each
round's clustering draws its prefixes, certification prefixes and suffixes off every one the
earlier rounds' clusterings read (`offLaw`), so no two rounds read the same string. Level 5 runs at
disagreement rate `εd / 2` and draws its probes from `D` conditioned on their first `k ≥ 1` letters
avoiding those of every prefix the round's clustering read. The band's far side is
`δ / (4 Kp G (|Q| + 2)(L + 1 + (|Q| + 2)|Σ|))`, so level 5 reads a state on its rarer side rarely
enough that its splits are all genuine. A harvest's sampler yields at least `ρ / 4` and finds a
string at most as often as the probes share its length-`k` prefix, so a population puts at most
`4 prefixMax / ρ` on a string, within `cap`, which also bounds the suffixes' atoms. Against `cap`
the clustering's collision conditions, linear in the schedule's prefix count `NP` and suffix pool
`NS`, hold, and the earlier rounds' reads, at most `6(K+1)NP` prefixes and `NS + 1` suffixes a
round, hold at most `m` of any population and `ms` of the suffixes. `NS` and `NP` are bounded by
`poolCount_le` and `prefCount_le_poly` (`counts_le`, `Proofs/LoopSimple.lean`).

Proved with no `sorry`. The rounds' draws are independent, and given the earlier rounds a round's
clustering run is drawn as `runMeasure` draws it at their populations conditioned off what they
read (`Proofs/LoopLaw.lean`, `Proofs/LoopIndep.lean`): a population's draws are made from uniforms
(`drawFrom`), and a round's population and reads are fixed by finite data, its last hypothesis's tree
and edge graphs, its reads of short strings and its clustering's prefixes and suffixes
(`Proofs/LoopCodes.lean`). A round's outcomes are measurable because, at fixed draws, each is fixed
by the oracle's answers at finitely many strings (`Proofs/LoopMeasurable.lean`), level 5 reading
only strings at most `L + 1 + maxLeaves` long (`Proofs/Level5Reads.lean`).

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
(`loop_cluster_round_le_persist`, `Proofs/LoopE1.lean`). Level 5 reads `z·v` with `v` a family
suffix and `z`'s first `k` letters a probe's, so its avoiding probes never read a string the
clustering read (`level5_congr_off`, `Proofs/Level5Avoid.lean`). Re-noising the clustering's
strings with a second, independent noise (`Oracle.renoise`, `Proofs/Renoise.lean`) leaves a fresh
family on which level 5's guarantee holds at the conditioned probes, whose prefix mass grows by at
most `1/(1 − m)` for `m ≤ |keys| · prefixMax` the mass removed (`Proofs/LoopAvoid.lean`); the actual
run is the same run, and its ending transfers back to `D` at disagreement `ε + m`, yield `(1 − m) y`
and a quarter of the sampler bad (`quarterEndsWell_of_endsWell`, `Proofs/LoopTransfer.lean`), which
still leaves a bad state with `1/(4|Q|)` of the population (`Proofs/LoopTermination.lean`).

## The round, by refinement

Each level proves one L\* round correct and states its ending through `OrthoDFA/RoundEnd.lean`:
the round ends well when it is consistent, the hypothesis and the tree parting on at most `ε` of
the probes, or ends in a harvest whose sampler finds more than half of what it finds at bad
states, is spread and yields. What makes a state bad is each level's own.

Level 5's harvest is its site. Every count starts afresh at each change of hypothesis, so the
test that fired counted the final hypothesis alone, and the sampler (`harvestLaw`) draws a fresh
probe from `D`, walks it under the final hypothesis and finds the probe's undecided first reads at
the site, as the tests counted them.
- `Spread`: it finds no string more often than draws share the string's length-`k` prefix, since
  every string a probe walked from `k` reads extends the probe's length-`k` prefix and a probe
  reads no string first twice (`Proofs/Level5Law.lean`).
- `Yields y`: it finds at least `y`, `harvestYield`, the least of `θs`, `θe φe/2` and
  `θpt φpt/2`, which the start's, an edge's and the middles' tests certify.

Levels 2, 3 and 4 claim the same ending, each with its own sampler: level 2's draws a fresh probe
under the final hypothesis and finds the undecided read it stops at.

1. #417, `Level2.Correct`: ideal reads; a state is bad where all its strings read undecided.
2. #419: indecision random, decided reads always right.
3. #420: rare wrong reads at every state.
4. This, `Level5Correct` and `Level5FamilyCorrect`: level 5, the counterexample search as
   `sifting.read` runs it (`indecisiveLStarLevel5`), read by the suffix family; a state is bad
   where it reads undecided more than `1.5θ` of the time.

## The statements

`OrthoDFA/Level5Claim.lean` and `OrthoDFA/Level5Family.lean` hold them, with everything they
need to be read; the algorithm is `indecisiveLStarLevel5` in `OrthoDFA/Level5.lean`. With
`n = |Q|`, `A = |Σ|`, `L = P.L`, `V = (n + 2)((n + 2)A + 1)`,
`G = log (4 V (2(n + 2)A + 4) / δ)`, `K = V (112/εd + 1)`, `W = (n + 2)(L + 1 + (n + 2)A)`,
`U = 4(n + 2)A log ((n + 3)A / δ)` and a
read-state bad where it reads undecided more than `1.5θ` of the time:

    Level5Correct: ∀ M (O : TriOracle μ M) θ εd (P : PrefixDistribution α) δ k J ū,
      0 < δ ≤ 1 → 0 < εd < 1 → 1 ≤ L → 0 ≤ θ →
      384 (L + 1)(n + 1) θ ≤ 1 →
      0 ≤ O.κ → 0 ≤ O.ε → 4 (O.ε + O.κ) K G W ≤ δ →
      1024 (L + 1) · U · prefixMax P.D k ≤ εd → 0 ≤ ū →
      ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
        settings.probeBudget ≤ K · G ∧
        P[let result := indecisiveLStarLevel5 settings (O.read · ω) draws
          ¬ EndsWell M bad P.D (3εd/2)
            (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart (n + 2) J ū
              result.hypothesis)
            settings.walkStart (harvestLaw P.D (O.read · ω).cut settings.walkStart
              result.hypothesis) (3εd/16) result.ending] ≤ δ + 32ū/εd

    Level5FamilyCorrect: Level5Guarantee at fam.oracle : TriOracle μ fam.M.

The settings (`Level5Settings`) are written below by their symbols: `Lmax = maxLeaves`,
`θs = startUndecidedRate`, `θe = edgeUndecidedRate`, `θpt = middleUndecidedRate`,
`εd = disagreementRate`, `θs₁ = startAltRate`, `θpt₁ = middleAltRate`,
`εd₁ = disagreementAltRate`, `φe = edgeTraffic`, `φpt = searchShare`, `a = testLevel`,
`n₀ = firstLook`, `exc = edgeExcess`, `k = walkStart` and `budget = probeBudget`.

The proof (`Proofs/Level5Correct.lean`) chooses, with `N = n + 2` and `L₁ = L + 1`: `Lmax = N`,
`θs = 1/2` against `θs₁ = 3/5`, `θpt = 1/2` against `θpt₁ = 3/4`, `εd₁ = 7εd/8`, `φpt = 3εd/4`,
`θe = φe = 3/4`, `a = e^{−G}`, `n₀ = ⌈6G⌉`, `exc = 2 L₁ (1 + 3N/4) G + 1` and
`budget = ⌈V (112 G/εd + 1)⌉`. A contradiction splits at once, so a hypothesis's searches but its
last stop at undecided middles: past `nEnd = ⌈112 G/εd⌉` probes success fires, or before them the
middles test does, and the round goes through fewer than `V` hypotheses.

At `|Q| = 10`, `|Σ| = 2`, `δ = 0.01`, with `ε` and `κ`, and the prefix mass, each taking the whole
of its cap:

| | `θ ≤` | `κ`, `ε ≤` | `prefixMax ≤` | uniform binary `k ≥` | budget `≤ K G` | yield |
|---|---|---|---|---|---|---|
| `L = 30`, `εd → 1` | 7.6e-6 | 2.9e-15 | 4.2e-8 | 25 | 5.3e5 | 0.19 |
| `L = 30`, `εd = 0.1` | 7.6e-6 | 2.9e-16 | 4.2e-9 | 28 | 5.3e6 | 0.019 |
| `L = 60`, `εd → 1` | 3.9e-6 | 1.4e-15 | 2.1e-8 | 26 | 5.3e5 | 0.19 |
| `L = 60`, `εd = 0.1` | 3.9e-6 | 1.5e-16 | 2.1e-9 | 29 | 5.3e6 | 0.019 |

The chosen budget is within 1% of `K G`. Against the records level (`recordsToMove` records to fix
an edge, `S = 2` splits that are not genuine), the budget is 11× smaller near `εd = 1` and 19× at
0.1 (`5.9e6`, `1.0e8`), the yield `3εd/16` is about 4800× larger than its
`(S + 1)εd/(256(n + 1)(|Q||Σ| + S + 1))`, and `θ` and the prefix mass are about the same; `κ` and
`ε`, which there were held by the records cap at `4.9e-7` and `9.7e-8` (near `εd = 1`, `L = 30`),
are now held by a single read on the rarer side, anywhere a probe or a re-sift could read it,
splitting wrongly: `1.7e8×` and `3.3e7×` tighter.

## The internal bound

`Proofs/Level5RiskDefs.lean` states what the claim is derived from; none of it needs to be read
to trust the claim.

    Level5RoundRisk: ∀ M (O : TriOracle μ M) θ (P : PrefixDistribution α) C J ū,
      0 ≤ ū → 32 ≤ εd · budget →
      ∫⁻ ω, P[the run of `applyProbe` over C.probeBudget probes from P.D does not end well, its
          harvest's sampler yielding harvestYield and a consistent end checking at most 3εd/2
          of the probes] ∂μ
        ≤ roundRisk σ O.κ O.ε θ P.D P.L C + 32ū/εd

The success test certifies the probes that search or could learn an unlearned edge, since such a
probe resets the hypothesis (`success_le`, through `site_le` charging only probes that keep it).
Where the probes reading a prefix's extension on a rarer side pass `εd/32 ≥ 1/budget`, the
`wrongTerm` below already exceeds one, so they cost nothing more; the undecided ones cost
`32 undecidedMass/εd` by Markov (`und_le_mass`), counted only where `undecidedMass ≤ ū`.

A read-state `q` is good (`O.Good θ q`) where it reads undecided at most `1.5θ` of the time.
`roundRisk σ κ ε θ D L C` is the least, over the proof's free parameters `kn : Knobs` meeting
`kn.Admissible σ κ ε θ D L C`, of `roundBound σ κ ε θ D L C kn`, the sum of:

* `edgeTail`, an edge's good undecided first reads past `θg` of its first reads:
  `K · E · (n + 2)|Σ| · exp(−φe(θe − 8θg)/4 / (2p₀(L + 1)(1 + 6θ(n + 1))))`;
* `startTail`, the good undecided start reads past `θgs`: `K · exp(θ₁(e^{l p₀} − 1)/p₀ − lθgs)`;
* `middlesTail`, searches stopping at a good undecided middle past `θgpt`:
  `K · E · exp(−θgpt φpt/2 / (2p₀(1 + 4(L + 1)θ₁)))`;
* `wrongTerm`, some step reading a string on its state's rarer side, a step reading at most
  `(L + 1) Lmax` prefixes followed by midfixes and re-sifting at most `Lmax² |Σ|` strings:
  `(ε + κ) · budget · ((L + 1) Lmax + Lmax² |Σ|)`;
* `testLevels`, `versionBound · (2 Lmax |Σ| + 5) · a`: at each of fewer than `versionBound`
  hypotheses, the start's, the middles', their gate's and success's tests, two potentials at
  each edge out of a leaf, and the success test passing with the rarer probes heavy each err at
  most `a`,

where here `K = classCount` bounds the class of trees (genuine splits only), `E = edgeMapCount`
their edge maps, `θ₁ = 1.5θ(n + 1)`, and `p₀ = prefixMax D k`, the largest mass of a length-`k`
prefix. `Level5RiskBound` (same file) is `roundRisk ≤ δ` under explicit conditions on `κ`, `ε`,
`θ` and the settings; `Level5Correct` substitutes its chosen settings into those conditions.

## Open

- **The shipped Python uses fixed settings**, not the ones `Level5Correct` chooses.

- **The middles field is per search.** A search makes up to `(L + 1)(n + 1)` first reads, and in a
  tree of depth `n + 1` at good read-states it stops undecided with chance near `1 − e^{−1.5θ R}`
  for `R` its reads, so the bound is the worst case, not slack. `pt_mean_le` unions over every
  position a search could stop at; the bisection, which stops at its first undecided midpoint,
  sifts at most `⌈log₂ L⌉` of them, which would raise `θ`'s cap by `(L + 1)/⌈log₂ L⌉` (6× at
  `L = 30`, 10× at `L = 60`), but needs an adaptive, first-read count of the middles like
  `edge_first_mean`'s.
- **`prefixMax` and `k`.** Every field concentrates over the probes' length-`k` prefixes, with a
  union over the class and its edge maps. The planned replacement (a): the tests count each string
  once, at its first read in the round, and a harvest is judged over those new strings; each new
  string's read is then fresh when charged, so the good read-states' undecided new strings at a
  place are at most binomial in the new strings read there, with no noise field, no union and no
  `prefixMax`.
- **Reads on the rarer side.** A split is genuine unless a read on the rarer side made the
  contradiction or the edge it contradicts untrue, and the re-sifts a split makes read strings no
  probe has to, so the bound unions over every word of at most `Lmax` letters after every prefix of
  every probe. A first-read count of the reads the round actually makes would replace
  `(Lmax + 1)|Σ|^Lmax` by the reads a probe makes.
- Each one-round theorem is per round, with the family, the seed and the band fixed, and treats the
  noise as fresh; the loop's theorem reads one noise across rounds (above).
- **The model's loop differs from the Python's.** Level 5 draws its probes avoiding the first `k`
  letters of the prefixes the round's clustering read, where the Python draws them from `D`. Each
  round's clustering draws its prefixes, certification prefixes and suffixes off every one the
  earlier rounds' clusterings read, where the Python does not avoid them. The uniform pool is drawn
  afresh each round, where the Python keeps one pool across rounds.

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
- `OrthoDFA/AdaptiveReads.lean` — `AdaptiveReadsBound`: reads drawn once per string,
  independently across strings, each bad with chance at most `ε`, and queried in rounds, each
  round choosing its strings from what earlier rounds read; some query of the first `n` rounds
  reads bad with chance at most `ε` times the most strings they query while none has.
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
- `OrthoDFA/Level5.lean`, `OrthoDFA/Level5Claim.lean` — level 5, the counterexample search
  (`indecisiveLStarLevel5`, returning the hypothesis and the ending), and `Level5Correct`.
  `Proofs/Level5RiskDefs.lean` holds the internal bound (`Level5RoundRisk`, `Level5RiskBound`),
  and `Proofs/Level5Correct.lean` derives the claim from it at the chosen settings. Probes walk
  from `k` as `sifting.read` does, and a disagreement is searched by main's bisection without
  recovery (`bisectBoundary false`), which stops at its first undecided midpoint. Every count is
  kept since the hypothesis last changed: the probes, the start's undecided read strings, each
  edge's first reads and those of them read undecided, the searches and the undecided middles they
  stop at. The start's undecided rate above `θs`, or an edge's undecided first reads exceeding `θe`
  of its first reads by `exc` of them, harvests; the searches' rate of stopping at an undecided
  middle above `θpt` harvests those middles; the searches' rate settling below `εd` succeeds. The
  start's, the middles' and success's rates are checked by `seqRate`, against `θs₁`, `θpt₁` and
  `εd₁`. Otherwise a contradiction (a clean disagreement at a learned edge) splits the edge's leaf
  at once (Kearns–Vazirani), and an unlearned edge a probe stops at is learned; either is a new
  hypothesis.
  - `Proofs/Level5Steps.lean`: a step keeps the hypothesis, its counts the probe's, or moves to a
    fresh one with the next version; each learned edge lowers the unlearned edges out of the
    leaves and each split adds a leaf, so a round goes through fewer than `versionBound`
    hypotheses (`version_lt`); within one every search but the last stops at an undecided middle,
    so past `nEnd` probes success or the middles test fires, and a round over
    `versionBound · nEnd` probes ends (`runEnds_of_budget`).
  - `Proofs/Level5Clean.lean`: a step reads its probe's prefixes followed by the tree's midfixes
    and, where it splits, the learned edges' witness successors followed by the new tree's
    midfixes; with none of those read on the rarer side (`WrongStep`), every learned edge and every
    contradiction is a true record, judged by the strings' states, so every split is genuine and
    the tree stays within `|Q| + 2` leaves (`truthInv_step`).
  - `Proofs/Level5Reads.lean`: given the probes, the round's reads come in rounds, each fixed by
    what earlier ones read while nothing has been read on the rarer side, a probe's at most
    `(L + 1) Lmax` and a split's at most `Lmax² |Σ|`; by `AdaptiveReadsBound` some step reads a
    string on the rarer side with chance at most the probes times
    `((L + 1) Lmax + Lmax² |Σ|)(ε + κ)` (`pi_wrong_le`).
  - `Proofs/Level5Harvest.lean`: within a hypothesis its probes are fresh draws, so each test's
    weight, its likelihood ratio or an edge's undecided strings less twice their running mean
    (and its reads' half-mean less its reads) exponentiated, is a product of factors averaging at
    most one where the hypothesis's sampler is not good; Ville's inequality bounds its passing
    `1/a` within the hypothesis by `a`, and a round ending badly passed one (`site_le`,
    `ratio_site_le`).
  - `Proofs/Level5Risk.lean`: on the noise event, a clean round ends by a test firing at a true
    hypothesis, and where that is not well one of the tests erred (`bad_of_clean`), which gives
    `Level5RoundRisk`.
  `Proofs/QueryTree.lean`, `Proofs/Level5Query.lean` and `Proofs/Level5Firsts.lean` give the
  fields' per-probe means: a probe is a query tree over independent reads (`run_probeQuery`), and
  a string's first read in it is fresh, so an edge's good undecided first reads average at most
  `1.5θ` of its first reads (`edge_first_mean`).
- `OrthoDFA/SuffixFamily.lean`, `OrthoDFA/Level5Family.lean` — `SuffixFamily`, its reads and
  their distributions (`readDist`), `SuffixFamily.oracle`, and `Level5FamilyCorrect`. Proved in
  `Proofs/Level5Correct.lean` from `Level5Correct`'s proof, the family's reads being independent
  with those distributions (`Proofs/SuffixFamily.lean`, from `FamilyReadBound`).
- `Proofs/Level5RiskBound.lean` — `Level5RiskBound`: explicit knobs, at which success fires at
  `nEnd` probes with at most `hS` searches and the middles test at `hS + 1` middles in a row, and
  `unionLog` bounding the unions' counts.
- `Proofs/SeqRate.lean` — `seqRate` in log form, its monotonicity in the hits, and counts at which
  it must fire (`seqRate_share`).
- `Proofs/Level5Event.lean` — `noiseEvent_le`, which unions the noise fields' tails:
  `goodEdge_le` (`Proofs/Level5EdgeRead.lean`), `goodStart_le` (`Proofs/Level5Noise.lean`) and
  `goodPT_le` (`Proofs/Level5PTNoise.lean`), each by `block_tail` (`Proofs/Level5EdgeNoise.lean`):
  per-probe quantities read only through strings sharing the probe's length-`k` prefix are
  independent across prefixes.
- `OrthoDFA/Stage.lean`, `StartAtK.lean`, `Loop.lean`, `Pass.lean`, `Automaton.lean` — the cut's
  sift down the shared tree (`siftCut`), curried edges (`EdgeTable`), walks and bisection, which
  level 5's statements build on.
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
