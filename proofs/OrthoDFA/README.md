# Machine-checked correctness of the E-L\* clustering algorithm

## The theorems

Every theorem below is proved in Lean with no `sorry`, using only `propext`, `Classical.choice`
and `Quot.sound` (`OrthoDFA/Verify.lean` prints this).

The headline is `LearnerLoopUniform` (`OrthoDFA/LearnerLoop.lean`, proved in
`Proofs/LoopUniform.lean`). Take probes uniform over the strings of length `L` and suffixes
uniform over those of length `L + |Q| + 3`, of which at least `pAP` keep every string's
membership. Then if `L ≥ L₀`, the learner's loop, walking from `L − 1`, stops within `|Q| + 1`
rounds at a round whose hypothesis's walk and tree part on at most `εd` of the probes and whose
read is wrong at most `εcov` and undecided at most `εcov` of the time, with probability at least
`1 − δ`, at most `Kp · G` probes a round. The whole run reads one noise, fixed per string as the
Python's is. With `s = 1/2 − η₀`, `δ' = δ/(3(|Q|+1))`, `ρ = εd/(256(|Q|+1)(|Q||Σ|+1))`:

    Θ  = 1/s + 8/εcov + 3072 |Q|(|Q|+1)
    P₀ = 128 (2Θ + 8(|Q|+1)/ρ)/(s² pAP) + 16 log((|Q|+4)/(δ'/2))/pAP²
    R₀ = 6000 (|Q|+2)² Θ³ log((|Q|+4) P₀ Θ/(δ' pAP/8))/s⁴
    L₀ = 2 + 8 (log(512 (|Q|+5)³ (R₀² + P₀² + 1)/(δ' ρ)) + 28)/(7 log |Σ|)

Unfolded, `7/8 · L₀ log |Σ| ≈ 6 log Θ + 8 log(1/s) + log(1/δ) + log(1/εd) + 2 log log(…) +
O(log |Q| + log |Σ|)`, logarithmic in `1/δ`, `1/εd`, `1/εcov`, `1/s`, `|Q|` and `1/pAP`.
At `|Q| = 3`, `|Σ| = 2`, `η₀ = εcov = εd = 0.1`, `δ = 0.01`, `pAP = 1/3`, `L₀ = 267`; the least `L`
at which `LearnerLoopSimple`'s hypotheses hold there, found numerically, is 241.

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
- `IdealRoundCorrect`: with every read fixed by its state, one L\* round ends well but for
  chance `(P + 1)(1 − ε)^n`.
- `Level4Correct`: with a `TriOracle` reading every string independently with its state's
  distribution, settings chosen by the proof make `indecisiveLStarLevel4`, the counterexample
  search, over at most `K · G` probes from a `PrefixDistribution` fail to end well with chance at
  most `δ`, once the reads' `κ` and `ε`, the goodness level `θ` and the probes' prefix mass are
  small enough. A harvest's sampler yields at least `ρ`.
- `Level4FamilyCorrect`: the same, read by a `SuffixFamily` (a DFA target, a suffix-free family
  and a band passing `BandPasses`), at `fam.oracle`.
- `LearnerLoopCorrect`: the learner's loop, clustering a family and running level 4 with it each
  round, stops within `|Q| + 1` rounds at a round whose hypothesis and tree agree on all but `εd`
  of the probes and whose read is rarely wrong, but for chance `(|Q| + 1)(2δc + δ)` (below).
- `LearnerLoopSimple`: the general form of the headline, at settings the theorem chooses, every
  hypothesis an explicit inequality on any probes and suffixes and any walk start `k ≥ 1`: with
  probability at least `1 − δ`, within `|Q| + 1` rounds the loop stops at a round whose hypothesis
  and tree part on at most `εd` of the probes and whose read is wrong at most `εcov` of the time
  and undecided at most `εcov`.
- `LearnerLoopUniform`: the headline above.

## The loop

`LearnerLoopCorrect` (`OrthoDFA/LearnerLoop.lean`): `indecisiveLStarLoop` clusters a family over
the populations so far, runs level 4 with it, and adds a harvest's sampler, scaled to a
probability, as a population of every later round. The whole run, every round's clustering and
level 4, reads one noise. Level 4 runs at disagreement rate `εd / 2` and draws its probes from `D`
conditioned on their first `k ≥ 1` letters avoiding those of every prefix the round's clustering
read. Within `|Q| + 1` rounds the loop stops at a round whose hypothesis's walk and tree part on at
most `εd` of the probes and whose read is decided wrongly at most `εcov + 2·indecisionLimit +
2·slack` of the time, but for chance `(|Q| + 1)(2δc + δ)`, for probes whose length-`k` prefix mass
is at most `cap · ρ / 4` and `ρ / (24 unionLog)`. A harvest's sampler yields at least `ρ / 4` and
finds a string at most as often as the probes share its length-`k` prefix, so a population puts at
most `4 prefixMax / ρ` on a string, within the clustering's collision cap `cap`; `cap` is also small
against the schedule's suffix pool `NS` and prefix count `NP`, and bounds the suffixes' atoms.

`LearnerLoopSimple` fixes everything the theorem may choose: `indecisionLimit = slack = ι` with
`ι = min(εcov/8, 1/(3072 (L+1)(|Q|+1)|Q|))`, the FNR gate at `1/4`, half of `εcov` for the
clustering, no splits that are not genuine, `|Q| + 1` rounds, suffixes of length `L + |Q| + 3`, and
`δ/(3(|Q| + 1))` for each round's clustering, collisions and level 4. Its cap is a closed form under
the clustering's (`capFloor`, `Proofs/LoopSimple.lean`), from `poolCount_le` and
`prefCount_le_poly`, and under `1/(3J(RB+1)(L+1))` and `δ'/(J(PB² + 16J²RB² + 1))` with `PB`, `RB`
bounding the pool and the prefix count. `LearnerLoopUniform` takes probes and suffixes uniform at
lengths `L` and `L + |Q| + 3` and `k = L − 1`, so `prefixMax = |Σ|^-k`. `cap_bound`
(`Proofs/LoopUniform.lean`) shows the cap is its first term, `δ'/(128(|Q|+5)³(RB² + PB² + 1))`,
and bounds the band's far side by `1/(Θ(L+1))`, `PB` by `(L+1) P₀` and `RB²` by `(L+1)⁷ R₀²`, so
`4/(cap ρ) ≤ (L+1)⁷ e^Λ` with `Λ = log(512(|Q|+5)³(R₀² + P₀² + 1)/(δ' ρ))` free of `L`;
`7 log(L+1) ≤ 28 + (L+1) log |Σ|/8` absorbs the `L`. The proved `L₀` is within 1.11–1.30× of the
least `L` at which `LearnerLoopSimple`'s hypotheses hold (241 at the point above, 59 at
`|Σ| = 16`, 300 at `εcov = 1e-9`): the absorption costs `8/7`, and at small `εcov` bounding `1/τ`
by the sum `Θ(L+1)` costs the rest. One noise costs about ten letters at `|Σ| = 2` over fresh
noise per round (257 then).

Proved with no `sorry`. The rounds' draws are independent, and given the earlier rounds a round's
clustering run is drawn as `runMeasure` draws it at their populations (`Proofs/LoopLaw.lean`,
`Proofs/LoopIndep.lean`): a population's draws are made from uniforms (`drawFrom`), and a round's
population is fixed by finite data, its hypotheses' trees and edge graphs and its reads of short
strings (`Proofs/LoopCodes.lean`). A round's outcomes are measurable because, at fixed draws, each
is fixed by the oracle's answers at finitely many strings (`Proofs/LoopMeasurable.lean`), level 4
reading only strings at most `L + 1 + maxLeaves` long (`Proofs/Level4Reads.lean`).

On one noise, what the earlier rounds hand on is fixed by the answers at the strings they read, and
a round's bad event by the answers at the strings it reads; off a collision the two sets are
disjoint, so the two are independent (`persist_stage_le`, `Proofs/LoopPersist.lean`). A round's
suffixes are `Ls` long and its prefixes shorter, so its clustering reads nothing an earlier round
read unless it repeats a suffix or a prefix (`Proofs/LoopCollide.lean`), which has chance at most
the pairs times the largest atom (`loop_coll_le`, `Proofs/LoopColl.lean`). Level 4 reads `z·v`
with `v` a family suffix and `z`'s first `k` letters a probe's, so its avoiding probes never read a
string the clustering read (`level4_congr_off`, `Proofs/Level4Avoid.lean`). Re-noising the
clustering's strings with a second, independent noise (`Oracle.renoise`, `Proofs/Renoise.lean`)
leaves a fresh family on which level 4's guarantee holds at the conditioned probes, whose prefix
mass grows by at most `1/(1 − m)` for `m ≤ |keys| · prefixMax` the mass removed
(`Proofs/LoopAvoid.lean`); the actual run is the same run, and its ending transfers back to `D`
at disagreement `ε + m`, yield `(1 − m) y` and a quarter of the sampler bad
(`quarterEndsWell_of_endsWell`, `Proofs/LoopTransfer.lean`), which still leaves a bad state with
`1/(4|Q|)` of the population (`Proofs/LoopTermination.lean`).

## The round, by refinement

Each level proves one L\* round correct and states its ending through `OrthoDFA/RoundEnd.lean`:
the round ends well when it is consistent, the hypothesis and the tree parting on at most `ε` of
the probes, or ends in a harvest whose sampler finds more than half of what it finds at bad
states, is spread and yields. What makes a state bad is each level's own.

Level 4's harvest is its site and the hypotheses the site's counts were taken under: one per
probe of the round at the start and at an edge, the stretch's own for the middles. Its sampler
(`harvestLaw`) picks one of those hypotheses uniformly, draws a fresh probe from `D`, walks it
under that hypothesis and finds the probe's undecided first reads at the site, as the tests
counted them. The tests are cumulative over the round, so this mixture is what they certify:
every good read-state's undecided reads are rare under each hypothesis, whatever came before, and
a count less twice the running sum of its means is a supermartingale.
- `Spread`: it finds no string more often than draws share the string's length-`k` prefix, since
  every string a probe walked from `k` reads extends the probe's length-`k` prefix and a probe
  reads no string first twice (`Proofs/Level4Law.lean`).
- `Yields y`: it finds at least `y`, `harvestYield`, the least of `θs/8`, `θe φe/2` and
  `θpt φpt/2`, which the start's, an edge's and the middles' tests certify.

Level 2's harvest is still a list, replayed on its final hypothesis from an anchor drawn uniformly
from `start, …, L` (`ReplayEndsWell`); moving it, and levels 3 and 3.5, to the sampler is a
follow-up.

1. #417, `IdealRoundCorrect`: ideal reads; a state is bad where all its strings read undecided.
2. #419: indecision random, decided reads always right.
3. #420: rare wrong reads at every state.
4. This, `Level4Correct` and `Level4FamilyCorrect`: level 4, the counterexample search as
   `sifting.read` runs it (`indecisiveLStarLevel4`), read by the suffix family; a state is bad
   where it reads undecided more than `1.5θ` of the time.

## The statements

`OrthoDFA/Level4Claim.lean` and `OrthoDFA/Level4Family.lean` hold them, with everything they
need to be read; the algorithm is `indecisiveLStarLevel4` in `OrthoDFA/Level4.lean`. With
`n = |Q| + S`, `A = |Σ|`, `L = P.L`, `Z = |Q| A + S + 1`, `ρ = (S + 1) εd / (256 (n + 1) Z)`,
`K = (n + 1)(2(n + 3)A + 1)(10/εd³ + 270/εd² + 1400/εd) + 8(n + 3) A (L + 1)`,
`X = 4 (2^(n+5) A + 7K²)/δ`, `G = log X + 2 log(2 log X)` and a read-state bad where it reads
undecided more than `1.5θ` of the time:

    Level4Correct: ∀ M (O : TriOracle μ M) θ εd (P : PrefixDistribution α) δ S k,
      0 < δ ≤ 1 → 0 < εd ≤ 1 → 1 ≤ L → 0 ≤ θ →
      384 (L + 1)(n + 1) θ ≤ 1 →
      0 < O.κ → 4 O.κ (8 (n + 3)(L + 1) √(A ρ / 3) + 1) ≤ ρ →
      0 ≤ O.ε → (L + 1)(n + 1) O.ε + 3 · unionLog σ S δ · prefixMax P.D k ≤ ρ →
      ∃ settings : Level4Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
        settings.probeBudget ≤ K · G ∧
        P[let result := indecisiveLStarLevel4 settings (O.read · ω) draws
          ¬ EndsWell M bad P.D settings.disagreementRate
            (searchingProbes (O.read · ω) settings.walkStart result.hypothesis)
            settings.walkStart (harvestLaw P.D (O.read · ω).cut settings.walkStart) ρ
            result.ending] ≤ δ

    Level4FamilyCorrect: Level4Guarantee at fam.oracle : TriOracle μ fam.M.

The settings (`Level4Settings`) are written below by their symbols: `Lmax = maxLeaves`,
`m = recordsToMove`, `θs = startUndecidedRate`, `θe = edgeUndecidedRate`,
`θpt = middleUndecidedRate`, `εd = disagreementRate`, `φe = edgeTraffic`, `φpt = searchShare`,
`a = testLevel`, `n₀ = firstLook`, `exc = edgeExcess`, `k = walkStart` and
`budget = probeBudget`.

`16ρ` is about the share of the probes at which the budget affords a record that is not true. The
proof (`Proofs/Level4Correct.lean`) chooses, with `N = n + 3`, `L₁ = L + 1` and `s = √(A ρ/3)`:
`Lmax = N`, `θpt = 1/4`, `θs = 1/2`, `φpt = 3εd/4`, `θe = 8s + 1/(L₁ N)`, `φe = 4s L₁/A`,
`a = e^{−G}`, `n₀ = ⌈(512 + 32/(9εd²)) G⌉`, `exc = 6 L₁ G`, `m` the records the caps ask for and
the budget `Level4RiskBound` asks for. `θe φe ≥ 32 L₁ ρ/3` holds the edge field's prefix mass to the
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
left.

## The internal bound

`Proofs/Level4RiskDefs.lean` states what the claim is derived from; none of it needs to be read
to trust the claim.

    Level4RoundRisk: ∀ M (O : TriOracle μ M) θ (P : PrefixDistribution α) C κs,
      2 · O.κ ≤ κs →
      ∫⁻ ω, P[the run of `applyProbe` over C.probeBudget probes from P.D does not end well, its
          harvest's sampler yielding harvestYield] ∂μ
        ≤ roundRisk σ O.κ O.ε θ P.D P.L C κs

A read-state `q` is good (`O.Good θ q`) where it reads undecided at most `1.5θ` of the time.
`roundRisk σ κ ε θ D L C κs` is the least, over the proof's free parameters `kn : Knobs` meeting
`kn.Admissible σ κ ε θ D L C κs`, of `roundBound σ κ ε θ D L C kn`, the sum of:

* `recordsTail`, records that are not true exceeding `κs` times the undecided strings plus `ρ`:
  `K · E · exp(−(ρ − 2ε(L + 1)(n + 1)) / (2p₀(1 + 4(κ(L + 1) + ε(L + 1)(n + 1)))))`;
* `edgeTail`, an edge's good undecided first reads past `θg` of its first reads:
  `K · E · (n + 2)|Σ| · exp(−φe(θe − 8θg)/4 / (2p₀(L + 1)(1 + 6θ(n + 1))))`;
* `startTail`, the good undecided start reads past `θgs`: `K · exp(θ₁(e^{l p₀} − 1)/p₀ − lθgs)`;
* `middlesTail`, searches stopping at a good undecided middle past `θgpt`:
  `K · E · exp(−θgpt φpt/2 / (2p₀(1 + 4(L + 1)θ₁)))`;
* `fakeRun`, the records that are not true racing the tests' allowances;
* `(S + |Q| + 1) · subOpen`, a sub-round keeping its tree too long;
* `testLevels`, `(2^(Lmax+2)|Σ| + 1 + 3 budget²) a`, the harvests' and success's tests,

where here `K = classCount` bounds the class of trees, `E = edgeMapCount` their edge maps,
`θ₁ = 1.5θ(n + 1)`, and `p₀ = prefixMax D k`, the largest mass of a length-`k` prefix.
`Level4RiskBound` (same file) is `roundRisk ≤ δ` under explicit conditions on `κ`, `ε`, `θ` and the
settings; `Level4Correct` substitutes its chosen settings into those conditions. At
`|Q| = 10`, `|Σ| = 2`, `L = 30`, `δ = 0.01` and `εd = 1/4`, the chosen settings admit
`κ ≤ 2.3e-7`, `θ ≤ 6.5e-6`, `ε ≤ 2.4e-8` and `p₀ ≤ 3.6e-9` at a budget of `3.9e8`, against
`κ ≤ 5e-8`, `θ ≤ 6e-6`, `ε ≤ 1e-8` and `p₀ = 2^-30` at `3.5e8` for the earlier hand-picked
settings.

## Open

- **The shipped Python uses fixed settings**, not the ones `Level4Correct` chooses.

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
- Each one-round theorem is per round, with the family, the seed and the band fixed, and treats the
  noise as fresh; the loop's theorem reads one noise across rounds (above).
- **The model's loop differs from the Python's.** Level 4 draws its probes avoiding the first `k`
  letters of the prefixes the round's clustering read, where the Python draws them from `D`. The
  uniform pool is drawn afresh each round, where the Python keeps one pool across rounds.

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
- `OrthoDFA/ProbeDistribution.lean` — how probes are drawn (`PrefixDistribution`), where level 2's
  replay of one is anchored (`uniformAnchor`), and the share of draws visiting given states
  (`coverage`).
- `OrthoDFA/RoundEnd.lean` — what it is for a round of the learner to end well, shared by every
  level: consistent, or a harvest whose sampler finds mostly bad states, is spread and yields.
- `OrthoDFA/Hypothesis.lean` — the learner's hypothesis: a tree sifting strings to its leaves, and
  edges between the leaves.
- `OrthoDFA/Level2Probe.lean` and `OrthoDFA/Level2.lean` — level 2 of indecisive L\*,
  `indecisiveLStarLevel2`. Its probe walks to where the walk stops, sifts the successor of an
  unlearned edge it stops at, and finds where the sift and the walk part with `tripartiteSearch`.
- `OrthoDFA/Level2Claim.lean` — `IdealRoundCorrect`: with every state's reads all undecided or
  all on its side, level 2 on `P` probes ends well but for chance at most `(P + 1)(1 − ε)^n`, `n`
  the clean probes in a row that end it.
- `OrthoDFA/Level4.lean`, `OrthoDFA/Level4Claim.lean` — level 4, the counterexample search
  (`indecisiveLStarLevel4`, returning the hypothesis and the ending), and `Level4Correct`.
  `Proofs/Level4RiskDefs.lean` holds the internal bound (`Level4RoundRisk`, `Level4RiskBound`),
  and `Proofs/Level4Correct.lean` derives the claim from it at the chosen settings.
  `Proofs/Level4Parts.lean` states its parts, as sub-rounds. Probes walk from `k` as `sifting.read`
  does; a clean disagreement records its prefix and target at its edge. Each place keeps its count
  over the round: the start's undecided read strings, and each edge's first reads and those of them
  read undecided; the start's undecided rate above `θs`, or an edge's undecided first reads
  exceeding `θe` of its first reads by `exc` of them, harvests; the stretch's rate of searches
  stopping at an undecided middle above `θpt` harvests those middles; the stretch's disagreement
  rate settling below `εd` succeeds. Otherwise one edge with `m` records at a target it does not
  point at is redirected there, or its leaf splits where its own target also has `m`; a split drops
  every record. A stretch starts afresh at every change of hypothesis. Each state reads its rarer
  decided side at most `κ` times as often as it is undecided. A sub-round runs at one tree of the
  class (genuine splits and at most `S` that are not) until a split, genuine (real) or not (fake),
  or the round ends. A split that is not genuine needs `m` records that are not true made in its
  sub-round, so while the round's such records stay below `(S + 1) m` its trees stay in the class.
  `SubRound` bounds a sub-round's chance of keeping its tree over `subT` probes with fewer than `B`
  records that are not true; `FakeRace` bounds the round's records that are not true reaching `(S +
  1) m`, given `NoiseEvent`; `HarvestGood` bounds a harvest whose sampler does not find mostly
  read-states that are not good, or does not yield. Proved in `Proofs/Level4Budget.lean`: `round_budget` composes sub-rounds, and
  `roundGiven_of` gives `Level4RoundRisk` from `SubRound`, `FakeRace`, `HarvestGood` and `SuccessSound`.
  `HarvestGood` is proved in `Proofs/Level4Harvest.lean`: the start's and an edge's undecided
  strings less twice the running sum of their means under the hypotheses the round kept, and an
  edge's reads' means' half-sum less its reads, exponentiate to supermartingales while the tree is
  in the class, so a test that fires certifies the sampler's yield; `NoiseEvent` holds the good
  read-states' share of it under every hypothesis of the class. A stretch keeps one hypothesis,
  and where its searches stop at an undecided middle at most `θpt` of the time, the middles' count
  over them is at most binomial.
  `SubRound` is proved in `Proofs/Level4Keep.lean`: a hypothesis recording less than `θr` of the time
  and keeping over `nEnd` probes has its middle-stopping searches rarer than `θpt'` of its searches,
  or its disagreements too many for success; each new hypothesis learns or redirects an edge,
  lowering a potential of at most `2 Lmax |Σ|`; and every record raises its edge and target towards
  the `m` at which it is fixed, a true one at one of `|Q| |Σ|` edges and targets, so with fewer than
  `B` that are not true the sub-round holds at most `|Q| |Σ| m + B - 1` records and its probes at
  hypotheses recording at least `θr` of the time are one binomial over `nRec`. `FakeRace` is proved
  in `Proofs/Level4Race.lean`: `NoiseEvent.spurious` bounds a probe's record not being true by `κ` times
  its undecided strings plus `ρ`, the edges' and the start's tests hold those strings over the round
  to `θe` of an edge's first reads past `Xe` once the edge is charged `φe` of the probes (an edge
  charged less holds them to that) and `θs'` of the probes past `Xs`, so the records that are not
  true less `ν/η` times the strings' excess exponentiate to a supermartingale up to `ν (θe (L + 1)
  Lmax + θs' + 2 Lmax |Σ| φe) + (e^η - 1) ρ` a probe. `Proofs/QueryTree.lean`,
  `Proofs/Level4Query.lean` and `Proofs/Level4Firsts.lean` give the fields' per-probe means: a probe
  is a query tree over independent reads (`run_probeQuery`), and a string's first read in it is fresh,
  so a record not being true, which needs a read on its state's rarer side (`untrue_minority`), has
  chance at most `κ` times the probe's undecided first reads, which the tests count (`und_twins`),
  plus `ε` times its first reads (`spurious_mean`); and an edge's good undecided first reads average
  at most `1.5θ` of its first reads (`edge_first_mean`).
- `OrthoDFA/SuffixFamily.lean`, `OrthoDFA/Level4Family.lean` — `SuffixFamily`, its reads and
  their distributions (`readDist`), `SuffixFamily.oracle`, and `Level4FamilyCorrect`. Proved in
  `Proofs/Level4Correct.lean` from `Level4Correct`'s proof, the family's reads being independent
  with those distributions (`Proofs/SuffixFamily.lean`, from `FamilyReadBound`).
- `Proofs/Level4RiskBound.lean` — `Level4RiskBound`: explicit knobs, Hoeffding's lower binomial tail
  for the termination tests, and `unionLog` bounding the unions' counts.
- `Proofs/Level4Risk.lean` — `Level4RoundRisk` at each admissible choice of knobs: `RoundGiven`
  (`Proofs/Level4Parts.lean`, proved in `Proofs/Level4Budget.lean`) with its noise event's chance
  bounded by `noiseEvent_le` (`Proofs/NoiseEventvent.lean`), which unions the fields' tails:
  `spurious_le` (`Proofs/Level4SpurNoise.lean`), `goodEdge_le` (`Proofs/NoiseEventdgeRead.lean`),
  `goodStart_le` (`Proofs/Level4Noise.lean`) and `goodPT_le` (`Proofs/Level4PTNoise.lean`), each by
  `block_tail` (`Proofs/NoiseEventdgeNoise.lean`): per-probe quantities read only through strings
  sharing the probe's length-`k` prefix are independent across prefixes.
- `OrthoDFA/Stage.lean`, `StartAtK.lean`, `Loop.lean`, `Pass.lean`, `Automaton.lean` — the cut's
  sift down the shared tree (`siftCut`), curried edges (`EdgeTable`), walks and bisection, which
  level 4's statements build on.
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
