# Machine-checked correctness of the E-L\* clustering algorithm

Every theorem below is proved in Lean with no `sorry`, using only `propext`, `Classical.choice`
and `Quot.sound` (`OrthoDFA/Verify.lean` prints this).

## The claims

`OrthoDFA/Claims/` states what each part of the learner costs, as big-O shapes in its inputs
alone. Each claim's inputs are a record (`ClusterInputs`, …) of counts and rates, and the claim
holds for every record in a region `R` that spells out the hypotheses, such as `0 < δ ≤ 1/2`. Every
cost and every tolerance is something the claim produces: it asserts some function of the inputs
does the job, and bounds that function with Mathlib's `=O[𝓟 R]`, one constant for all of `R`, so
the bound keeps its meaning with some inputs held fixed. A cap on an input, such as how heavy a
population's atoms may be, appears as the largest the claim tolerates, its reciprocal bounded.
Each claim is proved from the explicit theorems below.

- `ClusteringClaim` (`Claims/Clustering.lean`): over `ClusterInputs`, with `s₀ = 1/2 − η₀` the
  signal, `ε_s` the redraw and hash slack, `ε_u` the indecision limit and `ε_w` the band's cross
  limit, and no strings harvested, the band width, suffix pool and prefix count of every state it
  stops at, and the atom and collision masses its populations must stay under; it fails with
  chance at most `δ`, and the family cuts all but `εcov + ε_s` of the uniform pool.
  `ClusteringClaimSimple` bounds each by a single product, e.g. the prefix count by
  `O(nJ² log(nJ/(δ pAP εcov ε_s ε_u ε_w s₀))/(εcov² ε_s ε_u² s₀⁸))`.
- `Level2Claim` (`Claims/Levels.lean`): over `Level2Inputs`, level 2's one-round probe budget,
  `O(|Q|²|Σ| ε/(ε − y)² · log(|Q||Σ|/δ))`.
- `Level3Claim`, `Level4Claim` and `Level5Claim` (`Claims/Levels.lean`): over `LevelInputs`, one
  round's probe budget at that level, `O(|Q|²|Σ|/ε · log(|Q||Σ|/δ))`, its harvest's slope
  `O(|Q|(1/y + log L))` and its slack `O((|Q||Σ| log(|Q||Σ|) + log(1/δ))/y)`. Levels 4 and 5 also
  give how much of the probes' law any one walk start may hold (`p₀Tol`, `1/p₀Tol` of order the
  budget squared over `δ`) and how rarely reads may be wrong (`1/εwTol =
  O(Q'³|Σ|/ε · log(|Q||Σ|/δ) log L)` at level 4, `1/κTol = O(Q'²|Σ| log(|Q||Σ|/δ))` at level 5),
  their trees growing `spurious = O(log(1/δ))` leaves past level 3's `|Q| + 2`. Their budget and
  slope are level 3's with `|Q|` outside the log read as `Q' = |Q| + log(1/δ)`, and their slack is
  `O(Q'|Σ| log(Q'|Σ|)/y)`.
- `LoopClaim` (`Claims/Loop.lean`): over an input distribution `D`, over
  `LoopDistInputs` (`LoopInputs` with `β`, `0 < β ≤ 1/|Σ|`) and `R ≥ 1` the visit ratio of the
  relative rule (`LearnerLoopDFARelative`'s premise). The probes are `D`'s strings cut at a
  length uniform over `k = ⌈L/2⌉` to `L₁ = L + |Q| + 3` (`windowLaw`), the suffixes their first
  `Ls = L₁ + |Q| + 3` letters, and the coverage, `rerootAccuracy`, error and held-out strings are
  over their first `L` (`prefixLaw`). `D`'s strings must be at least `Ls` long, must extend each
  prefix of `k` to `L` letters by each letter at least `β` of the time, and must hold at most
  `p₀Tol` on any first `k` letters, `1/p₀Tol = O((L + |Q|)⁶|Q|¹⁰(1/β + |Q|)⁴R⁴
  log²(|Q|R/(β δ ξ pAP s₀))/(δ ξ⁴ pAP³ s₀¹¹))`. The learner's DFA is then right but for `γ + ξ`,
  its start picked on `O(log(|Q|/δ)/ξ²)` held-out strings, and, over `LoopDistRun` (the inputs with
  `L`), every prefix, suffix and probe the whole run reads is bounded: `O(|Q|⁴(1/β + |Q|)²(L +
  |Q|)²R² log(X')/ξ² · (1/s₀⁸ + |Q|⁴/s₀⁴))` prefixes, `O(|Q|(log(X')/(pAP s₀²) +
  log(|Q|/δ)/pAP²))` suffixes and `O(|Q|³|Σ|(1/β + |Q|)(L + |Q|)R/ξ · log(|Q||Σ|/δ))` probes,
  `X' = (L + |Q|)R/(β δ ξ pAP s₀)`. The totals grow with `L` because the settings the proof picks
  hold each round's disagreement rate at `m/(3(1/β + |Q| + 1)N)`, `m = 3ξ/(8R)`, over the window's
  `N = Θ(L)` lengths: the prefixes grow as `(L + |Q|)²` and the suffixes as `log L`.
- `LoopClaimUniform` (`Claims/Loop.lean`): `LoopClaim` at `D` uniform and `β = 1/|Σ|`
  (`loopClaimUniform_of_loopClaim`, `Proofs/ClaimLoopUniform.lean`), over `LoopInputs`. In place of
  `p₀Tol` it has the length `L₀` past which `|Σ|^-k ≤ p₀Tol`, `O(1 + log(|Q|R/(δ ξ pAP
  s₀))/log|Σ|)`, and its totals are over `LoopRun` (the inputs with `L`): `O(|Q|⁴(|Σ| + |Q|)²(L +
  |Q|)²R² log(X)/ξ² · (1/s₀⁸ + |Q|⁴/s₀⁴))` prefixes, `O(|Q|(log(X)/(pAP s₀²) +
  log(|Q|/δ)/pAP²))` suffixes and `O(|Q|³|Σ|(|Σ| + |Q|)(L + |Q|)R/ξ · log(|Q||Σ|/δ))` probes,
  `X = (L + |Q|)|Σ|R/(δ ξ pAP s₀)`. `LoopClaimSimple` bounds the prefixes by `O(|Q|⁸(|Σ| +
  |Q|)²(L + |Q|)²R² log(X)/(ξ² s₀⁸))` and the suffixes by `O(|Q| log(X)/(pAP² s₀²))`.

## The explicit theorems

The headline is `LearnerLoopDFA` (`OrthoDFA/LowLevelClaims/LoopDFA.lean`, proved in
`Proofs/LoopDFA.lean`): the loop outputs a DFA, and `LearnerLoopDFASimple` states the same at
explicit settings (below). The whole run reads one noise, fixed per string as the Python's is. Each
round runs level 3's algorithm (`indecisiveLStarLevel3`), guaranteed by `Level5RoundCorrect` at the
family's `κ`, reading the clustering's family. One-step consistency at `L − 1` is not enough for
this (`parity_toggle`, #438).

- `ClusteringGuarantee`: with probability at least `1 − δ` the clustering loop stops, and the
  family it returns cuts all but `εcov + εs` of the uniform pool as the noiseless oracle does, at a
  polynomial number of queries; `εs` is what the samplers' redraws and the hash's split may cost.
  The samplers draw without replacement, and a draw lands on what a
  state already holds at most `(|J| + 3)·npref` times the largest atom, so each sampler rejects
  more than `rejCount` of its draws only on a binomial tail (`SuccessiveDrawsTails`); a tame
  population need only keep every atom under `rateScale·(1/2 − η₀)/(2¹⁸(|J| + 3)·npref)` and
  `εs/(2⁸(|J| + 3)·npref)`, and under `δ·min(εs, indecisionLimit)²/(2¹⁵|J|(3 + 2⌊log₂ npref⌋))`
  for the hash to split it evenly.
- `ClusteringQualityGuarantee`: averaged over a population, the returned family decides a prefix
  wrongly, or leaves it undecided, only a bounded share of the time.
- `FamilyReadBound`: the suffix family's reads are independent across strings, every DFA state
  has one distribution its strings read with, and the band's check bounds each.
- `TripartiteSearchCorrect`: the search the probes use to find where the walk and the sift part
  finds what it reports, queries at most `2 + 3⌈log₂(hi − lo)⌉` positions (`2 + ⌈log₂(hi − lo)⌉`
  without recovery), and finds the turn exactly where nothing is undecided.
- `Level2RoundCorrect`, `Level3RoundCorrect`, `Level4RoundCorrect` and `Level5RoundCorrect`: one L\*
  round ends well but for chance `δ`, with ideal reads, random indecision, rare wrong reads, a few
  tolerated, and wrong reads scaling with indecision, a few tolerated.
- `FamilyKappaOracle` (`OrthoDFA/LowLevelClaims/FamilyOracle.lean`): the clustering's family reads
  are a `KappaReadOracle`, each state's side the one it reads more often, at the band's `κ`.
- `LearnerLoopDFA` and `LearnerLoopDFASimple`: the loop outputs a DFA. See below.

## The learner's DFA

The Python's `satisfies_preconditions` admits a target at its length `L` by the relative rule
(`LearnerLoopDFARelative`, below): some start `q` in its own `non_start_zone_states` zone re-roots it
well (`non_start_zone_accuracy_ceiling`). `LearnerLoopDFA` admits it by a covered rule instead:
`coveredStates M L c` are the states at least `c` of the strings of length `L` visit at some
position from `⌈L/2⌉` to `L`, and `rerootAccuracy M L S q` is the share of those strings whose run
from `q` stays in `S` and is classified as the target classifies them. Where some covered `q` has
accuracy at least
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
every hypothesis an inequality. Each of the four conditions on the walk start bounds the mass the
probes' law puts on one first `k` letters, `|Σ|^-k` for uniform strings, and they follow from
`(L₁ + 1)⁶ (E + E₂ + E₃) ≤ |Σ|^k`, `E`, `E₂` and `E₃` free of `L` and polynomial in `|Q|`, `|Σ|`,
`1/δ`, `1/c`, `1/ξ`, `1/(1/2 − η₀)` and `1/pAP`.
Bounding `log (L₁ + 1)` by its tangent at `T = 2(log (E + E₂ + E₃) + 12)/log |Σ| + |Q| + 4`, that
holds once `L ≥ L₀` (`dfa_pow`, `Proofs/LoopDFAArith.lean`), where

    L₀ = 2(log (E + E₂ + E₃) + 6 log T)/log |Σ| + 144(log T − 2)/(log |Σ| (T log |Σ| − 12)).

At `|Q| = 3`, `|Σ| = 2`, `η₀ = 0.1`, `c = δ = 0.01`, `pAP = 1/3`, `ξ = 0.1`, `L₀ ≈ 421.6`; the
least `L` with `(L₁ + 1)⁶ (E + E₂ + E₃) ≤ |Σ|^k` is 421, and the least meeting
`LearnerLoopDFASimple`'s hypotheses 371.

The proof runs over an input distribution `D` (`LearnerLoopDFADist`, `learner_loop_dfa_dist_core`,
proved in `Proofs/LoopDFA.lean`), from a last round's checks to a leaf that misjudges little taken
as a hypothesis: `covered_bridge` gives it under the covered rule and `summed_bridge` under the
relative one. The uniform theorems take `D` uniform over the strings of length `L + 2|Q| + 6`,
whose first `n` letters are uniform (`prefixLaw_uniformStrings`). Uniformity entered in three
places, each now a condition on `D`:

1. The walk start's prefix mass. Every collision the proof rules out, among the suffixes, the
   prefixes and the held-out strings, lies within one cylinder of first `k` letters, so `D`'s
   largest such mass `p₀` takes `|Σ|^-k`'s place in the four conditions. `LoopClaim` asks
   `p₀ ≤ p₀Tol = 1/((L + |Q| + 4)⁶ loopMassInv)` (`claim_mass_le`, `Proofs/ClaimLoopArith.lean`);
   for uniform strings `L ≥ L₀` gives it (`pow_of_tangent`).
2. Crossing every edge. A probe of length `i` and its extension by `a` both need mass, so
   `edge_of_occupied` asks `D` to extend each prefix by each letter at least `1/Ae` of the time
   (`prefixLaw_snoc_le`), and the edge-crossing rate is `c/((Ae+1)N)`. `Ae = 1/β` in general and
   `|Σ|` for uniform strings (`uniformStrings_floor`).
3. Coverage and `rerootAccuracy`. They are measured under `D`'s first `L` letters
   (`coveredStatesOn`, `rerootAccuracyOn`), the law the window's probes of length `L` follow; the
   occupancy a covered state gives the window is `occupancy_of_covered`.

`LearnerLoopDFARelative` (proved in `Proofs/LoopDFARelative.lean`) admits the target by the
relative rule: some `q ∈ nonStartZoneStates M L R q` has accuracy at least `1 − γ` with
`S = nonStartZoneStates M L R q`, `nonStartZoneStates` being the states whose visit rate from `q` is
at most `C = R(L+1)/(L+1−⌈L/2⌉) ≤ 2R` times that of a string from `⌈L/2⌉` on. No state need be
covered: the proof sums occupancy over the wrong states (`dfa_of_checks_summed`,
`Proofs/LoopDFASummed.lean`). A state is wrong when an edge out of its true leaf is wrong or
unlearned, or it is read on its wrong side. A string has one state, so summed over the wrong-edge
states `edge_of_occupied`'s count (below) charges the disagreeing probes `(Ae+1)N β` once and the
misled ones `N β'` per state (`wrongEdge_occupancy_le`), and the states read on their wrong side
hold at most `2εr + εu` of the window (`wrongSide_mass_le`). At `ε = m/(3(|Σ|+|Q|+1)N)`, `m = 3ξ/(8R)`, the wrong states occupy
the window less than `m`, so the runs from `q` through a wrong state of its zone cost at most
`Cm ≤ 3ξ/4`, and the pick the other `ξ/4`: `n` is `⌈512 log(16(|Q|+1)(|Q|+2)/δ)/ξ²⌉`. The probe
budget `B₃ ∝ 1/ε` grows as `R(|Σ| + |Q|)N/ξ` and `L₀` as its logarithm.

The probes over a window of lengths are the algorithm's change. Every prefix length is itself a
probe, so a probe and its extension by a letter that both agree, read nothing on a rarer side
past a midfix and nothing undecided at their sifts, cross a learned edge between their states'
true leaves (`edge_of_pair`, `Proofs/DFABridge.lean`). Call a probe misled at `a` when it, or its
extension by `a`, reads a string followed by a midfix of the tree on its state's rarer side
(`misledSet`). A covered state is visited `X ≥ c` of the time summed over the window, so an edge
out of its true leaf that is wrong or unlearned is crossed at each of its probes by a pair that
disagrees or is misled: `disagreeSet`, the probes that search, cross a wrong edge, stop at an
unlearned one or stop undecided at their start or end sift, holds `β` of the probes (level 3's own
success test bounds it at `ε`), and the misled ones are at most half of the state's probes shorter
than `L + 1` plus `β'` (`FewMisled`), so

    X ≤ (|Σ|+1)N β + X/2 + N β'

(`edge_of_occupied`). A misread does not depend on which probes cross the edge: those are fixed by
the target's states and the lengths, before any read. So the misled probes are a fraction of each
state's, not mass added to the disagreement. They are counted over the tree's own midfixes, at most
`|Q| + 1` of them where its tree is in `classSet |Q|` (`misled_tail`, `Proofs/LoopRenoise.lean`).
Level 3 reads only strings beginning with a probe's first `k` letters (`level3_congr_zone'`), so a
probe whose first `k` letters are no probe's reads past each midfix, itself and extended by a
letter, independently of the tree (`measure_inter_of_reads`, `Proofs/LoopPersist.lean`), wrong with
chance at most `κ`, and two such probes read independently where their first `k` letters differ;
the probes sharing a probe's first `k` letters weigh at most `B₃ prefixMax`. Given the tree's
midfixes, the rest of a fixed set `E` of mass `x` weigh `X_E` with `E X_E ≤ e x`, `e = 2(|Q|+1)κ`,
and, each first `k` letters holding at most `prefixMax` of the probes, `Var X_E ≤ prefixMax · e x`,
so by Chebyshev (`meas_gt_le_of_pairs`) they weigh more than `x/2 + u` with chance at most

    2 prefixMax · e/((2 − 4e)(u − 2 B₃ prefixMax))

where `2 B₃ prefixMax < u` and `e < 1/2` (`misledTail`), whatever `x`. The clustering, run at
`εcov = min(ε, ξ/(8N))`, reads every covered state on its true side (`sideOf_eq_of_mass`). Every
covered state's edges are then true and it accepts as the target does, so the hypothesis from the
covered start's true leaf tracks the re-rooted target on every run that stays covered
(`classify_start_le`).

The pick compares each leaf against the family's read, not the oracle's: the oracle's noise rates on
and off the language may differ, and agreement with such a read weighs a leaf's errors by `1 − 2η`
of the string's true label, which can rank a worse leaf first. The family's read of a string of
length `L` is wrong at most `(1025/1024) N εcov` of the time and undecided at most `(3/8) N εcov`,
with `N εcov ≤ ξ/8`, so a leaf's mean score is within that of its accuracy. Reads persist per
string, so the held-out strings' reads are fresh only where no held-out string repeats or begins
with the first `k` letters of a prefix the last round's clustering read or of one of its probes:
level 3 reads only short strings beginning with some probe's first `k` letters
(`level3_congr_zone'`, `Proofs/LoopRenoise.lean`), and the earlier rounds read other suffixes. That
collision has chance at most `n(6(|Q|+2)RB + B₃ + n) p₀`, `p₀ = |Σ|^-k` for uniform strings, which
`L ≥ L₀` makes at most `δ/(4(|Q|+1))` for each round that can be last. Off it, given the loop, each
leaf's score is within `nξ/8` of its mean given the strings, and that within `nξ/8` of `n` times its
mean over the strings, but for chance `4(|Q|+2)e^(−nξ²/32)` (`scoreBad_le`), the tree having at most
`|Q| + 2` leaves, so the picked leaf misjudges at most `γ + ξ/2 + (3/8 + 1025/512)Nεcov ≤ γ + ξ`
(`pickLeaf_le`). `δ` splits half to the loop, a quarter to the scores and a quarter to collisions.

## The loop

`indecisiveLStarLoop` (`OrthoDFA/Code/LearnerLoop.lean`) clusters a family over the populations so
far, runs level 3 with it, and adds a harvest's sampler, scaled to a probability, as a population of
every later round. The whole run, every round's clustering and level 3, reads one noise. Each
round's clustering draws its prefixes, certification prefixes and suffixes off every one the earlier
rounds' clusterings read (`offLaw`), so no two rounds read the same string. Level 3 runs at
disagreement rate `ε` and draws its probes from `D` conditioned on their first `k ≥ 1` letters
avoiding those of every prefix the round's clustering read. Whatever its two rates, the family's
read is on its rarer decided side at most `bandKappa` times as often as undecided (`BandKappaBound`,
`OrthoDFA/LowLevelClaims/BandKappa.lean`), and the family's size puts `bandKappa` at most `ζ =
δ'/(8W)`. So the family is level 5's oracle at `κ = ζ`. With `f = 0` level 5 tolerates one spurious
leaf, within level 3's `|Q| + 2`. Its race tail at `t = 1/(λ + 2κ)`, `t' = 2t` is at most
`e²(λ + 2κ)²`, `λ = κ m` (`race_tail_choice`, `Proofs/LoopSimple.lean`), and

    W = max(e √δ' W_m / 2, 1),

`W_m = (16/5)S log(8S(S+1)/δ') + 3 + (|Q|+2)|Σ|(|Q|+1)`, puts `λ + 2κ` under `√δ'/(4e)`, and with it
`2(|Q|+1)ζ` under `1/4`; `16 B₃² prefixMax ≤ δ'` and `B₃ ε ≥ 34` put `2 B₃ prefixMax` under `ε/16`
and `|Q||Σ| prefixMax` under `δ' ε/544`, so the Chebyshev bound above at `u = ε`, over the `|Q||Σ|`
pairs of a state and a letter, is under `δ'/4`; no two of a round's probes sharing their first `k`
letters takes the rest of level 5's `δ'/8`. A round's claim (`RoundSpec`,
`Proofs/LoopTermination.lean`) is level 3's ending, transferred to `D`, its tree in `classSet |Q|`,
and its misled probes few in each state's (`FewMisled`); it fails with chance at most `δ' + δ'/4`:
level 5's claim and the Chebyshev bound (`level3Fail_le`, `Proofs/LoopRenoise.lean`). A harvest's
sampler yields at least `ε/96` and finds a string at most as often as the probes share its
length-`k` prefix, so a population puts at most `96 prefixMax / ε` on a string, within `cap`, which
also bounds the suffixes' atoms; its strings are at most `L₁ + |Q|` long, the tree's midfixes at
most `|Q|`. Against `cap` the clustering's atom conditions hold, every population tame: its
samplers' rejections, `NP·cap` under the rate scale times the signal; the hash's balance and the
quality's deviation, `log₂ NP·cap` under `δ` times the rates squared; and the tables' own mass,
`NP·cap` under the slack. The earlier rounds' reads, at most `6(K+1)NP` prefixes and `NS + 1`
suffixes a round, hold at most `m` of any population and `ms` of the suffixes. `NS` and `NP` are
bounded by `poolCount_le` and `prefCount_le_poly` (`counts_le`, `Proofs/LoopSimple.lean`).

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
the sampler at any set of states (`looseEndsWell_of_endsWell`, `Proofs/LoopTransfer.lean`), and its
misled probes at `m` more (`real_inter_le_of_avoiding`).

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

Each level proves one L\* round correct and states its ending through
`OrthoDFA/LowLevelClaims/RoundEnd.lean`: the round ends well when it is consistent, the hypothesis
and the tree parting on at most `ε` of the probes, or ends in a harvest whose sampler is spread,
yields, and puts at most `slope·T + slack` of its law on the states read undecided at most `T` of
the time. Each level says what a state's chance of reading undecided is.

1. #417, `Level2RoundCorrect`: ideal reads; a state reads undecided where all its strings do.
2. #419, `Level3RoundCorrect`: indecision random, decided reads always right.
3. #420, `Level4RoundCorrect`: rare wrong reads at every state, a few tolerated as spurious
   splits.
4. #499, `Level5RoundCorrect`: wrong reads at most `κ` times the state's indecision, a few
   tolerated; the round the loop runs, at the family's `κ`.

## Open

- **The shipped Python uses fixed settings**, not the ones the claims choose.
- **The rarer reads past a midfix** are bounded over the tree's at most `|Q| + 1` midfixes, each
  read wrong at most `κ` of the time. They cap `ζ` only through `2(|Q|+1)ζ < 1/2`, at the
  README's settings `ζ < 0.0625`; level 5's race tail binds, at `ζ ≈ 7.5e-7`. It charges `κ`
  where the read's chance is `κ U` at the state of `y·m`; no claim the loop proves bounds `U` at
  `y·m` under the probes' law, only at the populations' own strings.
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

- `OrthoDFA/Claims/` — the claims above. Each reads the definitions of the files below it names.
  The rest of `OrthoDFA/` is in three directories: `LowLevelClaims/` states the explicit and
  supporting claims, which `Claims/` and the headline rest on; `Code/` is the algorithm and the
  definitions it runs, which every claim is about; `Proofs/` need not be read.
- `OrthoDFA/Code/Clustering.lean` and `OrthoDFA/LowLevelClaims/Clustering.lean` — the oracle, the
  algorithm, and `ClusteringGuarantee`, the claim. `Code/Clustering.lean` imports only Mathlib.
- `OrthoDFA/LowLevelClaims/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over a
  population, the returned family decides a prefix wrongly at most `εcov + slack` of the time on the
  uniform pool and at most `1/2 + slack` on any other, and leaves it undecided at most
  `2·indecisionLimit + slack` of the time on each.
- `OrthoDFA/Code/FamilyRead.lean` and `OrthoDFA/LowLevelClaims/FamilyRead.lean` — the suffix
  family's read of a string: accept at `X(w) ≥ kh`, reject at `X(w) ≤ kl`, undecided between, where
  `X(w)` counts the members `v` whose query `w·v` answers
  1.  `FamilyReadBound`: with the language a DFA's, the family suffix-free and the band passing
  `BandPasses`, the reads are independent across strings and every DFA state has one distribution
  its strings read with, which is accept at most `ε` of the time, or reject at most `ε`, or
  undecided at least a third with accept or reject at most `ε₂`; and is on its rarer decided side at
  most `ε`, or at most `κ` times as often as it is undecided. `BandPasses` is the check the band
  selection rule `evidence_margin_for_population_size` makes, at `ε = cross_limit`, `ε₂ =
  MINORITY_READ_LIMIT` and `κ = MINORITY_UNDECIDED_RATIO`.
- `OrthoDFA/Code/TripartiteSearch.lean` — `tripartiteSearch`, a bisection for where `agrees` turns
  from agreeing to disagreeing that steps past undecided positions.
  `orthogonal_dfa/l_star/tripartite_search.py` is the same function.
- `OrthoDFA/Code/DTree.lean` — the discrimination tree, and `DTree.siftWithPath`, sifting a string
  through it under a three-valued cut. `MidfixTree.sift_with_path` is the same function. It also
  holds `sift`, the same sift under a read, and `splitAt`, which splits a leaf. `MidfixTree.split`
  is the same split, naming the leaf by its state.
- `OrthoDFA/Code/HypothesisDFA.lean` — a hypothesis DFA: a tree and learned edges between its
  leaves, and the walk a string takes along them; `walkEnd` is its last leaf when it follows every
  letter. `HypothesisDFA.split` splits a leaf and re-sifts the witnesses of the edges into it
  (`hypothesis_dfa.HypothesisDFA.split_leaf`).
- `OrthoDFA/Code/Probe.lean` and `OrthoDFA/LowLevelClaims/Probe.lean` — `probe`: a string walked
  along the edges from its first `k` letters' leaf and sifted, and where the two part, found by
  `tripartiteSearch` without recovery (`counterexample_probe.locate`). `ProbeCorrect`: with the
  edges witnessed, a wrong edge is one the probe crossed with a string whose successor sifts off its
  target, an unlearned edge is the one it stopped at, the at most one undecided string it reports
  reads undecided and is a prefix of the probe followed by a midfix, and a probe on which the walk
  and the sift part (`Disagrees`) never comes out clean.
- `OrthoDFA/Code/RateTest.lean` — `rateSide`, the binomial test of a count of hits against a rate.
  `statistics.rate_side` is the same test.
- `OrthoDFA/Code/SeqTest.lean` — `seqRate`, the sequential likelihood-ratio test of a count of hits
  against a rate. `statistics.seq_rate` is the same test.
- `OrthoDFA/LowLevelClaims/ProbeDistribution.lean` — how probes are drawn (`PrefixDistribution`).
- `OrthoDFA/Code/RoundEnd.lean` and `OrthoDFA/LowLevelClaims/RoundEnd.lean` — what it is for a round
  of the learner to end well, shared by every level: consistent, or a harvest. A harvest is a
  sampler; it puts at most `slope·T + slack` of what it finds on the states read undecided at most
  `T` of the time, it finds no string more often than the probes share its first `k` letters
  (spread), and it finds `y` in all (yields).
- `OrthoDFA/Code/Level2.lean` — level 2 of indecisive L\*, `indecisiveLStarLevel2`. A disagreement
  at decided reads splits the edge's leaf at once (Kearns–Vazirani: the probe's prefix and the
  edge's witness part at the edge's letter and the midfix where their successors' leaves part), and
  an unlearned edge is learned. The undecided reads probes stop at, and the searches that stopped at
  an undecided midpoint (the disagreements), are counted since the hypothesis last changed. After
  each probe a sequential test (`seqRate`) checks each count against the midpoint of the two rates
  (`alternativeRate`): the round harvests once the undecided reads' test of `harvestRate` fires, and
  ends consistent once the disagreements' test of `disagreementRate` fires. Under the `j`-th
  hypothesis both run at level `δ / (2 (j + 1) (j + 2))`, so each test's levels sum to at most `δ /
  2`. With `harvestRate < disagreementRate`, one of the tests fires within `stretchLength` probes
  under one hypothesis, at most `12 ε / (ε − y)² · log(1/a) + 1` (`stretchBound_mid_le`).
- `OrthoDFA/LowLevelClaims/Level2Claim.lean` — `Level2RoundCorrect`: with every state's reads all
  undecided or all on its side (`ReadsByState`), `0 < y < ε < 1` (`harvestRate`,
  `disagreementRate`), `0 < δ < 1` and at least `12 S ε / (ε − y)² · log(2 S (S + 1) / δ) + S`
  probes, `S = |Q|((|Q| + 2)|Σ| + 1) + 2|Σ| + 1` the most hypotheses, level 2 ends well, consistent
  with its hypothesis and tree parting on at most `ε` of the probes or a harvest whose sampler
  yields `y`, but for chance at most `δ`. A harvest's sampler (`harvestLaw`, defined there) draws a
  fresh probe, walks it under the final hypothesis, and finds the undecided read it stops at.
- `OrthoDFA/Code/Level3.lean` and `OrthoDFA/Code/Level3Harvest.lean` — level 3,
  `indecisiveLStarLevel3`: level 2's probes with reads drawn at random, the search stopping at an
  undecided midpoint, as at level 2. A clean disagreement splits the edge's leaf at once and an
  unlearned edge is learned. Its counts are kept over the stretch since the hypothesis last changed
  and tested after every probe: the ends (probes whose start or end sift stops undecided) and the
  middles each by `seqRate` against an alternative (twice `endsRate`, twice `middleRate`), and
  success, counting the searches and the ends together, by `seqRate` against half
  `disagreementRate`. Its settings are the yield `y = harvestRate` and the disagreement rate `ε =
  disagreementRate`; the tests' rates follow from them: `endsRate = y`, `middleRate = 1/4` and
  `searchShare = ε/6`. Under the `j`-th hypothesis each test runs at level `δ / (8 (j + 1) (j +
  2))`. A harvest is the site whose test fired, `ends` or `middles`.
- `OrthoDFA/LowLevelClaims/Level3Claim.lean` — `Level3RoundCorrect`: with every string's read drawn
  once, independently, undecided with its state's chance and never on its state's wrong side, every
  length-`walkStart` prefix at most `p₀` of the probes, `0 < y ≤ ε/48`, `ε < 1`, `0 < δ`, and at
  least `36 S / ε · log(16 S (S + 1) / δ) + S` probes, level 3 fails to end well, consistent with
  its hypothesis and tree parting on at most `ε` of the probes (`EndsWell` over `searchSet ∪
  endsSet`, the probes that search or stop undecided at their start or end sift) or a harvest whose
  sampler yields `y` and puts at most `A·T + η/y` of its law on the states read undecided at most
  `T` of the time, with chance at most `δ`. Here `A = (|Q| + 1)(6/y + 16(2 + ⌈log₂ L⌉))` and `η = p₀
  log(8(|Q| + 1)K/δ)`, with `K = (1 + (|Q| + 1)³|Σ|)^|Q| (|Q| + 3)^((|Q| + 2)|Σ|)` counting the
  trees level 3 can grow with an edge map into each one's leaves. A probe's start and end sifts read
  at most `3(|Q| + 1)` strings, and a search queries at most `2 + ⌈log₂ L⌉` positions, each sift
  reading at most `|Q| + 1` strings. Making a string undecided can only stop the search sooner, so
  the search reaching a position and a string read there undecided have chance at most the product
  of theirs. At each state's rate `t` and every tree, but for chance `δ/8` at each site over the
  states, the probes stopping undecided at their start or end at the states read undecided at most
  `t` of the time are at most `6(|Q| + 1)t + η`, and those stopping at an undecided middle there at
  most `2(2 + ⌈log₂ L⌉)(|Q| + 1)t` of the searches plus `η` (`harvest_noise`); a middles harvest
  that ends well finds at least an eighth of the searches. The sampler (`harvestLaw`, in
  `Code/Level3Harvest.lean`) draws a fresh probe, walks it under the final hypothesis and finds the
  probe's undecided string at the site, as the tests counted it.
- `OrthoDFA/LowLevelClaims/Level4Claim.lean` — `Level4RoundCorrect`: reads wrong at most `εw` of the
  time (`EpsReadOracle`): level 3's algorithm, unchanged, fails to end well within `|Q| + f + 2`
  leaves with chance at most `δ`. Its premises are level 5's with `εw` in place of `κ` in the race
  tail: a probe's reads number at most `w = min(L + 1, 3 + ⌈log₂ L⌉)(|Q| + f + 1)` and a re-sift's
  `|Q| + f + 2`, each read wrong at most `εw` of the time, so a round of reads costs at most
  `εw max(w, |Q| + f + 2)`, and the tail charges `εw` on each of the at most `b` strings the
  L\* round reads. Both levels are proved from one argument on reads wrong at most `εw + κ U q`,
  internal to `Proofs/Level5.lean`.
- `OrthoDFA/LowLevelClaims/Level5Claim.lean` — `Level5RoundCorrect`: level 3's algorithm, unchanged,
  with each read on its state's wrong side with chance at most `κ U q` (`KappaReadOracle`), fails to
  end well within `|Q| + f + 2` leaves with chance at most `δ`, tolerating a few wrong reads. Its
  premises are level 3's at `|Q| + f` states, the sampler's bounds at `δ/16` at each site, and
  `B₃(B₃ − 1)/2 · p₀ + exp(t' κ m)/(1 + t)^(f + 2) ≤ δ/8` for `t κ ≤ 1 − e^{−t' κ}`. Each round
  of reads (a probe's, or a re-sift's) reads at most one string undecided, so costs at most `κ`,
  and `m` counts those strings over the whole L\* round, within `|Q| + f + 2` leaves.
  - A step reads in rounds: its probe's, then one per edge its split re-sifts.
  - Where no two probes share their first `walkStart` letters (the first term), every string read
    belongs to one probe. A round with a fresh wrong read then adds at most one spurious leaf or
    one bad edge, and a split on a bad edge drops it.
  - So the tree outgrows `|Q| + f + 2` leaves only after `f + 2` such rounds, which
    `RaceTailBound` bounds; within the allowance, `Level3AllowanceCorrect` gives `7δ/8`.
- `OrthoDFA/LowLevelClaims/FamilyOracle.lean` — `FamilyKappaOracle`: a suffix family's reads (a DFA
  target, a suffix-free family and a band read on its rarer decided side at most `κ` times as often
  as undecided, `RarerBounded`) are level 5's `KappaReadOracle`, each state's side the one it reads
  more often (`readSide`).
- `OrthoDFA/LowLevelClaims/SuffixFamily.lean` — `SuffixFamily`: the oracle, DFA, family and band,
  with its reads and each state's read distribution.
- `OrthoDFA/Code/LearnerLoop.lean` — the loop, `indecisiveLStarLoop`: each round clusters, runs
  level 3 reading the family at the gate's cut, and hands its harvest's sampler on as a population.
- `OrthoDFA/Code/LoopDFA.lean` and `OrthoDFA/LowLevelClaims/LoopDFA.lean` — `LearnerLoopDFA`,
  `LearnerLoopDFASimple` and `LearnerLoopDFADist`, `pickLeaf` and `rerootedDFA` (in `Code/`), and
  `windowLaw` and `windowStrings`, the distributions the loop draws its probes from over an input
  distribution and over uniform strings (in `LowLevelClaims/`).
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Utilities

These claims, also in `OrthoDFA/LowLevelClaims/`, state general lemmas the proofs use, about
probability rather than the algorithm. They are proved under `OrthoDFA/Proofs/` and named in
`Verify.lean` like the claims, but need not be read to trust the claims above.

- `OrthoDFA/LowLevelClaims/AdaptiveReads.lean` — `AdaptiveReadsBound`: reads drawn once per string,
  independently across strings, each bad with chance at most `ε`, and queried in rounds, each
  round choosing its strings from what earlier rounds read; some query of the first `n` rounds
  reads bad with chance at most `ε` times the most strings they query while none has.

- `OrthoDFA/LowLevelClaims/RaceTail.lean` — `RaceTailBound`: reads drawn once per string,
  independently across strings, each bad with chance at most `ε` plus `κ` times its chance of
  reading in `V`, and queried in rounds; it counts the fresh-bad rounds, those where a string no
  earlier round queried reads bad. Each round is fixed by what it and the earlier rounds read while
  fewer than `f` fresh-bad rounds precede it. If the new strings number at most `b` and those in `V`
  at most `m` over those rounds, each round costing at most `w`, then at least `f` rounds are
  fresh-bad with chance at most `exp(t'(ε b + κ m))/(1 + t)^f` for `t w ≤ 1 − e^{−t' w}`. The proof:
  `(1 + t)^c e^{−t'(ε q + κ v)}` at the counts so far has mean at most `1`.
- `OrthoDFA/LowLevelClaims/Allowance.lean` — `Level3AllowanceCorrect`: level 3's algorithm, its
  reads independent and undecided at their state's rate but with no premise on wrong reads, fails to
  end well or grows its tree past `|Q| + f + 2` leaves with chance at most `7δ/8` plus the chance
  that a step grows it past them (`GrowsRun`), under level 3's premises at `|Q| + f` states, the
  sampler's bounds at `δ/16` at each site. Level 3's stretches use right reads only to keep the tree
  in its class; the proof runs level 3's frame on the target padded with `f` unreachable states,
  with the leaf count as the invariant. A step toward levels 4 and 5.

- `OrthoDFA/LowLevelClaims/BitSum.lean` — `BitSumTails` (Hoeffding, 1956): a sum of independent bits
  at any rates lands at or below a count at least two under its mean no more often than a binomial
  with as many trials or more at a rate the mean clears, and likewise above.

- `OrthoDFA/LowLevelClaims/BandKappa.lean` — `BandKappaBound`: the vote `Bin(a, p) + Bin(N − a, r)`,
  `N ≤ n`, read at `kl` and `kh`, is on its rarer decided side at most `bandKappa` times as often as
  it is undecided, whatever `p` and `r`: for any count `c` inside the band, with `Y ~ Bin(n, 1/2)`,
  `bandKappa = max(P(Y ≥ kh)/P(c ≤ Y < kh), P(Y ≤ kl)/P(kl < Y ≤ c))`. The proof: by Newton's
  inequalities `P(X = j)/C(n, j)` is log-concave in `j`, so from `c` it does not rise or up to `c`
  it does not fall, and either way compares each tail count with each count of the window.

- `OrthoDFA/LowLevelClaims/SuccessiveDraws.lean` — `SuccessiveDrawsTails`: over independent draws,
  where `T` with any fewer than `n` more strings holds at most `t` of the law, the draws among the
  first `n` that land in `T` or repeat an earlier one reach a count no more often than `Bin(n, t)`
  does, and the distinct draws outside `T` that land in `P` reach a count no more often than `Bin(n,
  D(P))`, and stay at or below one no more often than `Bin(n, D(P) − t)`, does.

- `OrthoDFA/Code/CoveredStart.lean` and `OrthoDFA/LowLevelClaims/CoveredStart.lean` —
  `HypothesisDFA.Classifies`, the run of a hypothesis DFA from a leaf along a string, and
  `CoveredStartCorrect`: where every state of a set `S` has its
  edges learned and true and is read on its true side, the hypothesis from a state's true leaf
  (`Random.leafOf`) misjudges at most `γ` of the uniform strings of length `L` when re-rooting the
  target at that state classifies `1 - γ` of them with its run staying in `S`
  (`rerootAccuracy`). Also `uniformStrings`, the uniform distribution over the strings of a length,
  and `prefixLaw`, the law of a string's first `n` letters.

- `OrthoDFA/LowLevelClaims/TripartiteSearch.lean` — `TripartiteSearchCorrect`: each result says what
  `agrees` is where it points, the queries stay between the endpoints and number at most
  `2 + 3⌈log₂(hi − lo)⌉`, or `2 + ⌈log₂(hi − lo)⌉` without recovery, and with nothing undecided it
  finds the turn exactly.

- `OrthoDFA/LowLevelClaims/DTree.lean` — `SiftWithPathCorrect`: the reads are the string followed by
  each midfix on its way, no more than the tree is deep; a decided sift ends at a leaf, along the
  path its reads decide, and an undecided one at its only undecided read. `SiftAndSplitCorrect`:
  `sift` lands where `siftWithPath` does, strings at different leaves read decidedly apart where
  their paths part, and a split replaces its leaf by two children, reads its midfix there, and
  changes only the sifts that landed on it.

- `OrthoDFA/LowLevelClaims/HypothesisDFA.lean` — `WalkCorrect`: the walk starts at its leaf, follows
  one learned edge per letter, stops short only at an unlearned one, and stays on the leaves when
  the edges do. `SplitAndLearnCorrect`: splitting a leaf, or learning an edge from a witness, keeps
  every learned edge witnessed (`EdgesOK`).

- `OrthoDFA/LowLevelClaims/RateTest.lean` — `RateTestSound`: over independent draws `rateSide` says
  above with chance at most its failure chance when the true rate is at most the tested one, and
  below likewise when it is at least.

- `OrthoDFA/LowLevelClaims/SeqTest.lean` — `SeqRateSound` (Ville's inequality): over independent
  draws, testing with `seqRate` at every look fires at some look with chance at most `a` when the
  true rate is on the far side of `p₀` from `p₁`.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, one L\* round
correct at each level above, and the loop's DFA correct from a start it picks, under the Python's
covered-start precondition, for a model of the loop that differs from the Python as listed above.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
