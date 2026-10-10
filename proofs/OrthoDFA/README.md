# Machine-checked correctness of the E-L\* clustering algorithm

## The theorems

Every theorem below is proved in Lean with no `sorry`, using only `propext`, `Classical.choice`
and `Quot.sound` (`OrthoDFA/Verify.lean` prints this).

- `ClusteringGuarantee`: with probability at least `1 − δ` the clustering loop stops, and the
  family it returns cuts all but `εcov` of the uniform pool as the noiseless oracle does, at a
  polynomial number of queries.
- `ClusteringQualityGuarantee`: averaged over a population, the returned family decides a prefix
  wrongly, or leaves it undecided, only a bounded share of the time.
- `RoundOutcome`: if the DFA/DT check fails on a share `d` of strings, the harvest sampler finds a
  string on at least a `d/L` share of attempts, and harvests no single string too often.
- `RoundProgress`: but for a small chance of flipped node reads, a round ends in one of seven
  outcomes, each of which moves the learner forward.
- `RoundAtK`: a round walked from position `k` makes one claim per probe outcome, each holding for
  any reads but for the gate tests' failure chances.
- `StartExists`: a DFA that agrees with the target on a set of states misjudges few draws when
  started at the image of a covering state.
- `StartRootCovered`, `EndsCovered`, `BadShare`: the FNR gate's population bounds how often the
  walk's start is undecided at the root, the populations below the root are undecided at least as
  often as the walk's ends, and an often-undecided population has much mass at badly read states.
- `RoundTrichotomy`: one reading ends consistent (the gate settles above `acc` and its start
  disagrees on at most `1 − acc`), with live edges to rerun, holding a class that fired, or
  halving only above `τ₀` or with little mass left, but for `2(log₂(ng/30)+2)a + (1−ν)^nr`; off
  a small noise set every harvest class is close to its incidental rate.
- `RoundTrichotomyLevel`: the same over all of a round's readings, with the gate's chance spent
  as `a·2⁻ʲ` and a certificate of failure chance `α`, but for `4(log₂(ng/30)+2)a` plus
  `(1−ν)^nr + α` per reading in expectation.
- `RoundQualityLevel`: off a noise set of measure `δ`, the hypothesis a round ends with has every
  harvest class close to its incidental rate, at a fluctuation set by the readings it made.
- `RoundStrongReadings`: in the round as Python runs it (a fresh quiet streak per pass, every
  edge live, a class that fires ending the round, a backstop budget on probes), every reading but
  the last spends at least `patience` probes, so `readings × patience ≤ budget + patience`.
- `RoundStrongLeaves`: it ends with at most `|Q| + 2` leaves plus its noisy splits, those with one
  of their at most `2·depth + 2` sifting or parting reads off a reference placement's side.
- `RoundStrongSameState`: a split between two strings of one state is noisy.
- `RoundStrongLeafPaths`: its learned edges join leaves.
- `RoundStrongNoStop`: at the hypothesis it ends with, every attempt on an edge a draw's search
  ends at splits or adds a member.
- `RoundStrongTrichotomy`: it ends consistent, holding a class, halving, exhausted, or out of
  readings only where `Rmax × patience` is within the budget, but for `4(log₂(ng/30)+2)a`,
  `(1−ν)^nr` per reading and the certificate's failure chance per call, in expectation. Ending
  exhausted needs the refusal sample to find a live edge and no class firing at every reading
  until the budget is spent; `RoundStrongExhausted` bounds its chance.
- `RoundStrongQuality`: `RoundQualityLevel` for that round.
- `EdgeCause`: a draw whose search ends at an edge has a prefix with a read, down its state's
  route by a reference placement, that the cut decides on the other side; or the edge it ends at
  leads off the leaf its next state is placed at.
- `SpuriousDraw` and `RoundStrongSpurious`: however the hypothesis was chosen, a draw apart from
  the noise has such a read with chance at most `spurRate` (the misread chances of at most `ψ₀`
  over its prefixes and the midfixes up to length `n`) plus the chance it lies in `BadRoute` of
  the hypothesis's tree: routed through a midfix its state is misread at more than `ψ₀`, a state
  near the band there. For each reading of the round, for each refusal draw and fresh probe.
- `RoundStrongPower`: in the round, at some key the first split test in the power case (counting
  only strings no other key's tests or other reads have read, its sides' mean answers apart by
  `τ` beyond its threshold) or splitting is in the power case and does not split, with chance at
  most `e^{−τ²}` times the chances, summed over the keys, that there is such a test.
- `RoundStrongExhausted`: a pass ends after `patience` quiet steps in a row, and a step is quiet
  unless it reaches a split test (Python charges only the pass's probes to the budget). So the
  round spends at most `patience + 1` probes per step that is not quiet, and `patience` more. For
  draws of length `L`, where the budget at two leaves covers `patience + 1` probes for each of
  `|Q| + N₁ + N₂` such steps, the round ends exhausted with chance at most `RoundStrongPower`'s
  bound averaged over the draws; plus, over `N₁`, `spurRate` and the chance a step probes the draw
  in its tree's `BadRoute`, summed over every refusal draw and probe (Markov on the steps with a
  read off their probe's route); plus the chance that `N₂` steps that are not quiet have no such
  read and reach no test in the power case or splitting (the stated residual: too few members on a
  side, too few fresh strings, or too small a gap); plus the chance of a noisy split, which is
  open. A step that is not quiet splits, at most `|Q|` times without a noisy split, or adds a
  member after a test that did not split, which falls in the power term or the residual.

- `TallyRound`: in the tally loop, the family's read of a string taken as one accept, reject or
  undecided draw, one round over `T` probes reaches a state where no path-good transition's
  boundary has mass `q₀`, or ends in success, or ends in a harvest that bad read-states trigger,
  but for the chance that the noise event `TallyE` fails, `T · P(Bin(n, c₀q₀) < m)`,
  `P(Bin(T, (|Q| + 2)² |Σ| ρ) ≥ m)` and `T² (Lmax |Σ| + 1) a`. `TallyE` is a hypothesis on the
  reads over every genuine tree, not proved here.

## Open

Two claims are not proved, and are not assumed or sorried anywhere (`TallyRound` assumes its
noise event `TallyE` as a hypothesis, to be bounded separately):

1. The chance of a noisy split. `RoundStrongLeaves` bounds the leaves by the noisy splits, and a
   noisy split needs one specific decided read on the minority side, but the witness it reads is
   picked from the pool by earlier reads, so the per-string `depth·ρ` bound does not apply and a
   union over the pool is vacuous. Bounding the chance that a round makes any noisy split is open.
2. The chance a round ends exhausted is bounded by `RoundStrongExhausted` only up to the
   noisy-split chance above and the non-power residual, which is stated, not bounded.

Two gaps between the model and Python:

- Each theorem is per round, with the family, the seed and the band fixed, and treats the noise
  as fresh. Reads made in earlier rounds, which pin bits a later round reads again, are not
  modelled.
- The round's read log over-approximates its reads by every read a step or a reading could make,
  while Python's memo holds exactly what was asked, including reads only Python makes (the
  prefill, the table, `denoise_accept_labels`). The split tests' skip sets can therefore differ
  in rare coincidences.

## What to read

- `OrthoDFA/Clustering.lean` — the oracle, the algorithm, and `ClusteringGuarantee`, the claim.
  It imports only Mathlib.
- `OrthoDFA/ClusteringQuality.lean` — `ClusteringQualityGuarantee`: averaged over a population,
  the returned family decides a prefix wrongly at most `εcov + slack` of the time on the uniform
  pool and at most `1/2 + slack` on any other, and leaves it undecided at most
  `2·indecisionLimit + slack` of the time on each.
- `OrthoDFA/Stage.lean` — `RoundOutcome`: for any tree and DFA a round ends with, if the DFA/DT
  agreement check fails on a share `d` of sampler strings, the harvest sampler finds at least one
  string on at least a `d/L` share of attempts, harvests no single string `t` on more than
  `∑_{i ≤ |t|} min(i+1, L)/L · D(the draw starts as t does)` of them, and reads no single string
  on more than that.
- `OrthoDFA/Round.lean` — `RoundProgress`: through a suffix-free vote family whose node reads of
  states read undecided less than `uHi` flip with chance at most `φ`, a round ends, but for chance
  `(L+1)(N+1)·φ/ε + passReadBound·φ`, with:
  1. the DFA/DT check failing on at most `ε` of sampler strings;
  2. a population the next gate must act on;
  3. the pass halving the indecision limit;
  4. its bisection population forcing the next gate;
  5. a state its family reads badly;
  6. a visited string's extension read off the hypothesis's edge more often than not; or
  7. the pass running out of probes before a patience streak.

  `Pass.lean` states the pass, with its bisection population, and `Automaton.lean` the target.
- `OrthoDFA/StartAtK.lean` — the round walked from position `k` along learned edges only.
  `probeOutcome` says what one probe comes to: agree, start or end undecided, a pair, an edge, a
  triple, or a member at an unlearned edge. The gate reads fresh draws until its agreement test
  and its ends test have both settled. `RoundAtK` makes one claim per outcome:
  - Agreement: a passing gate's reading disagrees on at most `1 − acc + δ` of draws, and a
    refusing one's on at least `1 − acc − δ`.
  - Ends: the ends test reading above or below its threshold, `min(2·(depth − 1)·f, 1)`, is
    right to within `δ`.
  - Pairs: the pair test tripping means more than `θp` of searched draws end at a pair, and a
    refusal with nothing searched means at most `δ` of draws are searched.
  - Edge: the probe splits a leaf, adds a member, or stops at a string the cut cannot place.
  - Member: it sits at a leaf whose edge by some letter is unlearned, and is placed followed by
    that letter.

  The tests run at the looks `n₀, 2n₀, 4n₀, …` and the cap. These claims hold for any reads, but
  for `(3·(log₂(ng/n₀) + 2) + 1)·a + 2·exp(−2·ng·δ²) + exp(−min(n₀, ng)·δ)`.
  - Triples: over the oracle's noise, those harvested at states read undecided less than `uGood`
    are at most `uGood·depth·E[visits]`, plus the draws whose middles the pass may have read, plus a
    fluctuation `ε`, but for `prefixMax D k / ε²`. `visits` counts the middles a search
    visits.
- `OrthoDFA/StartState.lean` — `StartExists`: a DFA `H` that agrees with the target on a set `S`
  of target states, each read as `H`'s state `h q`. Started at `h q`, it misjudges at most `η` of
  the draws whenever the target re-rooted at `q` both stays in `S` and agrees with the target on at
  least `1 − η` of them. This is separate from the round: how `H` comes to agree on `S` is not
  assumed here.
- `OrthoDFA/Ends.lean` — `StartRootCovered`: the FNR gate reads a population of the sampler's
  draws cut to length `k`, and what it certifies there bounds how often the walk's start is left
  undecided at the root. Below the root, the round adds the populations `X·m`, `X` either end's
  draws and `m` a midfix below the root. `EndsCovered`: a sift left undecided below the root
  leaves some `x·m` undecided, so these populations are left undecided at least as often as the
  ends are. `BadShare`: a population left undecided `ā` of the time has at least
  `(ā − f)/(umax − f)` of its mass at states read undecided more than `f` of the time.
- `OrthoDFA/Trichotomy.lean` — `RoundTrichotomy`: one reading of the round. Off a noise set of
  measure at most `5·exp(−2ε²/prefixMax)`, every harvest class is at most its incidental rate plus
  slack: the draws sharing their first `k` letters with a string the pass read, and `ε`.
  There, but for the gate tests' failure chance per look and `(1 − ν)^nr`, the reading ends in
  one of: the gate settles above `acc` and its start disagrees on at most `1 − acc`; live edges
  turn up to rerun; some class fires; or the limit halves, and then `f ≥ τ₀`, or members do not
  fire on a first hit, or the walk's non-agreeing mass is at most `ν` and, where the gate settled
  below, every covering start leaves the classes and live edges at most `ν`.
- `OrthoDFA/RoundLevel.lean` — `RoundTrichotomyLevel`: the round over its readings. Each reading
  runs the pass with the previous reading's live-edge draws first, then the gate (at failure chance
  `a·2⁻ʲ`) and, where its test settles above, a certificate taken as a black box with failure
  chance `α`; anything else, an unsettled gate included, reads a refusal sample that reruns live
  edges, holds a fired class, or halves. The round ends consistent (its start disagreeing on at
  most `1 − acc`) and certified, holding a class, halving above `τ₀` or with the walk's
  non-agreeing mass at most `ν` (and every covering start at most `ν` where the gate settled
  below), or exhausted, but for `4(log₂(ng/30)+2)a` plus, per reading made in expectation, the
  refusal sample's miss and `α`. The exhausted exit is a problematic component: give-ups and stopped
  attempts are not yet bounded. `Proofs/RoundLevel.lean` proves it reading by reading.
  `RoundQualityLevel`: for any draws, off a noise set of measure `δ`, the hypothesis the round ends
  with meets `QualityHolds` against every probe its passes took, at a fluctuation `ε_r` set by the
  number `r` of readings it made. The reruns' probes are picked by reads, so the set covers every
  choice of live-edge draws, `2^((nr+1)·r)` of them for `r` readings, each `r` at chance `δ·2^-(r+1)`.
- `OrthoDFA/RoundStrong.lean` — the round as Python runs it: each pass starts a fresh quiet
  streak and stops at `patience` quiet probes or the backstop budget, every edge a refusal draw's
  search ends at is live, a gate passes only where its test settles above `acc`, the
  certificate's failure chance is spent per call, and on a refusal a class that fires ends the
  round holding it, else live edges rerun, else the limit halves. The split test counts the reads
  of a held-out block `K.block F`, skipping a string the round has read before unless a test at
  the same leaf and distinguisher counted it first; `KState.log` records what the round has read,
  over-approximated by every read a step or a reading could make. The suffix-free premise covers
  `F ∪ train ∪ block`. The certificate reads the tree, the edges, its draws and their bits. The
  knobs' `forced` keys, whose tests are taken as not splitting, are none in the round itself; the
  power proof couples the round to one forcing a key. Proved in
  `Proofs/RoundStrong.lean` and `Proofs/EdgeAttempts.lean`: `RoundStrongReadings`,
  `RoundStrongLeaves`, `RoundStrongSameState`, `RoundStrongLeafPaths`, `RoundStrongNoStop`,
  `RoundStrongTrichotomy` and `RoundStrongQuality` (the round's passes are decided by the bits at
  what they can read, segment by segment, `segRun_determined`).
- `OrthoDFA/Spurious.lean` — `EdgeCause`, `SpuriousDraw` and `RoundStrongSpurious`, proved in
  `Proofs/Spurious.lean`. A read off the route lies in a set of the noise and the draw that no
  hypothesis enters, or in `BadRoute` of the hypothesis's tree, which holds no noise; so the
  draw's independence from the noise is all the bound takes, however adaptively the hypothesis
  was chosen.
- `OrthoDFA/Exhausted.lean` — `RoundStrongPower` and `RoundStrongExhausted`, proved in
  `Proofs/Power.lean` and `Proofs/Exhausted.lean`. Power couples the round to one that takes the
  key as not splitting: the two agree until a test there splits, and the coupled round's first
  power-case test there is chosen by bits its strings are fresh from. Exhausted: each reading
  after the first reruns a live draw first, from the tree it was drawn against, so it starts with
  a step that is not quiet; `Proofs/Trace.lean` lists the round's steps in order, and no step
  tests at a key after its leaf splits.
- `OrthoDFA/Proofs/Visits.lean`, `HarvestBound.lean`, `HarvestClasses.lean`, `Quality.lean` —
  the classes' quality. Each class is a computation whose harvest is a first undecided read,
  tagged; `HarvestBound` is the triples' fresh-read argument for any such class and any pass
  decided by the bits at what it can read, with Hoeffding
  over the draws grouped by their first `k` letters, which are independent inside a cell of the
  pass.
- `OrthoDFA/Proofs/TrichotomyBatch.lean`, `Trichotomy.lean` — the batch claims. A refusal leaves
  every start disagreeing on more than `1 − acc`; a covering start's disagreements off its own
  start region fall in the walk's classes; below `τ₀` each class fires on its first hit.
- `OrthoDFA/Proofs/Query.lean`, `Triple.lean`, `PassReadsK.lean`, `TripleBound.lean`,
  `RoundAtK.lean` — the triples' claim. A probe's processing is a computation asking the cut one
  string at a time; a triple's harvest is a middle's first undecided read; the pass is decided by
  the bits its reads ask; given those, draws with different first `k` letters are independent,
  and Chebyshev in each cell of the pass bounds the fluctuation.
- `OrthoDFA/FamilyRead.lean` — the suffix family's read of a string: accept at `X(w) ≥ kh`,
  reject at `X(w) ≤ kl`, undecided between, where `X(w)` counts the members `v` whose query
  `w·v` answers 1.  `FamilyReadTrichotomy`: with the language a DFA's, the family
  suffix-free and the band passing `BandPasses`, the reads are independent across strings and
  every DFA state has one distribution its strings read with, which is accept at most `ε` of the
  time, or reject at most `ε`, or undecided at least a third with accept or reject at most `ε₂`
  and the rarer of them at most `κ` times the undecided. `BandPasses` is the check the band
  selection rule `evidence_margin_for_population_size` makes, at `ε = cross_limit`,
  `ε₂ = MINORITY_READ_LIMIT` and `κ = MINORITY_UNDECIDED_RATIO`.
- `OrthoDFA/TallyLoop.lean`, `OrthoDFA/TallyRound.lean` — the tally loop and `TallyRound`, as
  sub-rounds. Probes walk from `k` as `sifting.read` does; a clean disagreement records its prefix
  and target at its edge. Each place keeps its count over the round: the start's undecided read
  strings, and each edge's reads and undecided read strings; the start's undecided rate above
  `θs`, or an edge's undecided reads exceeding `θe` of its reads by `exc` of them, harvests; the
  stretch's disagreement rate settling below `εd` succeeds. Otherwise one edge with `m` records at
  a target it does not point at is redirected there, or its leaf splits where its own target also
  has `m`; a split drops every record. Each law reads its rarer decided side at most `κ` times as
  often as it is undecided. A sub-round runs at one tree of the class (genuine splits and at most
  `S` that are not) until a split, genuine (real) or not (fake), or the round ends. `SubRound`
  bounds a sub-round's chance of being fake and of being bad or unfinished, given the noise event
  `TallyE` and `Terminates`; `HarvestGood` bounds a harvest that is not mostly at read-states that
  are not good. Proved in `Proofs/TallySub.lean`: `round_of_sub` composes sub-rounds (at most
  `|Q|` real ones, failing past `S` fake ones) into `roundW`, and `tally_round_of` gives
  `TallyRound` from `SubRound` and `HarvestGood`. `HarvestGood` is proved in
  `Proofs/TallyHarvest.lean`: at each edge the good read-states' undecided reads less `2θg` of its
  reads exponentiate to a supermartingale while the tree is in the class, and at the start they are
  at most binomial. `SubRound` is not yet proved.
- `OrthoDFA/Verify.lean` — names the proofs of the claims and prints their axioms, which should
  be only `propext`, `Classical.choice` and `Quot.sound`.

Everything under `OrthoDFA/Proofs/` is checked by Lean and need not be read to trust the claim.

## Scope

This proves the clustering step (`sample_suffix_family` and its gate) correct, that a round
either passes the DFA/DT agreement gate or leaves a harvest that yields and is spread, and that a
round ends in one of the seven outcomes above. It does not prove that the E-L\* learner outputs the
target DFA.

## Building

```
cd proofs/OrthoDFA
~/.elan/bin/lake build
```
