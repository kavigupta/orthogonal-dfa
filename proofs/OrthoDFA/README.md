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
- `RoundStrongBudget`: in the round as Python runs it (attempts counted, edges given up, a fresh
  quiet streak per pass), the probe budget is never reached, so the round never ends exhausted.
- `RoundStrongReadings`: that round makes at most `readStar(leaves)` readings.
- `RoundStrongLeaves`: it ends with at most `|Q| + 2` leaves plus its noisy splits, those with one
  of their at most `2·depth + 2` sifting or parting reads off a reference placement's side.
- `RoundStrongSameState`: a split between two strings of one state is noisy.
- `RoundStrongLeafPaths`: its learned edges join leaves.
- `RoundStrongTrichotomy`: it ends consistent, holding a class, halving with the edges actually
  given up, or out of readings only past `readStar`, but for `4(log₂(ng/30)+2)a`,
  `(1−ν)^nr` per reading and the certificate's failure chance per call, in expectation.
- `RoundStrongQuality`: `RoundQualityLevel` for that round.
- `RoundStrongNoStop`: no attempt on an edge stops; the search ends between decided sifts and the
  witness's paths are decided, so every attempt splits or adds a member.
- `RoundStrongDeadEdge`: off a noise set of measure `δ`, the draws left at given-up edges are at
  most those at edges the reads' likelier sides get wrong, plus `ρ` times the expected reads of
  the walk, search and edge at strings read undecided at least `u` of the time, plus the chance a
  string read undecided less than `u` of the time is decided on its less likely side times every
  read, plus the classes' slack.

## Open

Three claims are not proved. Only the last is stated, in `OrthoDFA/HalvingStrong.lean`, and
sorried in `OrthoDFA/Proofs/OpenLemmas.lean`, which nothing in `Verify.lean` imports.

1. The chance of a noisy split. `RoundStrongLeaves` bounds the leaves by the noisy splits, and a
   noisy split needs one specific decided read on the minority side, but the witness it reads is
   picked from the pool by earlier reads, so the per-string `depth·ρ` bound does not apply and a
   union over the pool is vacuous. Bounding the chance that a round makes any noisy split is open.
2. Split power on wrong edges with margin, which would split `RoundStrongDeadEdge`'s residue at
   wrong edges into a tail and the edges without margin. Not yet stated: a wrong edge whose leaf
   holds a single witness-side member can be given up through that member's one, persistent read
   with a chance of a few percent, so the claim needs at least `m*` members on each side as well
   as the two halves' means outside their bands by a margin.
3. `RoundCleanCrossing`: the chance that some string the round's passes can read, read undecided
   less than `u` of the time, is decided on its less likely side is at most `crossWell u` times the
   expected number of such strings.

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
- `OrthoDFA/RoundStrong.lean` — the round with what Python carries across readings: every end of
  the split test but a split counts against its edge, an edge with `mmax` of them is given up,
  the strings stopping the test's guards are held, each pass starts a fresh quiet streak, a
  member added on a given-up edge does not reset it, and probes are counted against
  `probe_budget`. A gate passes only where its test settles above `acc`, and the certificate's
  failure chance is spent per call. Proved in `Proofs/RoundStrong.lean`, with no noise:
  `RoundStrongBudget` (the budget is never reached, so there is no exhausted exit),
  `RoundStrongReadings` (at most `readStar(leaves)` readings), `RoundStrongLeaves` (at most
  `|Q| + 2` leaves plus the noisy splits: those with one of their `≤ 2·depth + 2` sifting or
  parting reads off a reference placement's side), `RoundStrongSameState` (a split between two
  strings of one state is noisy) and `RoundStrongLeafPaths`. `RoundStrongTrichotomy`, proved:
  the round ends consistent (start disagreeing on at most `1 − acc`, certified), holding a class,
  halving as in `RoundTrichotomyLevel` with the edges actually given up (every covering start's
  residue claimed only where the gate settled below), or out of readings only past `readStar`,
  but for `4(log₂(ng/30)+2)a`, `(1 − ν)^nr` per reading in expectation, and the certificate's
  failure chances over its calls in expectation. `RoundStrongQuality`, proved: the end state's
  harvest quality at the realised readings' `ε_r`, the round's passes being decided by the bits
  at what they can read, read segment by segment (`segRun_determined`). Open, not
  sorried (see `notes/start-at-k.md`): the chance of a noisy split, and split power for edges
  with margin.
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
