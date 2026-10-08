# The round trichotomy: proof status

The Lean now states one theorem, `RoundTetrachotomy` (see the last section). The statements
discussed before it (`RoundTrichotomy`, `RoundTrichotomyAll`, `RoundTrichotomyEdges`, the gap-premise
`RoundTetrachotomy`, `RoundTetrachotomyBoth`, `RoundOrHarvest`) and their helpers are removed from
the Lean; their analysis stays here.

Claim under study (`RoundTrichotomy`, `Round.lean`): but for a small probability, a round ends with

1. the DFA/DT check disagreeing on at most `ε` of `D`, or
2. a harvest that is (a) spread and (b) heavy on badly read states, or
3. a pass that halves `τ` (#407: `2·unchecked > τ·reads`).

Result: steps 1, 2 and 5 of the planned route hold; step 3 (early errors) is not the obstacle;
step 4 fails at moderate `τ` in a concrete regime that the five-symbol trap's first refused
round sits in. Details below.

## Setting and notation

- `A` the target DFA, `q_j` the target state after a draw's first `j` letters.
- Noise is keyed by the string: reads of distinct strings are independent, and a read of `w` is `1`
  with a chance fixed by whether `w ∈ L(A)`.
- `Vote(s) = #{v ∈ F : read(s·v) = 1}`. Undecided when `lo < Vote(s) ≤ hi`; the middle-of-band
  reading accepts when `lo + hi < 2·Vote(s)`.
- `u(q)`: the chance a string at `q` is left undecided (`stateIndecision`).
- `ν(y) = D(x starts with y)/L`, `V(q) = Σ_{y→q} ν(y)`; premise `ν(y) ≤ κ·V(A.state y)`.
- `d = D(DFAandDTDisagree)`, `Y = P_{x,e}(the replay harvests)`, `R̄` the reads per attempt.

## Premises

- (P1) String-keyed independent noise, as in `Oracle`.
- (P2) `lo ≤ hi`.
- (P3) Spread: `ν(y) ≤ κ·V(A.state y)`.
- (P4) Gap: every state has `u(q) ≤ uLo` or `u(q) ≥ uHi` (`GapPremise`). This is the clustering-purity
  fact the discussion called "bimodal": a state every suffix reads the same way has a vote a full
  signal from the middle, so `u ≤ acceptable_fnr`, which is `τ/10` by default.
- (P5) Freshness: the noise the replay reads is independent of the noise the pass used to build
  the hypothesis. False at the root's strings (every walk reads them). Needed for Steps 4–5 as
  stated; see "Open".

## Step 1: errors are state-level — holds

For fixed `s` the strings `s·v` (`v ∈ F`) are distinct, so `Vote(s)` is a sum of independent
Bernoullis whose rates depend on `s` only through `A.state(s)`:

    law(Vote(s)) = law(Vote(s'))   whenever   A.state(s) = A.state(s').

So `u` and the chance that the middle-of-band reading sends `s` to a given side are state-level.
The realised reading of a particular string is not.

**Lean fix (done).** The old `edgeWrong q` asked for some string reaching `q` to be misread. Under
string-keyed noise, some string of nearly every state is misread almost surely, so that `badness`
was 1 almost everywhere and `HarvestBad` held trivially. It is replaced by `edgeError`: a chance
over fresh noise, at worst over strings and letters. `badness = max(u, edgeError)`.

## Step 2: yield counts every in-sync anchor — proved

`InSync x e` says the walk from `ε` is, after `e` letters, where the reading puts them.

    InSync x e ∧ DFAandDTDisagree x  ⇒  SuffixDisagree x e            (suffixDisagree_of_inSync)

The proof: the walk restarted at an in-sync anchor is the walk from `ε` from there on. With
`AnchoredYield` this gives

    Y ≥ (1/L) · E[ #{e < L : InSync x e} ; x disagrees ]              (inSync_yield_holds)

Both are proved in Lean with no sorry, on `propext`, `Classical.choice` and `Quot.sound` only.

The restart argument generalises (paper only): if the walk restarted at `e` meets the walk from `ε`
at any later position, they end alike. So the anchors that fail are those where the walk from `ε`
is out of sync and the restarted walk never rejoins it.

## Step 3: early errors are not the obstacle — holds, with a caveat

An anchor fails only where the walk from `ε` is out of sync. By Step 1, a decided reading is wrong
only with a chance `η` that is negligible by the family's sizing (about 1e-14). So an
out-of-sync stretch starts at a visit to a wrong `(q, c)` pair, or at a bad state's coin-flip
middle reading. Neither is tied to short prefixes.

From (P3), for any `m`:

    P(x visits q before position m)  ≤  Σ_{|y|<m, y→q} L·ν(y)  ≤  L·κ·|α|^m · V(q),
    P(x visits q)                    ≥  V(q)·L/(L+1),

so

    P(all of x's visits to q fall before m | x visits q)  ≲  L·κ·|α|^m.

So a state's visits cannot concentrate on positions below `m* = log_|α|(1/(Lκ))`. They can
concentrate on any band of positions above `m*`: κ constrains string concentration, not the time
profile. Measured `β = E[#in-sync anchors / L | disagree]`:

- 0.37–0.66 on 14 of 15 refused rounds;
- 0.55 by `k*` (the last in-sync position) on the one early-mismatch round (subseq seed 2), where
  the first mismatch gives 0.10. That round halves: 117 of 149 quiet probes were unchecked.

So early errors are not what breaks the trichotomy, and the planned "heavy early error dismissed by
NO_SPLIT" candidate needs no separate treatment: a decided early error is caught by the pass. The
alternatives are that its crossings were unchecked, which halves, or that it is a bad state, which
is Step 4.

## Step 4: bad-heaviness — fails at moderate τ

Per attempt, the harvest's items are:

- **clean-state undecided reads**: about `C = R̄·u_clean` per attempt, each with badness at most `uLo`;
- **bad-state undecided reads**: about `B = R̄·s_b·u_bad`, where `s_b` is the bad states' share of
  reads; each has badness at least `uHi`;
- **disagreement prefixes**: at least `Y_edge` per attempt, each with badness about 1.

`HarvestBad` asks `E[badness] > 2τ`.

**The regime that breaks it.** Bad states are badly read but rarely visited, with all three of

    s_b · L / 2  >  ε                          (the gate fails)
    s_b · u_bad + u_clean  ≤  τ / 2            (no halving)
    B·u_bad + C·u_clean + Y_edge  ≤  2τ·(B + C + Y_edge)   (not bad-heavy)

Concretely, take τ = 0.1, u_clean = 0.01 (the `acceptable_fnr` bound), u_bad = 0.4, R̄ = 7,
L = 40 and s_b = 0.003:

- **Not (1):** `d ≈ L·s_b·½ = 0.06 > ε = 0.02`. A bad state's middle reading is a coin.
- **Not (3):** the unchecked rate per read is `s_b·u_bad + u_clean ≈ 0.011 < τ/2 = 0.05`.
- **Not (2b):** `B ≈ 0.0084`, `C ≈ 0.07`, and the disagreement-prefix yield is `≤ d·β/L`-ish,
  small here. So `E[badness] ≈ (0.0084·0.4 + 0.07·0.01)/0.078 ≈ 0.05 < 2τ`.

It is not just a failure of the statement. The harvest population's indecision under the current
family is about 0.05, below τ, so the next family can pass it unchanged.

The five-symbol trap's first refused round sits in this regime. Its pass harvest was 154 strings:
142 at the clean, common state c2, 3 at the bad state Q. Its quiet probes were 3–5% unchecked, so
it did not halve. The learner still progressed, because the boundary population grown by replay in
round 2 was 64–67% Q. That growth is outside what the lemma models.

**What does hold (small τ).** Clean items have badness at most `uLo` and bad items at least `uHi`,
so `E[badness] > 2τ` follows from

    (uHi − 2τ)·(bad items)  >  (2τ − uLo)·C.

With `uHi ≥ 4τ`, `uLo ≤ τ`, bad items `≥ Y − C ≥ d/L − C` and `d > ε`, it suffices that

    R̄·u_clean  <  ε/(2L).

That is the `RoundOrHarvest` regime: τ is small enough that clean contamination is negligible.
With `u_clean ≤ τ/10` it holds for `τ < 5ε/(L·R̄)`, no halving needed.

## Step 5: spread — holds up to one sorry

`HarvestSpread` follows from `RoundOutcome`'s yield and harvest-spread parts plus
`D(x[:i] = t[:i]) = L·ν(t[:i]) ≤ L·κ·V`. This is `harvestSpread_of`, still sorried; the argument
is in its docstring.

## Open

- **Freshness (P5).** The pass and every replay read the root's strings. So the noise there is
  pinned, and Step 1's state-level laws hold only for prefixes the pass did not read. The model's
  `pinned` conditioning is the way to state this; nothing here does it yet.
- **Two progress mechanisms in one badness.** A disagreement prefix is credited badness about 1.
  But its state may be decided, and then the FNR gate gives no progress on it; progress comes from
  the split test once its strings are members. `HarvestBad` should probably split into an indecision
  part (judged against τ) and a wrong-edge part (judged by minority share in the leaf).
- **Moderate τ.** Step 4's regime needs one of:
  - a premise `R̄·u_clean ≪ s_b·u_bad`, which holds at signal 0.3, where `u_clean ≈ 5e-5`;
  - a halving trigger that also fires on refusal with a light harvest;
  - a harvest-quality measure that reflects the replay's growth, which made the trap progress.

## Case (2) over every population the round makes (`RoundTrichotomyAll`)

The FNR gate holds the next family to `τ` on every population the round leaves, not just the
harvest. So case (2) is restated over all of them.

**The populations.**
- **The harvest.** `Walked` replays plus `Sifted` replays (a member of leaf `s`'s population
  extended by `c`, harvested when undecided), mixed in proportion to what each found.
- **The per-state populations.** Length-`L` draws that the hypothesis walks to `s` and that the tree
  settles at `s` (`settlesAt`). `StateSource` keeps an aim only where it rests.

**(2a) `PopulationIndecisive`.** Some population P has

    E_{t∼P}[ u(state t) ] > 2τ,

with `u` the current family's state-level indecision. A family held to `τ` on P must then read P's
states differently. `u` has to be state-level, over fresh noise: a population's realised indecision
under the same suffixes is selection-biased. Per-state strings were kept because they were decided,
and harvested ones because they were not. The next family's suffixes give fresh reads.

**(2b) `WrongEdgeHarvest`.** At some leaf `s`, the harvest's disagreement prefixes are at least
σ of what the round's populations place there:

    nH · h_wrong(s)  ≥  σ · (nS · h + nH · h(s)),

where
- `h`, `h(s)` and `h_wrong(s)` are the harvest masses: all items, items at `s`, and disagreement
  prefixes at `s` whose successor states are all read cleanly;
- `nH` and `nS` are the population sizes.

A harvested string the tree places is exactly a disagreement prefix.

For the next round's split test to refuse to call the leaf one state, `σ` must be at least about
`_MIN_DETECTABLE_SPLIT` = 0.1. `NO_SPLIT` fires only when the minority side is too small to hide a
10% split. The test also needs enough members, up to `_MEMBER_LIMIT` = 1500.

Prefixes whose successor is read badly are excluded. Their disagreement comes from the successor's
coin-flip reading, and splitting their own clean leaf does not mend it. So `edgeError`, which
credits them about ½, over-credits them as progress.

**Does it cover step 4's regime?** There, q is rare and badly read (u_q ≈ 0.4), the clean states
are at u_c ≈ τ/10, and q is visited on a share s_b of reads.

- **Harvest (2a).** Undecided items arrive at `q` and at clean states in the ratio
  `s_b·u_q : u_c` (Sifted at edge `(s,c)`: `π(p|s)·u_q : u_c` per provenance; mixing provenances in
  proportion to their finds keeps the overall ratio). Disagreement prefixes have clean states. So
  the harvest's E[u] exceeds 2τ iff about

      s_b · u_q · (u_q − 2τ)  >  u_c · (2τ − u_c),

  which needs s_b > 0.024 at τ = 0.1, u_c = 0.01, u_q = 0.4. Not covered for s_b = 0.003.
- **Per-state (2a).** The hypothesis's edges send q's length-`L` strings to some leaf t. Among the
  aims that settle at t, q's share is

      π̃(q|t) ≈ P_L(q) · (1 − u_q) · P(settles at t | q decided) / P(walk = t, settles at t).

  E[u] > 2τ needs π̃(q|t) > (2τ − u_c)/(u_q − u_c) ≈ 0.49. So q must be about half of its leaf at
  length `L`.

  This is a premise on q's length-`L` mass relative to its leaf, which depends on the hypothesis.
  Its target-only sufficient form is "every badly read state has length-`L` mass comparable to the
  largest leaf", the rejected "common at length `L`" premise.

  ν/V premises do not imply it:
  - V averages over positions, while the population is length-`L` only.
  - For periodic targets (counters of letters mod k), P_L(q) can be 0 with V(q) > 0.
  - For aperiodic irreducible targets, P_L(q) → V(q) once L exceeds the mixing time. Even then
    it bounds the absolute mass, not the share within a leaf.
- **(2b).** The coin-read q produces disagreements only at edges into q, whose prefixes are
  excluded. A decided wrong edge (a decided state merged into the wrong leaf) produces prefixes at
  its own state. Those cover (2b) once that state is about 10% of its leaf's population strings.
  That again needs relative mass, or enough harvest: `nH·h_wrong ≥ 0.1·nS·h`.

**Verdict.** `RoundTrichotomyAll` is still false, in a narrower residual regime. All of these hold
at once:
- some badly read state q (u_q ≥ uHi) with 2ε/L ≲ V(q) ≲ u_c(2τ − u_c)/(u_q(u_q − 2τ)): the gate
  fails through q's coin readings, yet the harvest is clean-dominated;
- q's share of the leaf its strings walk to, at length `L`, is below (2τ − u_c)/(u_q − u_c);
- no decided wrong edge carries 10% of its leaf;
- no halving: Σ_q visit-share · u_q ≤ τ/2.

At defaults this is 0.001 ≲ V(q) ≲ 0.024, with q's within-leaf length-`L` share below about ½.
It is stated as a definition, not a theorem.

**Empirics.**
- In the coordinator's 180-target sweep, every refused round halved or was bad-heavy.
- The three rounds closest to this regime (generated target 20) had u_max 0.08–0.13. That is not
  a coin-read state at u ≈ 0.4, but the middle-band case the gap premise (P4) excludes. They were
  bad-heavy only through disagreement prefixes.
- So no measured round sits in the residual regime. The trap's first refused round is borderline,
  and progressed through the replay-grown harvest.

**Closing the residual.**
- (i) A premise: every badly read state is at least a constant share of the leaf its strings walk
  to, at length `L`. This is hypothesis-dependent, so better phrased as a clustering/stage fact.
- (ii) Algorithmic: aim a per-state population at each (leaf, letter) edge whose members·c are
  undecided at above-baseline rate. That is a `Sifted` source kept as its own population, so it is
  judged at `π(p|s)u_q : u_c` rather than diluted by the global mix. This turns the per-provenance
  concentration, which exists, into a per-population one, which the gate sees.

## With #408's edge populations (`RoundTrichotomyEdges f a r`)

**What #408 does.**
- **Which edges.** An edge read is `Read(X, c)`: draws of a population source `X`, extended by `c`,
  sifted. The pass makes these only in edge resolution (`decisive_target`), which harvests the
  leading run of a leaf's members whose `·c` is undecided, stopping at the first that places. Reads
  are keyed by source and letter, not by leaf.
- **The test.** A read that harvested at least 3 strings is replayed 32 times. It is held apart as
  `("edge", n)` when the share of replays that harvest exceeds 2 × (the pass's unchecked share per
  read) × (the replays' reads).
- **What a replay draws.** It draws from the whole source `X`, not from one leaf. So the
  concentration on a badly read q is `π_X(p) = X(state ∈ pred_c(q))`. Only per-state sources
  concentrate: the source of leaf `s` gives `π(p|s)`, which is about 1 when `p` has its own leaf.
  The uniform source gives P_L(p), no better than the pass's own baseline.

**Why the factor 2 holds, and what it should be.**
- **Edge composition.** Per replay, a string is harvested with chance `π·u_q + (1−π)·c_e`, where
  `c_e ≤ r·uLo` is a clean string's chance of being undecided (`r` nodes deep, each undecided at
  most `uLo`).
- **The share lemma** (`share_bad_indecisive`, proved): if at least θ of the population's
  undecided mass sits on states with `u ≥ uHi`, then

      E_P[u] = ∫u²/∫u ≥ θ·uHi.

  So (2a), `E_P[u] > 2τ`, holds once `θ > 2τ/uHi`.
- **Selecting at f × the clean bound.** If the edge is held apart when its undecided rate exceeds
  `f·r·uLo` (`EdgeSelected`), then `θ ≥ 1 − 1/f`. So the condition on f is

      f  >  uHi / (uHi − 2τ),

  which is `f > 2` at the gap `uHi = 4τ`. #408's 2 is exactly the boundary there, and any f > 2
  works.
- **The baseline must be the clean bound, not the pass's average.** #408 compares against
  `unchecked/reads`, which can sit below `uLo` when the pass happened to read only very clean states.
  An edge reading clean states at up to `uLo` then passes the test at θ < ½. That is a false split,
  harmless to the trichotomy but not to the population mix. The principled baseline is the known
  bound `uLo` = `acceptable_fnr` per read.

**The number of replays (`EDGE_PROBES`).** Each replay is a Bernoulli trial with
`p = π·u_q + (1−π)·c_e`, and the test asks whether `p > T = f·r·uLo`. With K edges tested per round
(at most |leaves|·|α|) and the split decision's share of the round's error budget `δ_e`,
multiplicative Chernoff on both tails gives:

    detect an edge with p ≥ (1+γ)T :  n ≥ 2(1+γ)·ln(2K/δ_e) / (γ²·T)
    never select one with p ≤ (1−γ)T :  n ≥ 3·ln(2K/δ_e) / (γ²·T)

The q-edge sits far above T. At π ≈ 1 and u_q = 0.4, against T = 2·5·0.01 = 0.1, that is γ ≈ 3.
Its miss chance at n replays is exp(−n·(p − T)²/(2p)), so n ≥ 2p·ln(K/δ_e)/(p − T)².

At K = 20 and δ_e = 0.01 that is n ≈ 8·ln(2000)·0.4/0.09 ≈ 270 for a marginal edge (γ = ½ around
T). For the clear q-edge it is n ≈ 2·0.4·7.6/0.09 ≈ 68. #408's 32 replays give a miss chance of
about exp(−32·0.09/0.8) = 0.027 per clear edge, but no control on marginal ones.

**The minimum harvest (`EDGE_MIN_HARVEST`).** It plays no part in correctness; it only saves
replays, and it costs detections.
- The pass's count for a read is a sum of leading undecided runs, one per member configuration of
  each leaf. A configuration changes only on `add_first` or a split.
- With one configuration, `count ≥ m` has chance about `(π·u_q)^m`: 0.064 for m = 3 at π·u_q = 0.4.
  So a q-edge with `u_q` well below 1 is mostly never tested. The trap's e1, at u ≈ 1, is tested
  every time.
- Principled value: m = 0. Test every (per-state source, letter) pair, at a cost of K·n replays per
  round, with K ≤ |leaves|·|α|. If cost forces a gate, m = 1 misses with chance `(1 − π·u_q)^{K_conf}`.

**Parametric statement.** `RoundTrichotomyEdges f a r` takes as parameters
- `f`, the selection factor;
- `a`, the clean per-node bound (`uLo`);
- `r`, the read depth.

Its edge disjunct is `EdgeSelected ∧ EdgePopulationIndecisive`. The conditions they must meet are
irreducible: `f > uHi/(uHi − 2τ)` and the gap premise with `uLo = a`. With them, `EdgeSelected`
implies `EdgePopulationIndecisive` by `share_bad_indecisive`. The replay count n enters only the
error term above, `K·exp(−n·(p−T)²/(2p))`. It stays a definition, not a theorem, because of the
residual below.

**Residual regime with #408 (why it's not a theorem).** The edge test reaches a coin-read q only
through a per-state source concentrated on a predecessor p of q. The regime left open has every
predecessor p of every rare, badly read q with

    π(p|s)  ≤  f·r·uLo / u_q     (≈ 2·5·0.01/0.4 = 0.25 at defaults)

in every leaf s it settles in, at length L. The other conditions are the same as before: the gate
fails, there is no halving, and the harvest is clean-dominated. That is, p is merged into leaves
dominated by other, common states, distinguished from them only by suffixes that pass through q.
That is natural: such a suffix is coin-read, so the tree cannot separate p from them. ν/V-type
premises do not bound π(p|s), for the same reason as before. It is a share within a
hypothesis-dependent leaf, at length L.

**What would close it: iterate the concentration.**
- The strings an edge population holds are `x·c` with x a p-string, in the ratio `π·u_q : c_e`
  against clean ones. So the prefixes `x` of its undecided finds are concentrated on p at

      π' = π·u_q / (π·u_q + c_e).

- Using that prefix population as the next round's source for edge `c` multiplies p's odds by
  `u_q/c_e` (8 at defaults) per round. So the edge test reaches the threshold within
  `log_{u_q/c_e}(1/π)` rounds, whatever π starts at. Concretely, #408 would also keep, for each
  selected-or-tested edge, the population of prefixes whose `·c` came out undecided, as a source for
  later edge tests.
- With that, the residual reduces to `π(p|s) > 0`, and the trichotomy would hold over a bounded
  number of rounds rather than per round. That is still a multi-round statement, not the
  single-round one stated here.

**Recommended #408 constants, from the above.**
- factor `f = 2(1 + γ)`, with γ = ½ (so f = 3) at the gap `uHi = 4τ`; any `f > uHi/(uHi − 2τ)` is
  correct;
- the baseline is `acceptable_fnr` per node read, not the pass's unchecked share;
- `EDGE_PROBES ≈ 2p*·ln(K/δ_e)/(p* − T)²`, about 70 for the clear q-edge, where p* is the smallest
  edge rate to be detected;
- `EDGE_MIN_HARVEST = 0`: test every per-state source and letter.

## Rollover chains: a bounded-rounds trichotomy

**The rule as implemented on #408.** Each round, every (per-state source, letter) edge and every
live chain is replayed. Its undecided rate per node read is tested sequentially, at level δ_e/K,
against the clean bound `a` = `acceptable_fnr` per read:
- **above `f·a`:** promote to an `("edge", n)` population;
- **between `a` and `f·a`:** roll over. The chain becomes a `RejectionSource` over its parent that
  keeps x iff x·c is undecided under this round's frozen sifter, and it persists as a candidate;
- **at most `a`:** drop.

**One link's effect** (`rolledLaw`, `rolled_odds`, proved). Suppose every link reads x·c afresh.
Then x survives k links with chance `∏_j u_j(state(x·c))`, so the chain draws from its source
weighted by that product. If badly read strings have `u_j ≥ uHi` and clean ones `u_j ≤ r·a` at every
link, the bad part keeps at least `uHi^k` of its mass and the clean part at most `(r·a)^k`, so

    odds_k  ≥  odds_0 · (uHi / (r·a))^k,        odds_0 = π / (1 − π).

**Rounds to promotion.** Promotion needs a bad share of at least `T/uHi`, with `T = f·r·a`. So
(`rolloverRounds`; `rollover_promotes` holds the arithmetic, sorried)

    R = ⌈ log_{uHi/(r·a)} ( T·(1 − π) / ((uHi − T)·π) ) ⌉ + 1,

with conditions `1 ≤ f` and `T < uHi`. At f = 3, r = 5, a = 0.01 (T = 0.15, base 8) and uHi = 0.4:
- π = 0.05 gives R = 3;
- π = 0.01 gives R = 3;
- π = 0.001 gives R = 5.

**The independence premise** (`FreshLinks`). For each x, no string that link j's family reads on
x·c (the strings x·c·m·v, for midfixes m on its path and suffixes v ∈ F_j) was read by an earlier
link. Under string-keyed noise this, and only this, makes the links' outcomes independent given x's
state.

Overlap breaks it. A clean x whose x·c was undecided at link j through its shared reads stays more
likely undecided at link j+1. In the extreme, an identical family and tree, its survival is 1 from
then on, and the odds stop growing after one link.

In the Python, `suffix_pool` persists and families re-select many of the same suffixes. The root's
midfix is ε, so x·c·v repeats for every shared v. The premise is likely false as implemented. The
fix is cheap: each link reads x·c only with suffixes no earlier link of that chain used, for example
the family minus the chain's used suffixes, or a fresh family-size draw from the cluster's group.

**Bounded-rounds trichotomy (notes only).** Take R consecutive rounds, none of which passes the
DFA/DT check within ε. Assume:
- (P1)–(P4), with `uLo = a`;
- `f > uHi/(uHi − 2τ)` and `f·r·a < uHi`;
- `FreshLinks` for every chain;
- a badly read state q (u ≥ uHi under every family of the stretch);
- a predecessor p of q that some per-state source holds at share π > 0 and whose chain survives
  level 0 (see residual).

Then, except with probability at most R·δ_e from the sequential tests plus R·(1−ε)^patience from the
pass, within `R = rolloverRounds f a uHi π r` rounds one of these happens:
1. some round halves;
2. some round leaves (2a) or (2b), including a promoted edge population, which is (2a) by
   `share_bad_indecisive`;
3. some round's family reads q with u < uHi, which is progress in `Termination`'s count of
   undecided-state sets.

It is not a Lean theorem on this branch. The multi-round learner, with families changing round to
round, lives in #400's `Learner.lean`. This branch models one round.

The Lean pieces:
- **Definitions:** `rolledLaw`, `rolloverRounds`, `FreshLinks`.
- **Proved:** `rolled_odds`.
- **Sorried:** `rollover_promotes`, the arithmetic from `rolled_odds`, believed true.

**Residual regime: the drop rule.** A chain is dropped when its rate is at most the clean bound
`r·a`. The rate is `π_k·u_q + (1 − π_k)·c_k`, where `c_k` is its clean strings' actual rate. When
clean strings are much cleaner than the bound (`c ≪ r·a`, typical at signal 0.3), a chain with

    π(p|s) · u_q  +  (1 − π)·c_s  ≤  r·a      (≈ π ≤ r·a/u_q = 0.125 at defaults)

is dropped at level 0, though it would have been enriched by `u_q/c ≫ uHi/(r·a)` per link. So the
residual is: every predecessor p of the badly read q has `π(p|s)·u_q + (1 − π)·c_s ≤ r·a` in every
per-state source.

**Closing it.** Drop a chain only when its rate fails to rise over its parent's: a sequential test
of `rate_k > rate_{k−1}`, not of `rate_k > r·a`. Under fresh links a chain with any bad mass has a
strictly rising rate, since its bad share grows; one with none stays flat. So this keeps exactly the
chains that carry bad mass, and R is then as above with the base `u_q/c` in place of `uHi/(r·a)`.
The cost is that chains with tiny π live about log(1/π) rounds. Live chains stay at most K·R.

**What else changes across rounds.** The family changes each round, so `u_j` and the clean
rates `c_j` are per link. That is why `rolled_odds` takes them per link, with only the bounds `uHi`
and `r·a` uniform. A family that reads q below uHi at some link is outcome 3 above.

## One round, four outcomes (`RoundTetrachotomy f a uHi δ r`)

**The statement.** But for δ, a round ends with
1. agreement within ε; or
2. a population the next gate must act on: (2a), (2b), or a promoted edge population; or
3. halving; or
4. `ChainAdvances`: some chain source X has positive mass on `badAlong c`, the draws whose `·c` is
   read at least `uHi` undecided. X is a live chain carried into the round, or one of this round's
   per-state populations. The chain carried past this round's link (`rolledLaw X c [link]`) has
   that mass multiplied by at least `uHi` and the rest's by at most `a`, so the odds grow by at
   least `β = uHi/a`, per node read.

Premises: `Valid`, the gap premise with `uLo = a`, `f > uHi/(uHi − 2τ)` and `f·r·a < uHi`.

**What is proved.** `chainAdvances_of_mass` (only `propext`, `Classical.choice` and `Quot.sound`):
under the gap premise, (4) holds whenever some source has positive `badAlong` mass. The proof is
`rolled_odds` with one link plus `withDensity_real_eq`. Fresh reads for this round's link are what
make `rolledLaw` the chain's actual law. That is a modelling bridge, not a hypothesis of the Lean
statement, since the Lean law is defined as the fresh-read one.

**Why `RoundTetrachotomy` is still a definition.** A round outside (1)–(3) that has no such source
is the residual: every predecessor p (along any letter c) of every badly read state has zero mass in
every live chain and every per-state population. Since per-state aims are length-`L` strings the
hypothesis walks to a leaf and the tree settles there, that means:
- (a) **periodic targets** whose predecessors never occur at length `L`; or
- (b) **predecessors the hypothesis always misplaces at length `L`.** It walks them to one leaf and
  the tree settles them at another, so neither leaf's source keeps them. That is a disagreement at
  the predecessor itself.

Closing (a) needs per-state sources at lengths `L` and `L+1` (or `L..L+period`). Closing (b) needs
the per-state source to keep aims by where they settle, not where they were walked. #408 already
identifies chains by source objects, so a "settled at s" source is the natural parent.

**(4) is weak on its own; the content is in the count.** (4) holds whenever any source carries
predecessor mass, regardless of whether anything accumulates. What turns it into the bounded-rounds
bound:
- **Rule.** Drop a chain only when its rate fails to rise over its parent's. Under a fresh link,
  the rate of a chain with bad mass rises strictly: the size-biased rate exceeds the mean by
  `Var(u)/E[u] > 0`. So with a power-one sequential test, which never concludes "no rise" against a
  rising rate, a chain with bad mass is never dropped. One without bad mass is kept with chance at
  most `δ_e/K` per test.
- **Potential.** For each edge into a badly read state (a predecessor source and letter), let
  `Φ_e = log_β(odds of its highest live chain)`, capped at the promotion odds
  `log_β(T/(uHi − T))`. Each round outside (1)–(3) every live chain with bad mass advances, so
  `Φ_e` rises by at least 1. A chain reaching the cap is promoted: (2).
- **Bound.** A stretch of rounds outside (1)–(3) therefore lasts at most

      R = rolloverRounds f a uHi π_min r

  rounds after the first chain on a bad edge starts, where `π_min` is that chain's starting
  predecessor share. Termination's counting of undecided-state sets covers the stretch ending
  because q stops being badly read.
- **Number of chains.** At most the edges into badly read states, plus false keeps, which are
  δ-charged at `δ_e` per round.
- **With a capped test** (n replays), a rising chain whose rise `Var(u)/E[u]` is below the
  resolution `≈ √(p·ln(K/δ_e)/n)` can be dropped. That adds a residual `π ≲ p·ln(K/δ_e)/(n·u_q²)`
  at the start of a chain.

## The ν root (`nuRoot`): the coin-read residual is gone

**The source.** `nuRoot D L` draws x from D and i uniformly below L, and returns x[:i]. So y comes
out with chance ν(y) = D(x starts with y)/L, for |y| < L. `RoundTetrachotomy` now quantifies over
`nuRoot S.D S.L :: live ++ per-state populations`.

**Proved** (only `propext`, `Classical.choice` and `Quot.sound`):
- `nuRoot_pos`: a string x with |x| < L that D begins with has positive ν-root mass.
- `chainAdvances_of_visited`: under the gap premise, outcome (4) holds whenever some string read
  before position L leads by some letter c to a badly read state. The ν root then has positive
  `badAlong c` mass, and `chainAdvances_of_mass` applies.

The old residual (a) periodic predecessors and (b) misplaced predecessors is gone. Neither
condition matters to the ν root: it holds every prefix that reads pass through.

**A badly read state behind the gate's failure has a visited predecessor.** The gate's walk starts
where the middle reading puts ε, so it is in sync at position 0 by definition. A disagreement needs
the middle reading of some x[:j], j ≥ 1, to leave the walk. If that comes from a coin-read state
(vote near the middle, so u ≥ uHi), then x[:j−1] is a visited predecessor with j − 1 < L, and
`chainAdvances_of_visited` gives (4).

A state reached only at position 0 (the initial state, never re-entered) cannot cause a
disagreement. It is also transient, which the premises exclude.

**What still is not covered: decided wrong edges.** A middle reading leaves the walk in one of
three ways:
- **(i) A coin-read state.** Covered: (4).
- **(ii) A clean state's middle reading flipped.** It needs the vote to cross the middle, a full
  margin beyond the band. Its chance η is negligible (~1e-14 per read by the family's sizing), and
  charged to δ as `L·η` per draw.
- **(iii) A decided wrong edge.** Two clean states p1 and p2 share a leaf, and p2's c-successor
  differs from the edge's target.

(iii) is outside (4), because `badAlong` is about indecision, and clean successors have u ≤ uLo.
It is outside (2b) whenever p2's disagreement prefixes are under σ of what the round's populations
place at the leaf. The pass meets the edge, the split test answers `NO_SPLIT` (p2 is under about 10%
of the members), and #403 harvests the prefix. It is outside (3) when the crossings are checked.

So `RoundTetrachotomy` is false in exactly this regime:
- d > ε arises only through decided wrong edges;
- each wrong-edge state is under σ of its leaf's population strings;
- no halving.

It stays a definition.

**What would close it: a disagreement chain from the same ν root.** It keeps x when the middle
reading of x·c lands off the hypothesis's edge from x's leaf. For p2-strings that happens with
chance about 1; for p1-strings with chance about η. So one link enriches p2's odds by about 1/η,
and the promoted population is almost pure p2. As a population it puts p2 at its leaf at its own
size, so (2b) holds once that size is at least σ/(1 − σ) of the leaf's other members, at most
`_MEMBER_LIMIT`. A `RoundTetrachotomy` whose (4) also counts this chain is then true modulo the η
term. Its (4) would be "a chain with bad or wrong-edge mass advances", proved like
`chainAdvances_of_mass`, with the edge-disagreement chance in place of u.

**R in terms of V.** The ν root's starting share for edge c is its `badAlong c` mass:

    π_0 = Σ_{|y|<L, y·c → badly read} ν(y) = V_{<L}(pred_c(badly read states)),

V restricted to positions below L. So the stretch bound is

    R = rolloverRounds f a uHi (V_{<L}(pred_c(Q_bad))) r,

with the largest V over letters c. The ν root's level-0 rate is about the pass's baseline, so it
relies on the rise test, not the f·a bar, to be kept.

## The disagreement chain, and (4') (`RoundTetrachotomyBoth`)

**The new pieces.**
- `edgeDisagreeProb x c`: the chance, over fresh noise, that the middle reading of x·c lands off the
  hypothesis's edge out of where the middle reading puts x.
- `EdgeGapPremise η wHi`: every edge read lands off its edge at most η or at least wHi of the time.
  For two clean states the readings are decided, so `ed` is about 0 or about 1, except for
  middle-reading flips of chance about η. A coin-read successor gives about ½, which counts as
  wrong once wHi ≤ ½.
- `ChainAdvancesBy X g lo hi`: a chain whose link keeps x with chance g(x) advances. X holds
  strings with g ≥ hi, and the link multiplies their mass by at least hi and the rest's by at most
  lo.
- **(4') `ChainAdvancesEither`:** some source advances an indecision chain (g = u(state(x·c))) or a
  disagreement chain (g = `ed`(x, c)).

**Proved** (only `propext`, `Classical.choice` and `Quot.sound`):
- `chainAdvancesBy_of_mass`: the generic advance, from `rolled_odds` with one link.
- `chainAdvancesEither_of_visited`: with the ν root among the sources, (4') holds whenever some
  string read before position L has u(x·c) ≥ uHi or `ed`(x, c) ≥ wHi.
- `visited_clean_of_not_advances`: a round that advances no chain reads every edge out of every
  visited string cleanly. Its successor has u < uHi, and the edge read lands off at most η.

**What is left: one probabilistic lemma.**

    P( edge gap ∧ the gate fails by more than ε ∧ every visited edge read is clean ) ≤ δ

Given it, `RoundTetrachotomyBoth` follows from `visited_clean_of_not_advances`. In fact (1) ∨ (4')
alone does; (2) and (3) are not needed for the one-round statement.

**Why the lemma is believed true, with δ = L·η/ε + η·N_pass.**
- A draw the gate counts against the hypothesis has a first position j ≥ 1 where the middle reading
  leaves the walk. That is the realised event behind `ed`(x[:j−1], x_j).
- If the gate's reads are fresh relative to the hypothesis, the expected disagreement mass is at
  most `Σ_j E[ed] ≤ L·η`, and Markov gives `L·η/ε`.
- The hypothesis is built from the same noise, at the strings the pass read. At each pass-read pair
  the realised reading deviates from its fresh chance at most η, and the union over the N_pass pairs
  the pass read adds `η·N_pass`. Off those pairs the gate's reads are fresh.

**Why it isn't stated in Lean yet.** Its δ needs N_pass, the number of strings the pass reads.
`Pass.lean` counts node reads only over the quiet window (`PassState.reads`), not the whole pass's
read set. That is the same missing reads model that keeps `RoundOrHarvest`'s overlap term unstated.

So `RoundTetrachotomyBoth` is not yet a theorem. There is no residual regime of targets any more:
what remains is that one lemma and the reads model it needs.

**The counting note, updated.** With both chains rooted at the ν prefixes:
- An indecision chain on edge c starts at `π₀ = V_{<L}(pred_c(badly read))`.
- A disagreement chain starts at `π₀ = V_{<L}(strings whose c-edge is wrong)`. It enriches by
  about `wHi/η` per link, so in practice it promotes in one round. Its promoted population is (2b)
  once its size is at least σ/(1 − σ) of the leaf's other members.
- R = `rolloverRounds` with base `uHi/a` for indecision chains and `wHi/η` for disagreement chains.

## `round_tetrachotomy_both`: a theorem, modulo one sorried step

**The statement** (`Proofs/Round.lean`). For a round with
- `Valid`;
- the gap premise (a, uHi), with 0 ≤ uHi and 0 ≤ wHi;
- ε > 0;
- D supported on length-L strings;
- `MidFlipPremise φ`: a node read of a string whose state is not badly read lands on one side of
  the middle but for chance φ;
- `BadVisited`: every badly read state is reached by a letter from a string read before
  position L;
- η + 2(N+2)φ < 1,

we have

    P( edge gap ∧ ¬(1) ∧ ¬(2) ∧ ¬(3) ∧ ¬(4') )  ≤  (L+1)·2(N+2)·φ / ε  +  passReadBound·φ,

where `passReadBound = (|seed| + N(L+1))·(1+|α|) · (N+2)·(1+|α|)` bounds the pass's node reads.

**The proof.**
- **Proved:** the event is inside the event of `gate_flip_bound`. `visited_clean_of_not_advances`
  makes every visited edge read clean. `BadVisited` turns "every visited successor is not badly
  read" into "no state is badly read".
- **Sorried (`gate_flip_bound`):**

      P( every visited edge read off at most η ∧ no badly read state ∧ the gate fails by more than ε )
        ≤ (L+1)·2(N+2)·φ/ε + passReadBound·φ.

**Why `gate_flip_bound` is believed true.**
- A draw the gate counts against the hypothesis first leaves its walk at some j ≤ L, through the
  middle readings of x[:j−1] and x[:j].
- Every state is read cleanly, so each node read lands on its state's side but for chance φ. The
  sides' readings agree with the edge, since otherwise `ed` ≥ 1 − 2(N+2)φ > η.
- So some node read on those two paths flipped. Each path is at most N + 2 deep, because each probe
  splits at most once.
- **Fresh reads.** Node reads the pass did not make are fresh given the pass, so the expected
  disagreement mass is at most (L+1)·2(N+2)·φ. Markov gives the first term.
- **Pass reads.** The pass's node reads are at most `passReadBound`. Every string it sifts is a
  seed string or a probe prefix, extended by at most a letter. Every midfix it reads at is the final
  tree's, at most N + 2 of them, since trees only grow, preceded by at most a letter (the split
  test's distinguishers). Read adaptively, each is fresh when read, so the chance any flipped is at
  most `passReadBound`·φ.

**What the sorry contains.**
1. **Structural:** the pass is determined by its node reads at that set. This is an induction over
   `probeStep`/`settle`, with trees only growing. It is not done, and it is not analytic.
2. **Probabilistic:** the adaptive-read independence, the analogue of `probe_couple` on #400.
3. **The Markov and union-bound arithmetic.**

So the step is not a pure analytic sorry: (1) is structural work still owed.

**Not done.**
- The pass's read set as a Lean `Finset` with its card bound. The bound above is argued in the
  sorry's reason, not built.
- `RoundOrHarvest` in the same form. Its overlap term needs the same read set and the same
  adaptive independence.

## Discharging the sorries

**`rollover_promotes`: proved** (8f8eae9). From the list-product bounds `list_prod_ge` and
`list_prod_le`, and `withDensity_real_eq`. It needs `[IsProbabilityMeasure μ]`, the setting's
standing assumption, so that `stateIndecision` lies in [0, 1].

**`harvestSpread_of`: proved** (f7f4612). From `round_outcome_holds`'s yield and harvest-spread
parts, plus `D(first i letters are t's) = L·ν(t[:i])` for i ≤ |t| (`take_eq_iff_prefix`) and the κ
premise.

**`gate_flip_bound`: not provable as stated. Stopped here.** Its argument needs a premise that does
not hold for the Python's families: that distinct node reads use disjoint noise bits.

- **Where bits are shared.** A node read of a string z is the vote over the bits at z·v, for
  v ∈ F. Reads of z and z′ share a bit iff z·v = z′·v′ for some v, v′ ∈ F. With |z| < |z′| that
  means z′ = z·w and v = w·v′: some v′ ∈ F is a proper suffix of some v ∈ F.
  - That is impossible when F is suffix-free, for example equal-length suffixes.
  - It is possible as soon as ε ∈ F. The Python's family is clustered around the empty suffix,
    which is always in it. Then the read of z = z′·v′ shares its ε-bit, which is z itself, with the
    read of z′ through v′.
- **Both halves of the bound use disjointness.**
  - **The pass half.** "Each pass read is fresh when it is read, so P(any pass read flips) ≤
    passReadBound·φ" needs the newly read string's bits to be unseen. With shared bits, the pass
    chooses its next read, for example the new midfix after a split, from outcomes that share bits
    with it. Conditional on what it has seen, a new read can flip with chance well above φ.
    Splits are triggered by readings that differ, so there is selection toward flips.
  - **The gate half.** "Gate reads off the pass's are fresh given the pass" needs disjointness
    too. With ε ∈ F, a gate read's own bit x[:j]·m can be a pass read's bit z′·v′.
- **What is still true.** A fixed (non-adaptive) union bound holds regardless. The problem is only
  adaptivity combined with shared bits. So the bound is not believed to hold for every F. With
  ε ∈ F it can fail by selection on shared bits, whatever its exact size.

**Options (a decision for the user).**
1. **A premise:** distinct node reads use disjoint bits, or F \ {ε} suffix-free with the ε-overlap
   handled separately. With F suffix-free, `gate_flip_bound` is believed true as stated. The Python
   violates it through ε ∈ F.
2. **An overlap term in δ.** With ε ∈ F and the other suffixes of length ℓ, an overlap needs a read
   string to end in a whole family suffix. For random probes that has chance about
   `passReadBound · (N+2) · |F| · |α|^{−ℓ}`. Adding it changes the statement.
3. **An algorithm change:** read the gate, or the pass, with ε left out of the vote, so that F is
   suffix-free.

**The structural part was not built**: the pass's node-read set as a `Finset`, the determination
lemma, and the `passReadBound` size bound. It is only worth building once the premise question is
settled. The covering argument in `gate_flip_bound`'s docstring checks every read site (sift,
`firstDisagreement`, `tally`, `decisiveTarget`, `edgeMisses`, `anchorMisses`). Every one is
`b·e·m`: b a seed string or probe prefix, e ∈ {1} ∪ α, m a midfix of the final tree. Trees only
grow, and every midfix is a letter followed by an earlier midfix.

## Option 4 (shared bits bounded adversarially): does not work on its own

The idea was a bound k on how many of a read's |F| bits other reads also read, independent of
the run, so that a read's flip chance is at most φ_k: Hoeffding with the margin cut by k/|F|.
No such k exists as a function of F and the midfixes, because of ε ∈ F.

- **What sharing is.** A read of g and a read of h ≠ g share a bit iff g·v = h·v′ for some
  v, v′ ∈ F. In the Python, F = {ε} ∪ F′ with F′ the sampler's draws, all of length L
  (`_draw_cohort` → `UniformSampler.sample`), so F′ is suffix-free. Then the only solutions are
  through ε:
  - h = g·v for some v ∈ F′ (h's own bit is g's bit through v), or
  - g = h·v for some v ∈ F′ (g's own bit is h's bit through v).
- **So k(g) = [g ∈ h·F′ for a read h] + #{v ∈ F′ : g·v is a read string}.** The second count is
  not bounded by F and the midfixes. Whether g·v is read depends on which probe prefixes and
  seed strings the run happened to read, and in the worst case it reaches |F′|. Any bound on it is
  a bound on the chance that one read string is another extended by a whole family suffix,
  over the probes and the gate's draws. That is option 2's overlap term.
- **Once those events are excluded, k = 0.** With F′ suffix-free, ε-sharing is the only sharing,
  so excluding the overlap events leaves no shared bits at all. With the Python's families,
  option 4 reduces to option 2 and adds nothing.

**What the statement becomes under option 3** (ε left out of the vote; a Python change, not
made). Reads vote over F′ only. F′ suffix-free makes distinct read strings use disjoint bits:
g·v = h·v′ with v, v′ ∈ F′ forces g = h, since otherwise one of v, v′ is a proper suffix of the
other. That is the premise the original argument needed, so `gate_flip_bound` keeps its
statement with φ unchanged, plus one premise on the family:

    VoteSuffixFree F′ : ∀ v v′ ∈ F′, v ≠ v′ → ¬ (v′ is a suffix of v)

which is irreducible (a property of the input family) and true of the Python's F′. The bound
stays `(L+1)·2(N+2)·φ/ε + passReadBound·φ`. What it still owes is unchanged: the pass's
read set as a `Finset`, that it determines the pass, and adaptive independence (each new read
string's bits unread, so its vote is independent of the past). `ε` would still anchor the
clustering; only the reads would leave it out.

## The edge gap need not be assumed (`Proofs/EdgeGap.lean`)

`round_tetrachotomy` is `round_tetrachotomy_both` with the `EdgeGapPremise … s.hyp` conjunct gone
from the bounded event. It takes two premises on the edge gap's constants instead:

    2(N+1)·φ ≤ η    and    wHi ≤ 1 − 2(N+1)·φ.

It rests on `edgeGap_or_advances`, proved with only `propext`, `Classical.choice` and
`Quot.sound`:
- **Every state read cleanly gives the gap.** Read every node on its likelier side
  (`majRead`, `majPath`). With every state read cleanly, a node read leaves its likelier side
  with chance at most φ (`flip_le`). So the middle reading of x leaves `majPath x` with chance at
  most φ times the path's reads (`midPath_ne_majPath_le`). An edge's chance of landing off its
  edge is then at most ℓφ or at least 1 − ℓφ, with ℓ the two paths' reads
  (`edgeDisagreeProb_bimodal`). Paths have at most `depth` reads (`route_length_le_depth`), and a
  round's tree is at most N + 1 deep (`roundEnd_depth_le`, from `splitAt_depth_le` and
  `probeStep_tree`). So the gap holds at η, wHi.
- **Otherwise an indecision chain advances.** Some state is read badly, `BadVisited` reaches it
  from a visited string, and the ν root advances its chain (`chainAdvancesBy_of_mass`,
  `nuRoot_pos`).

`round_tetrachotomy` still rests on the sorried `gate_flip_bound`.

## No gap premise: the ladder (`round_tetrachotomy_ladder`)

`GapPremise` (every state's indecision ≤ a or ≥ uHi) served one purpose: making an indecision
chain's link enrich its bad states' odds by a fixed ratio. It isn't needed.
- **A gap is guaranteed.** g(x) = u(state(x·c)) takes at most |Q| values. So any monotone ladder
  θ 0 ≤ … ≤ θ K with K > |Q| rungs has a rung i that no value falls strictly inside
  (`exists_rung_gap`, a pigeonhole). That gives g ≤ θ i or g ≥ θ (i+1) for every draw, with no
  premise.
- **Outcome (4'')** (`ChainAdvancesLadder`): an indecision chain advances at some rung, its draws
  reading at least θ (i+1) multiplied by at least that, the rest by at most θ i. Or a
  disagreement chain advances, as in (4').
- **`round_tetrachotomy_ladder`** is `round_tetrachotomy` with `GapPremise` replaced by the ladder
  conditions:
  - θ monotone, with 0 ≤ θ 0;
  - |Q| < K;
  - θ K ≤ uHi.

  The error bound is the same, there's no extra δ term, and it rests only on the sorried
  `gate_flip_bound`. `edgeGap_or_advancesLadder` and `exists_rung_gap` use only `propext`,
  `Classical.choice` and `Quot.sound`.

**What the user's continuous argument needs, and what it doesn't.**
- **Not needed:** a φ(u) term. States read below uHi are already covered by `MidFlipPremise`
  (one φ for every state under uHi), which `GapPremise` also needed. Dropping the gap adds no
  premise and no δ term.
- **The cost is the per-link ratio.** With θ i = θ 0 · β^i, β = (uHi/θ 0)^{1/K}, a link
  multiplies a chain's odds by at least β, not uHi/a.
- **For counting, θ 0 should be the promotion rate T = f·r·a.** Then a promoted chain's population
  reads at least T, which is outcome (2). States below T stay covered by `MidFlipPremise`.
  - At uHi = 0.4, T = 0.15 and |Q| = 10, that's β = (0.4/0.15)^{1/11} ≈ 1.09.
  - So promotion from a share π takes about ln(1/π)/ln β rounds, about 80 at π = 0.001.
  - Under `GapPremise` it was about 3 at uHi/(r·a) = 8.
  - That's a real slowdown in the bound, not in the algorithm: the algorithm is unchanged; the
    bound just can't assume the states' indecision is bimodal.

## `gate_flip_bound`: proved under a suffix-free vote family

With the vote family suffix-free (option 3, `SuffixFree (F ∪ train F)`: no suffix in it ends
another), distinct read strings use disjoint bits, and `gate_flip_bound` is proved. The
tetrachotomy theorems that used it (`round_tetrachotomy_both`, `round_tetrachotomy`,
`round_tetrachotomy_ladder`, now in `Proofs/GateFlip.lean`) have no sorry left.

**New premises,** on all four theorems:
- `SuffixFree (S.F ∪ S.K.train S.F)`: the property disjointness needs. The pass reads through
  the training half too, which isn't required to lie in `F`.
- `0 ≤ φ`: without it the old statement was false. With φ < 0, `MidFlipPremise` makes every
  state badly read, the event is empty, and the bound is negative.
- `[IsProbabilityMeasure S.D]` on `gate_flip_bound` itself. The others get it from `S.Valid`.

**Structural part** (`Proofs/PassReads.lean`):
- **What the pass reads.** Every read is `b·e₁·e₂·m`: `b` a seed string, probe prefix or witness,
  each `e` empty or a letter, `m` a midfix of the tree before the probe. The congruence lemmas
  (`probeStep_congr`, `settle_congr`, …) show two oracles agreeing there take the pass through the
  same probe.
- **Phase by phase** (`phase_congr`, `phase_determined`): the state after `k` and `k + 1` probes is
  decided by the bits at `nodeReads` against the tree after `k`.
- **The whole pass** (`roundEnd_determined`): decided by the bits at what it reads against its
  final tree, a `Finset` of at most `passReadBound` strings (`card_roundReads`).

**Probabilistic part** (`Proofs/GateFlip.lean`):
- **`cell_bound`.** Partition by the cell (value of `T`, pattern of its bits). A cell decides `Z`
  and is independent of any string's bits off `T` (`indep_noiseAlg` with `disjoint_vBits`), so
  the chance a string of `Z` off `T` flips is at most `M·φ`.
- **`passFlip_le`.** The pass's own reads, one probe at a time. The strings first read against
  the tree after `k` probes are decided by the bits read before and are fresh, at most
  `|bases|·(|α| + 1)²` per probe. That's `(N + 1)·|bases|·(|α| + 1)²·φ ≤ passReadBound·φ`.
- **`freshFlip_le`.** A draw's prefixes' likelier paths, off the pass's reads. At most
  `(L + 1)(N + 1)` strings, given the pass, so `(L + 1)(N + 1)φ`.
- **`flip_of_disagree`, deterministic.** With every visited edge read on its edge
  (`majPath_step`), a disagreeing draw has a flip on some prefix's likelier path.
- **`gate_flip_section`.** Markov over the draws.
- **`gate_flip_bound`.** The sum over probe draws (`prod_le_tsum`); a positive-mass probe has
  length `L`.

The bound is unchanged: `(L+1)·2(N+2)·φ/ε + passReadBound·φ`, with room to spare
(`(L+1)(N+1)φ/ε` suffices).

## Consolidated: one theorem (`RoundTetrachotomy`, `round_tetrachotomy`)

Only the ladder version is kept, with the ladder fixed: a factor `β > 1` is passed in and the
rungs are `uHi / β^i`, `i = 0..|Q|+1`, so `θ₀ = uHi·β^{−(|Q|+1)}` and the ladder fits below `uHi`
by construction. The pigeonhole (`exists_rung_gap`, now over an antitone ladder) gives a rung pair
of ratio exactly `β`.

**Outcome (4)** (`ChainAdvances`): some chain's odds multiply by at least `β`.
`ChainAdvancesBy X g β hi` says `X` has mass on `g ≥ hi`, the link multiplies that mass by at least
`hi` and the rest's by at most `hi/β`. An indecision chain does so at a rung `hi = uHi/β^i`,
`i ≤ |Q|`; a disagreement chain at `hi = wHi`.

**Constants fixed inside the statement.** `wHi = 1 − 2(N+1)φ`, the edge-gap lemma's upper side
when every state is read cleanly. Internally `η = wHi/β`, so both chains advance by `β`.

**Premises:**
- `S.Valid`, `0 < ε`, `1 < β`;
- `0 ≤ φ` and `2(β+1)(N+1)·φ ≤ 1`. The second is `β·η_edge ≤ wHi` with `η_edge = 2(N+1)φ`, the
  edge gap's lower side. It is not implied by the rest, and it implies the gate's own condition
  `η < 1 − 2(N+1)φ`, since `β > 1`.
- the length, `SuffixFree`, `MidFlipPremise` and `BadVisited` premises as before.

There's no premise on `uHi`. If `uHi ≤ 0`, rung `i = 0` gives `hi = uHi ≤ 0`, and every chain with
mass advances trivially. `chainAdvancesBy_of_mass` no longer needs `0 ≤ hi`: it's the one-link case
done directly, without `rolled_odds`. So `0 ≤ wHi` was dropped as well.

`1 < β` only rules out a vacuous reading. For `0 < β ≤ 1`, (4) holds at the top rung whenever a
state is read badly, so the theorem would still hold there.

**Bound:** `(L+1)(N+1)·φ/ε + passReadBound·φ`. That is what `gate_flip_bound` proves; the earlier
`(L+1)·2(N+2)·φ/ε` was looser.

**The gap-premise version is a corollary, not kept.** It reads with the faster per-round factor
`uHi/a` under `GapPremise` (every state ≤ a or ≥ uHi). If the multi-round count needs that rate,
add it back as a corollary alongside the count: with the gap, the rung `(a, uHi)` replaces the
ladder.

Outcomes (2) and (3) are still in the statement, pending the decision whether the one-round
statement should be the dichotomy (1) ∨ (4).
