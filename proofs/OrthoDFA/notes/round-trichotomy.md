# The round trichotomy: proof status

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
