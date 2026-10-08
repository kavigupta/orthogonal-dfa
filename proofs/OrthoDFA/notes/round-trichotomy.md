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
