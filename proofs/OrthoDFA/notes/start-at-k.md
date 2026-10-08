# Starting at k: the round, on paper

Supersedes `round-progress.md`.

## The scheme

Fix `k < L`. Draws `x ~ D` have length `L`. A round runs three steps.

**1. Walk check.** Read a batch of `n_w` probes against the current hypothesis `H`.
- Sift `x[:k]` with the cut. If it is undecided, that is an *anchor block*.
- Otherwise follow only learned edges from its leaf. The first unlearned edge `(s₁, c)`, at
  position `j`, is an *edge block*.
- If the blocked share of the batch exceeds `θ_w`, emit the *walk source* and end the round.
  - On an anchor block it outputs `x[:k]`.
  - On an edge block it outputs `x[:j+1]`, but only if that string sifts undecided.

**2. Counterexample check.**
- Walk from `k` as above, then sift `x`.
  - An undecided sift is blocked.
  - A decided sift that disagrees with the walk is bisected over `[k, |x|]`. Undecided midpoints
    are placed at the middle of the band (#411).
  - Then come the guards and the split test, with NO_SPLIT treated as UNDECIDED (#412).
- Patience as now.
- If the blocked share exceeds `θ_c`, emit the *boundary source*. It outputs the undecided string
  where the probe was blocked.

**3. Done.**
- The start is the leaf `q` whose run of `H` over `x` best agrees with the sift of `x` on a
  separate batch.
- `ε` is never read.

Populations are sets of distinct strings.

## Decided: F1–F3

**F1.** On an edge block at `(s₁, c)`, position `j`:
1. If `x[:j]·c` sifts undecided, output it.
2. Otherwise, if `x[:j]` sifts to `s₁`, output `x[:j]` as a new member of `s₁`. An unlearned edge is an
   empty leaf, or a leaf all of whose members are undecided at `·c`, so this is what edge resolution
   lacks.
3. Otherwise output nothing.

So the walk source's yield is its blocked share minus case 3.

One gap is left in case 3: `x[:j]` sifting *undecided* is not a wrong earlier edge. It is an
undecided string of length `≥ k`, of the same kind as an anchor block's, and could be output too.
Then case 3 is exactly "`x[:j]` sifts decided to a leaf other than `s₁`": an earlier wrong edge.

**F2/F3.** Everything is measured on the gate's fresh batch, against a hypothesis frozen after
patience: no splits, no re-votes, no new members. The batch is up to 2000 draws with an early stop.
- On that batch: the blocked share, the decided disagreements, and agreement. The walk trigger is
  measured the same way, on its own frozen batch.
- Patience only ends the pass; it certifies nothing.
- Decided disagreements seed the next pass.

The outcomes:
- a walk source;
- a boundary source;
- the gate passes;
- the gate refuses, with its decided disagreements seeding the next pass.

Every probability bound is now Hoeffding, or the gate's sequential test, on a frozen hypothesis.
The F2 and F3 problems are gone.

**The fourth outcome is concrete progress only if two things hold.**
1. **The gate's disagreement is the k-walk check's:** the walk from `x[:k]`'s sift along `x[k:]`,
   against `x`'s decided sift.
2. **The seeded probes are rerun against the same frozen hypothesis**, in the same round, before the
   family changes.

Given both, by persistent noise each seeded probe disagrees decided again. The walk uses learned
edges only, so it was not blocked. With #411 the bisection over `[k, |x|]` always lands on an edge,
except at an exact tie with the middle. Then:
- the placeholder guard cannot fire, since the walk took only learned edges;
- a learned edge has a witness;
- the witness sifts to `s₁`, since the tree is frozen;
- so the probe reaches the split test unless one of two reads is undecided:
  - `sprime` itself, placed by the middle in the bisection;
  - a read of `sprime·c·m` or `witness·c·m` in the search for the distinguisher.

The split test either splits, or answers UNDECIDED and adds `sprime` as a member.

So each seeded probe:
- splits a leaf;
- adds a member;
- meets an undecided string at a guard, which is not kept today and should go to the boundary
  source; or
- meets a decided wrong read, which is bounded by the clustering guarantee's `crossLimit`.

**If either condition fails, it is not progress.**
- If the gate scores the exported DFA from its chosen start over all of `x`, a refusal's
  disagreements can vanish when walked from `k`. That is F5: an error in the first `k` steps, or in
  the start, that the k-walk never sees.
- If the next pass belongs to a new round, the family and so the reads have changed, and a seeded
  probe need not disagree at all.

## (a) The round theorem (before F1–F3)

**Claim.** Except with small probability, a round ends with
- (W) a walk source of yield `≥ θ_w − δ`, or
- (B) a boundary source of yield `≥ θ_c − δ`, or
- (D) done, with `H`'s check disagreement `d_k ≤ ε` and blocked share `b_k ≤ θ_c + δ`.

The yield is the chance one attempt outputs a string, with the noise and the hypothesis fixed.

**Walk check.** Here the tree and edges are fixed and the probes are i.i.d.
- So Hoeffding gives `P(the batch share > θ_w and the true share < θ_w − δ) ≤ exp(−2 n_w δ²)`.
- This holds without any adaptivity argument.

**F1. A source's yield is not its blocked share.**
- An edge block outputs `x[:j+1]` only when that sifts undecided.
- An edge can be unlearned while the draws that reach it sift decided:
  - its leaf has no members; or
  - its members are all of a state whose successor by `c` is badly read, while the draws reach the
    leaf through another state, merged into it, whose successor is clean.
- On such draws the probe is blocked but the source outputs nothing.
- So its yield is `P(anchor block) + P(edge block with x[:j+1] undecided)`, which can be far below
  the blocked share.
- Concrete regime: a split leaves a new leaf with no members. Every draw through it then makes an
  edge block that outputs nothing.
- Fix options:
  - output `x[:j]` there, a member for `s₁` (cheap, but a different kind of string);
  - count only blocks that output a string toward `θ_w`.

**F2. The counterexample check's share is measured against a moving hypothesis.**
- The tree splits during the pass. Edges are re-voted after every probe, as the pool grows.
- The boundary source replays against the final hypothesis, so its yield is not the share
  measured over the pass.
- It is exact if the share is measured over a stretch where `H` does not change. Two ways to get
  that:
  - Measure on a fresh batch against the final `H` after the pass. Hoeffding as above.
  - Use the final patience window, but only if a quiet probe changes nothing: no anchor seeding,
    and no re-vote that moves an edge.

**F3. Patience bounds one epoch, not the round.**
- Between resets the hypothesis is fixed (given the F2 condition). An epoch whose `H` has
  `d_k > ε` survives `p = 149` quiet probes with chance `≤ (1−ε)^p ≈ 0.05`.
- But a pass has many epochs, since every SPLIT and every UNDECIDED (now including NO_SPLIT)
  resets.
- So `P(done with d_k > ε) ≤ E[#epochs with d_k > ε]·(1−ε)^p`.
- That is not small. About `|Q|` splits, each after about 2–3 UNDECIDED, already gives
  `30 · 0.05 > 1`.
- Fix: decide (D) on a fresh batch against the final `H`: disagreement share `≤ ε − δ` and blocked
  share `≤ θ_c`. Hoeffding again. This is the gate's role today.
- A failed batch has to send the pass back to probing, which is a sequential test. Its union over
  the at most `max_probes / n` batches is cheap.
- That adds a fourth exit, where the batch fails after the pass ends. It needs its own outcome
  or the loop.

**Spread holds.** Every harvested string `t` has `|t| ≥ k`, so
`P(an attempt outputs t) ≤ P(x[:|t|] = t) ≤ p_max^k`, with `p_max` the largest letter
probability. (The final sift's string is `x·m`, so it is bounded by `p_max^L`.)
- So the spread notion holds with `φ` = "the attempt outputs a string":
  - `P(φ) =` yield `≥ θ − δ`;
  - `max_t P(t | φ) ≤ p_max^k / (θ − δ) =: κ`.
- After `n` distinct strings, the fresh-string yield is `≥ (θ − δ)(1 − nκ)`.
- For a non-i.i.d. sampler, `p_max^k` becomes the largest prefix probability at length `k`.

## (b) The export lemma

**The selection.**
- Choose `q*` on a batch of `n_s`, separate from (D)'s.
- With probability `1 − δ_s`, the export's error is at most `min_q err(q) + 2√(ln(2|Q|/δ_s)/(2n_s))`.
  This is Hoeffding plus a union over the `|Q|` starts.

**F4. Score acceptance, not leaf equality.**
- Re-rooting at a covered state `q_c` keeps only acceptance (`covered_accuracy_ceiling` compares
  accept labels). The end state after re-rooting is generally different from the true end state.
- Scoring "run from `q` lands on the sift's leaf" can be low for every `q` while acceptance
  agreement from `h(q_c)` is 0.99. Here `h(q)` is the leaf holding state `q`'s strings.
- A start-dependent counter shows it: the end state carries an offset, but acceptance ignores it.
- So score "run from `q` accepts iff `x`'s accept read does".

**The bound.** Then
`err(h(q_c)) ≤ (1 − ceiling) + s_c`, where `s_c` is the chance `H`'s run from `h(q_c)` leaves
`h(A(q_c, x[:i]))` at some step and does not come back before the end.

**The occupancy premise.**
- Wanted: every state `A` visits from a covered start within `L` steps is occupied at the check's
  positions.
- **Not "every position".** That fails for length-periodic targets. With length parity in the
  state, a state sits at alternate positions only, so the per-position `μ` is 0.
- The natural form averages over the check's window:
  `μ := min over those states q of (1/(L−k)) Σ_{j ∈ [k,L)} P(A(x[:j]) = q)`.
- It is a target-and-sampler quantity, like `covered_accuracy_ceiling`.
- **It is not derivable from what we have:**
  - Python's "covered" is endpoint mass at `L` only.
  - A late-absorbing state, such as "seen 1111", is common at `L` but rare at `j ≈ k`.
  - The Lean's only state-mass condition is the spread `ν(y) ≤ κ·V(A.state y)`. That is an upper
    bound on prefix mass, not a lower bound on state mass.
  - No occupancy lemma exists in the Lean.
- So it is a new premise, and an irreducible one.

**What it buys.**
- **Step match.** An export deviation at step `i` uses edge `(q, c)` with chance at most
  `P(on track, at q) · p_c ≤ p_c`.
- **Window charge.** By the occupancy premise, the check's window meets `(q, c)` with chance
  `≥ μ·p_c`, averaged over `j`.
- **Resync.** "No resync within `m` steps" is non-increasing in `m`. The export at `i < k` has
  more steps left than the check at `j = k`, so `r(L−i−1) ≤ r(L−k−1)`.
  - For `i ≥ k`, match `j = i`.
  - For `i < k`, charge position `k`.
- **Result.** That gives `s_c ≤ (k + 1)(d_k + b_k)/μ` per-position, or `(L/μ)` times the window
  average with the averaged `μ`.

**F5. Masking.**
- The step `S_i ≤ T_j / μ` needs the check's run to be on track at `j`.
- An earlier deviation that later resyncs (and so isn't counted in `d_k`) can be off track exactly
  at `j`. That masks a deviation at `j` which the export, starting on track, does pay.
- So the `μ` bound holds for the first deviation only up to `P(the check run is off track at j
  but on track at the end)`. The check's end disagreement does not control that.
- It needs two interacting wrong edges, one always resyncing.

Ways to close it:
1. **A local-edge check.** Define `e_loc := P_{x, j}(H's step from x[:j]'s sift ≠ x[:j+1]'s sift)`
   and measure it directly. That costs one sift per position: about `L − k` sifts per probe instead
   of `log L`. Then `err ≤ (1 − ceiling) + (L/μ)·e_loc + selection`, with no masking.
2. **The (D) batch measures the export's acceptance error directly**, at the chosen start. Then
   (D) needs no export lemma at all. But a refusal there needs its own outcome, which is the old
   problem.
3. **Keep end disagreement and add a masking term**, bounded by the share of draws with any
   deviation in `[k, j)`. It is again not controlled by `d_k`.

## Decisions needed

- **F1:** what the walk source outputs on a decided edge block.
- **F2:** where the blocked share is measured.
- **F3:** (D) by a fresh batch, and what a failed batch does.
- **F4:** acceptance scoring for the start.
- **F5:** how the export lemma's masking is closed, or which check replaces end disagreement.
- **The occupancy premise:** window-averaged, over the states reachable from covered starts.

## Lean status

`OrthoDFA/StartAtK.lean` states `RoundAtK`, `CheckYield` and `SourceSpread`, and
`Proofs/StartAtK.lean` proves all three with no sorry.

**The model.**
- The pass is #411 and #412: `probeStepK`, `runPassK`.
- Each check is `SequentialRate`: `binomial_side_of_boundary` at failure chance `a` from `n₀`
  draws on, read one draw at a time, falling back to the batch's rate if it never settles.
- There is no separate walk check. Every round rebuilds the tree from the root, so the first
  hypothesis has two leaves. The pass always runs and the gate reads its result. A walk block is a
  blocked check, and the check source outputs `walkOutput`'s string for it.
- The gate's agreement is over all draws, with the start and the whole draw placed by the cut
  where it can and at the middle where it cannot. A walk that meets an unlearned edge counts as
  disagreeing. That is the reading the Python is moving to; the edge-block case is my choice and is
  to be confirmed.
- `θw`, `θc` (`reads × fnr_limit` in the Python), `acc`, `a`, `n₀` and both batch caps are
  parameters, with `θw, θc, acc ∈ [0, 1]` and `a, δ ≥ 0`.

**The sequential test.** A fixed-batch bound does not cover early stopping. The proof bounds each
reading by the test's own failure chance at every look, plus Hoeffding at the last draw:
`N·a + exp(−2Nδ²)`.
- The count of a batch's first `n` draws is exactly binomial (`pi_count_ge`, `Proofs/BinomLaw.lean`).
- `binomSfGe` grows with the rate (`binomSfGe_mono`).
- So a look against `θ` at a rate `≤ θ` fires with chance `< a` (`look_above_le`, `look_below_le`).

**Not exclusive.** A tripped check only adds its source and halves the limit, so a round can both
add a source and pass. The theorem claims each reading's consequence, and its bound sums them.

**Seedless refusal.** A refusal none of whose first `n₀` draws can be carried adds the check
source and halves the limit. Carried means a decided disagreement, or an unlearned edge reached
through a wrong earlier one.
- The source's yield is at least the gate's disagreement less `δ`, but for `exp(−min(n₀, ng)·δ)`.
- That holds because every draw the gate counts is output by the check source or can be carried
  (`gateDisagrees_cover`), and a batch missing a set of mass above `δ` in its first `n₀` draws is
  that unlikely (`miss_first_le`).
- The Python reads at least `n₀` draws before refusing, so "none of its read draws can be
  carried" implies the Lean hypothesis. But only once the Python also carries a wrong-earlier-edge
  draw: its prefix `x[:j]` disagrees, decided. Until then, such draws count against the gate
  while the source outputs nothing for them.

**Refusal.** The refusal claim rests on the pass keeping its edges learned. That invariant is
proved for the hypothesis the gate reads (`runPassK_learned`).

**Continuation.** The theorem is per reading. The round's continuation loop re-gates after each
continuation, so the round's error is the sum over the gates it runs.

**Split soundness is not needed.** A split can be false only at a highly indecisive state, whose
selected strings were picked by the full-family read at the split's node. A round reading such a
state harvests heavily (blocked draws, halving), so it is not the round that finishes. Every round
rebuilds the tree from the root, so a false split dies with its round. In the finishing round a
duplicate leaf makes the DFA non-minimal, not wrong. The gate and the certificate judge the end
state, and the pass is bounded by its probe budget, not by its leaf count.

**Start state (`StartExists`, `StartState.lean`).** This is separate from the round. Suppose `H`
agrees with the target on a set `S` of target states, read through `h`. Then from `h q`, `H`
misjudges at most `1 − cov(q)` of the draws, where `cov(q)` is the share on which the target
re-rooted at `q` stays in `S` and agrees with the target. No occupancy premise and no leftover term
is needed.

The assumption `cov(q) ≥ 1 − η` is the weakest of its kind. Off `S`, `H` can do anything, and on
the agreeing runs it tracks the re-rooted target exactly.

How it relates to `covered_accuracy_ceiling`, which drops the stay-in-`S` part:
- The ceiling implies the assumption when `S` contains every state the best covered start's runs
  visit, for example when `S` is closed under the target's steps.
- It does not imply it otherwise. With length parity in the state and `L` even, the covered states
  are all even, and runs from them leave on every odd step.
- The matching check for `satisfies_preconditions` is the share of strings whose run from a
  covered start stays within the covered states and agrees.

## The triple harvest (#413), on paper

**Rule.** A decided disagreement on the frozen hypothesis is bisected over decided reads only. The
final bracket `[a, b]` has the walk agreeing at `a`, a decided disagreement at `b`, and every read
strictly between undecided:
- `b − a = 1` is an edge, and the split test runs;
- `b − a = 2` is a triple, and its middle is harvested;
- `b − a ≥ 3` is a run, with no harvest; it feeds halving.

**The harvested string must be the undecided read, not `x[:a+1]`.** A middle sift is undecided at
its first undecided node `d`, so the read that failed is `x[:a+1]·d`, at state
`δ(state(x[:a+1]), d)`. The state of `x[:a+1]` itself can be read well at the root while a deeper
node of its route is read badly. Harvesting `x[:a+1]` would hand the FNR gate a string its family
already decides.

**Quality claim (relative, per gate batch).** Fix the frozen hypothesis. For each pair of
consecutive edges, consider draws whose decided ends bracket it as above. Whether such a draw is a
triple or lands on an edge depends only on its middle sift, which reads fresh strings when the
pass did not read them. Its first undecided node has indecision `u_i`. So the draws whose
harvested read comes from a state with `u < u*` satisfy:

    D(triple, harvested state's u < u*) ≤ (S/(1 − S)) · D(edge-landing on those pairs)
                                          + (middles the pass read) + fluctuation,
    S = Σ_{nodes on the middle's route, u_i < u*} u_i ≤ depth · u*.

The edge-landing draws are decided disagreements that reach the split test, so they are progress.
Either they are few, and then a passing gate bounds them by `1 − acc + δ` (the absolute form
`ε·u/(1 − u)`), or the refusal reruns them as seeds.

This needs **no randomness in the pass's probes**. The pass stays a black box. The randomness is
the gate batch's fresh draws, together with the fresh noise of middle strings the pass did not
read, which is `gate_flip_bound`'s machinery (`cell_bound`). Concentration over distinct middle
strings uses the spread: a middle has length `≥ k + 1`, so one string carries at most
`p_max^(k+1)` of the draws.

**`u_sift²` is not what quality needs.** Two adjacent undecided reads make a run, which is never
harvested, so they cannot contaminate the harvest. Its mass, about `(L − k)·u_sift²`, is what keeps
runs rare. That is a liveness condition, and it is what halving controls. The contamination term
is the first-order `depth · u*` above.

**Draws blocked at the ends never become triples.** A triple needs decided reads at the anchor
`x[:k]` and at the end `x`. Halving on a refusal with nothing decided drives down only what the FNR
gate measures:
- **End, root read:** the uniform population is full-length draws, so the clustering guarantee
  bounds the family's indecision there, averaged by state weight at length `L`. Derivable.
- **Start, root read:** covered by a population of sampler draws cut to length `k` (approved).
  `StartRootCovered` (`Ends.lean`) turns the clustering's per-population bound into a bound on the
  start's root indecision. Using the guarantee there needs its existing premises for that
  population, in particular `collisionMass ≤ cap`, which constrains how small `k` can be.
- **Deeper nodes of either:** not derivable, so the round gets an ends source (decided). The gate
  counts end sifts undecided below the root. When that rate per below-root read exceeds
  `fnr_limit`, the round holds the deeper undecided strings. Quality: incidental indecision at
  good states (`u ≤ f`) contributes at most `f` per read, so the bad share is at least `1 − f/rate`.

**The ends test as #413 runs it is not a valid sequential test.** `binomial_side_of_boundary(deep,
below, fnr_limit)` treats each below-root read as an independent trial. Given the persistent
noise, though, draws are the independent unit, and their read counts vary. Counterexample: 90% of draws have one end undecided at the first node below the root (one read,
one hit). The rest read decided through 86 levels at both ends (172 reads, no hit). Then
`ρ = 0.9/18.1 < 0.05`, yet all of the first 30 draws are light with chance `0.9^30 ≈ 0.042`. At
`f = 0.1` the test then reads above, at the first look the agreement test can stop at. So
`P(trip ∧ ρ < f − 0.05) ≈ 0.042`, more than `ng·a + exp(−2·ng·0.05²) ≈ 0.002` at `a = 1e-6` and
`ng = 2000`. A valid version with no extra queries thins to one read per end: draw a level `j`
uniformly among the `depth − 1` below the root. The end is a trial when its sift read level `j`,
and a hit when that read was the undecided one. Trials are then i.i.d. `Bernoulli(ρ)`, so the
exact binomial test and `pi_count_ge` apply.

`read_fresh` also stops jointly: each test is re-read every draw, and reading stops once both have
settled. `seqAbove` instead takes each test's first settled side. The claims survive any stopping
rule, since a settled side at the stop is a settled side at some look, and the union over looks
already pays `ng·a`. The Lean definition should still follow the joint rule.
