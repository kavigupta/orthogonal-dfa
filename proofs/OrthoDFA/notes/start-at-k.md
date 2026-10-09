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

**The relative bound does not survive the exact search.** The search reads its middle's neighbours
only when the middle is undecided. Had the middle been decided, the search would narrow to
`(j, hi)` or `(lo, j)` and go on, so it lands on the edge at `j` or `j + 1` only when that bound
was already `j + 1` or `j − 1`. Otherwise it can end at another edge, a pair, or another triple:
the walk can re-agree further right. So "triple vs edge-landing on the same pair" is not a
counterfactual pair. What does hold, over the fresh noise of the middle's reads:
- `V_j` is the event that the search visits `j` with `j − 1` agreeing and `j + 1` disagreeing,
  both decided. It depends only on reads of other prefixes, and every read it conditions on is
  decided.
- A middle string that coincides with one of those reads is decided, so collisions only help.
- Hence `P(triple at j, harvested u < u*) ≤ S · P(V_j)`, with `S = depth · u*`.
- Summed, `D(triple, u < u*) ≤ S · E[Φ]`. `Φ` counts the flanked middles the search visits. It is
  zero off decided disagreements and at most `⌈log₂(L − k)⌉ + 1`.

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
- **Deeper nodes of either:** no test. Each round adds, from its tree, the populations
  `("start", m)`, which are length-`k` draws followed by `m`, and `("end", m)`, which are draws
  followed by `m`, for every midfix `m`. The root read of `x·m` is the node-`m` read of `x`.
  - `EndsCovered`: the ends' rate of undecided reads below the root is at most the sum, over the
    midfixes below the root, of how often the reads leave `X·m` undecided. It holds for any reads
    and any tree, so it covers the frozen one. Pooled with a uniform midfix, the bound is `M`
    times the pooled rate.
  - `BadShare`: a population left undecided `ā` of the time, over the noise, has at least
    `(ā − f)/(umax − f)` of its mass at states with `u > f`.
  - The next family is then held to the FNR limit on these populations by the clustering
    guarantee.
  - Gap: `EndsCovered` is about the realized reads, while `BadShare` is about the expected reads.
    Joining them is the fresh-noise step.

`read_fresh` also stops jointly: each test is re-read every draw, and reading stops once both have
settled. `seqAbove` instead takes each test's first settled side. The claims survive any stopping
rule, since a settled side at the stop is a settled side at some look, and the union over looks
already pays `ng·a`. The Lean definition should still follow the joint rule.

## Planned: the best-start gate (waiting on Python)

**Agreement.** Sift `x` once and score each start `q` by whether the DFA from `q` accepts `x`
as the tree's label does.
- Each start gets its own sequential test at the doubling looks. With a Bonferroni split over `|Q|`,
  each test costs `|Q|·(looks)·a` in failure chance.
- Passing at `q̂` gives `D(DFA from q̂ misjudges) ≤ 1 − acc + δ`. That is `StartExists`'s `η`, so
  the gate and the export measure the same thing.
- Refusing gives the same error `≥ 1 − acc − δ` for every `q`.
- The selection term is the union over `q`. It needs the stopping rule for several tests, which
  Python has not fixed yet.

**Processing only disagreeing draws.**
- `probeOutcome` runs only where `q̂` disagrees, so each per-outcome mass becomes the mass of that
  outcome *and* gate disagreement.
- The pair test's trials are the searched draws among those, and its proof is unchanged.
- The triple bound reads `D(disagree ∧ triple, well-read) ≤ uGood·depth·E[visits·1(disagree)]`,
  with the same fresh-read core.

**Ends from the pass's quiet window.** The start half is read from the last `patience` probes.
- That window ends the pass because all its probes are quiet, and a start-undecided probe is
  always quiet. Given a fixed tree and edges, the window's draws are independent draws from
  `D | quiet`. The first run of quiet probes depends only on which probes are quiet.
- So the start rate it measures is `D(SU)/D(quiet) ≥ D(SU)`.
  - Firing gives `D(SU) ≥ (θ − δ)·D(quiet)`.
  - Not firing gives `D(SU) ≤ D(SU)/D(quiet) ≤ θ + δ`, which errs in the safe direction.
- Members added inside the window re-vote edges. That changes `D(quiet)` from probe to probe, so
  the firing claim takes the smallest `D(quiet)` over the window.
- `δ` here comes from `patience` samples, not `ng`.
- This needs the pass's probes to be random draws from `D`. Today the model takes any probes.

**Certificate** tests only `q̂`; no change to the Lean, since the certificate is not modelled.

## Lean status: `RoundAtK` proved

`RoundAtK` is proved with no `sorry` (axioms: `propext`, `Classical.choice`, `Quot.sound`).

**Per-outcome batch claims, for any reads.** These are:
- agreement;
- ends, under joint stopping at the doubling looks;
- the pair test, which costs a single `a` and needs no union over looks;
- refusal with nothing searched;
- edge ⇒ split test;
- member ⇒ unlearned edge.

**The triples' claim, over the noise.** The bound is
`D(triple, well-read) ≤ uGood·depth·E[visits] + |passReadSet|·prefixMax(k+1) + ε·(1 + uGood·depth·L)`.
It fails with chance at most `prefixMax(k)/ε²`. The proof chain:
- `qProbe` writes a probe's processing as a computation that asks the cut one string at a time.
- A triple's harvest is a middle's first undecided read (`triple_first_read`).
- One draw's fresh first reads are bounded by `uGood` times its tagged reads (`fresh_first_le`,
  via a relative `cell_bound`).
- The pass is decided by the bits its reads ask (`runPassK_determined`).
- Inside each cell of the pass, draws with different first `k` letters read disjoint bits, so
  Chebyshev applies (`cell_tail_le`). The cells partition the noise (`triple_holds_le`).

**Still to model, once Python lands:**
- the best-start gate, with agreement read from the root read only;
- the ends test run only on refusal, on 480 fresh draws plus the quiet window;
- certification of the chosen start alone.

## One theorem: the round's trichotomy (on paper, #413 at 9f9facb)

**The round as it stands.**
- **Gate:** each draw is sifted once, and its label is the middle side of its root read. Every
  start `q` is scored by whether the DFA from `q` matches that label. The best start's rate is
  tested at doubling looks, Bonferroni over `|Q|`.
- **Processing:** a draw that the best start *so far* disagrees on gets the seven-outcome read.
- **Refusal:**
  - the searched draws are rerun;
  - the triples are held;
  - the pair test runs once, over the searched draws;
  - a refusal with no searched draw halves the limit;
  - the ends are read: the end half on 480 fresh draws, the start half on the pass's last quiet
    probes.
- **The loop stops** when the gate passes, when there is nothing searched, when the budget runs
  out, or when a continued pass splits nothing.

**Draft statement.** Except with small probability, at least one of:
1. **Consistent.** The gate passes, and the DFA from `q̂` disagrees with the labels on at most
   `1 − acc + δ` of `D`.
   - Fold: `D(DFA from q̂ ≠ target) ≤ 1 − acc + δ + labelErr`, where `labelErr = D(the root
     read's middle side ≠ target)`.
   - Contrapositive on refusal: every start disagrees on at least `1 − acc − δ`. By
     `StartExists`, no `(S, h)` on which `H` matches the target then has coverage
     `η < 1 − acc − δ − labelErr`.
   - This is the cleanest join because the gate measures the DFA against the labels and
     `StartExists` measures it against the target. They meet through one triangle inequality
     in `labelErr`, with no new premise.
   - An "argmax within `2δ` of the best start" form would need uniform Hoeffding over `|Q|` at a
     look that can be as early as 30, so it is weak.
2. **Good harvest.** A held population has mass at least `M₀` and bad share at least `q₀`, where
   bad means `u > c·f`.
3. **Halving.** Plus a separate claim: below `τ*`, halving without a good harvest is unlikely.

**Why it is not provable for 9f9facb.** On refusal the disagreeing mass splits across outcome
classes. Each large class has to force outcome 2 or 3, and several cannot:
- **(a1) Edges left over when a continued pass splits nothing.** Each leftover edge produced a
  member (`add_first sprime`) or stopped at a string the cut cannot place. Neither one is a
  harvest or a halving.
- **(a2) Unlearned edges.** The gate neither reruns nor holds them. Only the pass adds members.
- **(a3) The start region.** A draw the best start gets wrong but whose walk from `k` agrees
  disagrees only within its first `k` letters: the DFA from `q̂` is not at the anchor after `k`
  letters.
  - Nothing searches there, and the pass never tests those edges either.
  - A target whose states early in the string are not reached again after position `k` is
    refused every round. Each refusal has nothing to rerun and halves the limit, at any `f`.
- **(a4) Adaptive processing.** Which draws are processed depends on the best start so far, which
  depends on the other draws' full agreement vectors.
  - So the pair test's exact conditional-binomial argument fails.
  - What remains is a stochastic-dominance argument costing `ng·a`, about some start rather than
    `q̂`.
- **(a5) No power.** "A large class forces its test" needs enough draws. A clear refusal can stop
  at 30, which leaves the pair test about `30·(1 − acc)` searched draws.
- **(b) `τ*` fails as stated.**
  - Halving on nothing-to-rerun fires on the structural classes (a2) and (a3) at any `f`.
  - Pairs are first-order, not `(depth·τ)²`. A split's midfix is `c·m` with `m` already a midfix,
    so `x[:j]·(c·m) = x[:j+1]·m`. When the middle's read fails there, the neighbour's read of the
    same string fails too, and P(pair | middle undecided) can be near 1. Pairs from well-read
    states are only bounded by about `visits·depth·u_good`.
  - Pairs from bad states could only be tied to `f` through a population at their node. Interior
    lengths have none: the gate covers the root at lengths `k` and `L`, and node `m` only where an
    earlier ends test fired. The weakest honest premise is occupancy domination: at each midfix,
    the states at interior lengths are dominated, up to a factor `κ`, by those at lengths `k` and
    `L`. It would also need start and end populations held at every midfix, every round.

**Proposed changes (for decision; nothing changed yet).**
- **R1.** On refusal, read one fixed fresh sample (the ends' 480), and classify each draw against
  the final `q̂`. Every refusal test and harvest uses that sample. This fixes (a4) and (a5).
- **R2.** Hold pairs (both reads) and unlearned-edge members as harvests, replayed like triples.
  - Pairs then become "excess over contamination", like triples and ends.
  - A pair trip with a pair harvest whose bad share is below `q₀` then implies
    `f > τ* ≈ θp·(1 − q₀)/(c·depth·(log₂L + 1))`. That is the `τ*` claim, with no occupancy
    premise.
- **R3.** Leftover edges: count the members the continued pass added as a harvest of split
  evidence. Their good property is different in kind (decided parting evidence at `s1`). The
  alternative, stopping only once the measured edge rate on the R1 sample is small, has an
  unclear termination.
- **R4.** The start region: either extend the search into `[0, k]` from `q̂`, or take as premise
  that no target state is transient (each state seen before `k` recurs after `k` with mass at
  least `κ` times its early mass). The premise is (a3) and F6 again.
- **R5.** Drop "nothing to rerun ⇒ halve". With R1–R4 every class has its own test or harvest.

**R1 approved: the refusal sample.** On refusal, `q̂` is fixed and up to 480 fresh draws are read
at the looks `30, 60, …, 480`. Reading stops once every refusal test has settled. On those draws:
- each draw is classified against `q̂`, and the ones `q̂` disagrees on get the seven-outcome
  read;
- the ends test (both halves), the pair test, the triple harvest and the reruns all use this
  sample.

What changes in the statement:
- **Processing:** a draw is processed when `Dis_q̂` holds, a fixed predicate, so the processed
  draws are i.i.d. This resolves (a4).
- **Class masses:** each class is the mass of that outcome *and* `Dis_q̂` under `D`.
- **The quiet window is gone,** and with it the random-probe requirement and its `1/D(quiet)`
  selection bias. The start half's claims read like the end half's.
- **Each refusal test, read at the joint stop:**
  - settled above means the class rate exceeds its threshold, at a cost of `a` per look;
  - settled below means the rate is under its threshold, at the same cost;
  - unsettled at 480 compares the counts, with a Hoeffding tail at `n = 480`.

  This gives power: a class rate of at least `θ + δ` makes its test fire, except with chance
  `looks·a + exp(−2·480·δ²)`. That resolves (a5).
- **The pair test** may use `pair_test_le` only if the stop is blind to which processed draws
  were searched. The ends test's draws are never searched, but the pair test's own settling is
  not blind. So the pair test also costs `looks·a`, summed over the looks.
- **Failure budget:** the gate's and the refusal sample's look terms, the Hoeffding tails at the
  gate's stop and at 480, `prefixMax(k)/ε²` for the triples, and a union over the round's gate
  readings (at most 1 plus the number of splits).

Still pending: R2 (pair and unlearned-member harvests), R3 (leftover edges), R5 (drop
nothing-to-rerun halving), and R4. The proposed R4 route is the path-based coverage
precondition, which excludes the transient-state example.

**R2 approved, with `θp = 1/2`.** Pairs (both reads) and unlearned-edge members are held as
harvests, replayed like triples.

**A pair's quality is judged by its middle read.**
- The middle read is a tagged first undecided read. Every read before it in the probe is decided:
  the walk, and the earlier middles' neighbours. So the fresh-read lemma applies as for triples:
  `D(pair, middle well-read, Dis_q̂) ≤ c·f·depth·E[visits·1(searched ∧ Dis_q̂)]`.
- That is at most `c·f·depth·(log₂(L − k) + 1)·D(searched ∧ Dis_q̂)`.

**Pair test.**
- **Trips:** `D(pair ∧ Dis) > (θp − δ)·D(searched ∧ Dis)`. The held pairs' bad share is then
  more than `1 − c·f·depth·(log₂(L − k) + 1)/(θp − δ)`.
  - So a trip with bad share under `q₀` forces `f > τ* = (θp − δ)(1 − q₀)/(c·depth·(log₂(L − k) + 1))`.
  - That is the requested `τ*`, up to `δ`.
- **Does not trip:** pairs are at most `θp + δ` of the searched draws. So edges and triples are at
  least `1/2 − δ` of them, and those carry the progress: the reruns, and the triple harvest with
  its own bad-share bound.

**Unlearned-edge members** are now a harvest. Each held `u` sits at a leaf `p` whose edge by some
letter `c` is unlearned, and `u·c` is placed. So the next round's vote has a voter for `(p, c)`.

Pending: R3 (leftover edges), R4 (the start region, likely the coverage precondition), and R5
(nothing-to-rerun halving).

**R3 approved, as a stopping rule.** Reruns continue while the latest refusal sample has edges.
The round stops when the gate passes, when a refusal sample has no edges, or when the budget runs
out.

**A stop on "no edges".** Every draw in the refusal sample is read to an outcome, and none was an
edge. So by Hoeffding at the sample's size, `D(edge ∧ Dis_q̂) ≤ δ_e`, except with chance
`exp(−2·n·δ_e²)`. The other classes then carry at least `1 − acc − δ − δ_e`.

**Budget exhaustion needs its own outcome.** Making it unlikely needs two facts:
- **(A) Real wrong edges get split.**
  - Each rerun of a real wrong edge splits, adds `sprime` ahead of the member limit, or stops at
    a string the cut cannot place.
  - The split test fires after about `m* ≈ log(2T/splitFpr)/gain` members on the minority side.
    This needs:
    - `m* ≤ memberLimit`, because `members` keeps only the first `memberLimit`;
    - the training half to classify the members correctly;
    - the parting reads to be decided. A parting read the cut cannot place gives "stopped", which
      adds nothing.
  - Those reads are interior reads at the leaf `s1`. Only the fresh-read bound covers them, and it
    covers well-read states only.
- **(B) False edges do not keep the loop going.**
  - At a leaf that is truly one state, a decided disagreement comes only from reads decided the
    wrong way. These are not bounded where the search reads: the clustering guarantee bounds wrong
    decisions only on the gated populations, the root at lengths `k` and `L`. This is the F6
    occupancy issue again.
  - "No edges" also means a zero count. A false-edge rate of about `1/(n·(1 − acc))` per draw keeps
    every refusal sample non-empty, so reruns go on until the budget runs out.

**Proposal.**
- Make budget exhaustion with edges remaining outcome 4. Prove that outcome 4 is unlikely when
  three conditions hold:
  - the budget is at least `m*·|leaves|·|Σ|` reruns' worth of probes;
  - `m* ≤ memberLimit`;
  - the false-edge rate is below the stopping threshold.
- Change "no edges" to "the edge rate settles below `θe′`" on the refusal sample. Then false edges
  below `θe′` cannot force exhaustion. (B) becomes the explicit condition "decided-wrong edges
  are below `θe′`". That is a family property at interior reads, so it is either a premise or a
  fourth way for the round to end, not something to derive.

**R4 approved, via the coverage precondition.** The precondition is about the target `A` and the
sampler `D`:
- `S` is the set of target states visited, at some position in `[k, L]`, by at least
  `min_coverage` of the draws.
- Some `q₀ ∈ S` has a good set
  `G = {x | A's run from q₀ along x stays in S, and A from q₀ accepts x iff A accepts x}` with
  `D(G) ≥ 1 − η`.

**The bound.** Take any `h : S → leaves` that preserves acceptance. Let `Mask(h)` be the mass of
draws in `G` on which `H`, started at `h q₀`, leaves `h ∘ (A's run)` while the walk from `k` still
agrees. This is the masking residue: an error made inside the first `k` letters, or one that
cancels later. Then

    err(h q₀) ≤ η + labelErr + Mask(h) + N,

where `N` is the mass of draws whose walk from `k` ends in some outcome other than agree. That mass
does not depend on the start. So a refusal, which puts `err(h q₀) ≥ 1 − acc − δ`, forces

    N ≥ 1 − acc − δ − η − labelErr − Mask(h).

Where `min_coverage` enters: a wrong edge of `H` out of `h q′`, for `q′ ∈ S`, is crossed after
position `k` by at least `min_coverage · D(c after q′)` of the draws. There it shows up as an
edge, pair or triple, unless it is masked. So `Mask` is the only start-region term left.

**Gap introduced by R1: the refusal sample reads only the draws `q̂` disagrees on.**
- The forcing bound is about `N`, the non-agree mass over all draws. A draw whose walk from `k`
  disagrees while `q̂` happens to match its label is never read.
- Bounding the start-region class for `q̂` itself goes in a circle, because `q̂ ≠ h q₀` in
  general.
- **Proposed R1′:** read the walk from `k` on every draw of the refusal sample (at most 480
  reads), whatever `q̂` says. The class masses become unconditional, and the start-region class
  drops out.

**Approved: R5, R1′, R3 by a per-edge cap, and `k = ⌈L/2⌉`.**

**Classes of the refusal sample.** Every draw is read to an outcome, so each class is an
unconditional `D`-mass:
- `SR`, `ER`: the start or end sift undecided at the root.
- `SD`, `ED`: the start or end sift undecided below the root. These are what the ends test
  measures.
- `BL`: undecided at an unlearned edge, on `x[:j+1]` or `x[:j]`.
- `MB`: members at unlearned edges.
- `PR`, `TR`: pairs and triples.
- `EG`: edges, split into live edges and edges that have been given up (`Gup`).

**Refusal.** Every start has `err ≥ 1 − acc − δ`. With the coverage bound this gives

    SD + ED + MB + PR + TR ≥ Need := 1 − acc − δ − η − labelErr − Mask − Root − Blk − Gup − δ_e

- `Root = SR + ER`. It is bounded by the clustering guarantee's root populations.
- `Blk = BL`. It is a residue unless it is held as a harvest (R6, below).
- `δ_e` bounds the live edges at a stop on "no live edges": with a zero count over at least 30
  draws, this fails with chance at most `exp(−30·δ_e)`.

**Pair test does not fire.** Then `PR ≤ ρ·(TR + δ_e + Gup)`, with `ρ = (1/2 + δ)/(1/2 − δ)`. So
one of `SD`, `ED`, `MB`, `TR` is at least

    M₀ := (Need − ρ·(δ_e + Gup))/(4 + ρ).

**The gap: large `f` with no halving.** The approved rules give no guarantee here.
- An ends class at least `M₀` fires its test only if `M₀ > (depth − 1)·f + δ`.
- A triple class at least `M₀` is a good harvest only if
  `c·f·depth·(log₂(L − k) + 1) ≤ (1 − q₀)·M₀`.
- Above `τ* = min(τ_e, τ_T, τ_P)`, a refusal can end with a large class that is only incidental
  indecision. The thresholds are:
  - `τ_e = (M₀ − δ)/(depth − 1)`;
  - `τ_T = (1 − q₀)·M₀/(c·depth·(log₂(L − k) + 1))`;
  - `τ_P = (1/2 − δ)(1 − q₀)/(c·depth·(log₂(L − k) + 1))`.
- In that case there is no fired test, no good harvest, and no halving, because R5 removed
  halving on nothing-to-rerun. So the requested trichotomy holds only for `f ≤ τ*`. There,
  halving forces a good pair harvest.

**Proposed R5′.** Give each harvest class a test against its incidental rate:
- ends: `(depth − 1)·f`, as now;
- triples and pairs: `c·f·depth·(log₂(L − k) + 1)` times the searched share.

Halve on a refusal where no class's test fires. Every refused round then either holds a class
that fired its test, whose bad share is at least `1 − incidental/rate`, or halves. Halving forces
a class of at least `M₀` under its incidental rate, so `f > τ*`. That is the requested shape.

**Proposed R6.** Hold the `BL` strings, interior undecided reads at an unlearned edge, as a
harvest. This removes the `Blk` residue.

**The per-edge cap.** Reruns are at most `m_max·Σ_round(leaves·|Σ|)` plus the number of splits.
Splits are not capped, so budget exhaustion stays a fallback outcome unless the number of leaves
is bounded. Every non-split result (undecided, no split, stopped) has to count toward the cap,
otherwise "stopped" reruns are not bounded.

## Lean status: `RoundTrichotomy` proved

`OrthoDFA/Trichotomy.lean` states the trichotomy for one reading in the pooled form (D8–D14), and
`round_trichotomy` proves it with only `propext`, `Classical.choice` and `Quot.sound`.

- **Noise set:** `μ E ≤ 5·prefixMax(k)/ε²`. Off it, the triples, pairs, start-deep, end-deep and
  blocked classes are at most `c·f` times their tagged reads plus `|passReadSet|·prefixMax(k)` and
  `ε(1 + c·f·M)`, `M` the most tagged reads a draw makes in the class (`depth·L`, `depth − 1` or
  `2·depth`).
- **Batch failure:** `2(log₂(ng/30)+2)a + |paths|·Bin(ng, acc − δc) ≥ gateCut + (1 − ν)^nr`.
- **Halving:** `f ≥ τ₀ = (1 − a)/(nr·max(depth − 1, 2c·depth, c·depth·searchSteps))`, or
  `θM·nr ≥ 1 − a`, or every covering start with `h q₀` a leaf has
  `need = 1 − acc − η − labelErr − Mask − Root − Dead ≤ ν`.

The proof:
- A refusal settles below `acc` for the best start, so for every start, which has fewer hits.
  Validity costs `a` per look over the starts.
- A covering start's disagreements are off `CoverGood`, misread labels, masked, or in the walk's
  non-agree classes. So those carry more than `need`.
- Below `τ₀` every class's rate times its trials is under `1 − a`, so a test with one hit fires.
  A halving therefore reads to the cap with no draw in a class or at a live edge.
- `searchSteps` is `log₂(L − k) + 1`, as the Python uses. A search visits at most `⌈log₂(hi − lo)⌉`
  middles.

## The classes' slack on a binary alphabet (D22)

At `L = 40`, `k = 20`, binary, `prefixMax(20) = 2⁻²⁰ ≈ 9.5e-7`, the slack had been vacuous:
`|passReadSet|·prefixMax(k) ≈ 6` at 4000 probes, and Chebyshev needed `ε ≳ 0.05` per reading.

- **Read strings.** The pass walked from `k` reads only seed strings and probe prefixes at least
  `k` long (`passReadSet k`, with the determinism proof narrowed to prefixes from `k`). A draw's
  harvested read is `x[:i]·m` with `i ≥ k`, so it can be a read string only if `x[:k]` is the
  first `k` letters of one. The slack is `|kPrefixes k (passReadSet …)|·prefixMax(k)`: one per
  probe, one per seed string at least `k` long, and at most `(|Σ|+1)(|midfixes|+1)` per shorter
  seed string. At 4000 probes that is `0.0038`, plus `0.001` per thousand long seed strings.
- **Hoeffding.** Inside a cell of the pass the draws grouped by their first `k` letters are
  independent, each group moving the sum by at most its mass times the scale, so the noise set
  is `5·exp(−2ε²/prefixMax(k))`: `3e-8` at `ε = 0.003`, `1e-5` at `ε = 0.0025`.
- **So** at `ε = 0.003` a class's slack is about `0.004 + 0.003·(1 + c·f·M)`: `0.007` for the ends
  and blocked reads, `0.007–0.013` for triples and pairs at `f` from `1e-3` to `1e-2`.

## Lean status: `RoundTrichotomyLevel` proved (D38)

`OrthoDFA/RoundLevel.lean` models a round as up to `Rmax` readings, each with fresh probes, gate
batch, refusal sample and certificate sample. `round_trichotomy_level` proves, for any reads,

    P(the round's end breaks its claim) ≤ 4(log₂(ng/30)+2)·a
        + E[#readings]·((1−ν)^nr + α + Pmax·Bin(ng, acc−δc) ≥ gateCut(ng, acc, a/2^Rmax/Pmax)).

Reading `j`'s draws are independent of everything before it, so the chance it ends the round
badly is at most its own bound whatever the history; summing over the readings the round makes
gives the expected count. On a certificate refusal only `D(NAOff) ≤ ν` is claimed: the gate passed,
so nothing bounds a covering start's disagreement below, and `need` may be at most zero there.

Problematic component: the exhausted exit (budget of `Rmax` readings or more than `Pmax` leaves).
It stays because give-ups on genuinely wrong edges and attempts stopped at undecided reads are not
yet bounded; false splits are bounded separately (claim 4). D39–D41 plan to bound the rest with
attempt counters, a stopped-read harvest and a dynamic budget.
