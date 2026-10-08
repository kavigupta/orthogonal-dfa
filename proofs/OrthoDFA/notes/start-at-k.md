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

**Not yet stated:** the export lemma (F4–F6).
