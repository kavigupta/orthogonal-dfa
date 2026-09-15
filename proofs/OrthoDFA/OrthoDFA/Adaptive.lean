import OrthoDFA.Distributional

/-!
# The adaptive clustering loop: the integrated theorem

`sample_suffix_family` is a retry loop: cluster, measure the FNR **per population**, and
if any population is too indecisive grow the pool and try again — stopping at a
data-dependent time.  This file states and proves the guarantee **for that loop**, with
no fixed budget:

> with probability `≥ 1 − δ` the loop **terminates**, and **whatever** family it returns
> preserves acceptance on `≥ 1 − εcov` of **each** prefix population.

Everything the statement needs is present and constrained: the persistent RCN oracle, the
collection of prefix populations, the suffix distribution with its findability `pAP`, the
loop's own FNR return test (defined, not abstract), and an unbounded growing schedule.
The draws carry their laws and independence; the returned family is *defined* by the
clustering.

Proof: the two-part decomposition —

* `validity_of_returned` — whatever is returned is valid, *whenever* it is returned,
  except w.p. `δ/2`;
* `loop_terminates` — the loop returns at some round, except w.p. `δ/2`;
* `sound_and_terminating` composes them.

The composition is proved; the two halves are `sorry`, with their proof plans recorded.

**Known modelling gap (flagged, not hidden).**  The prefix draws here are i.i.d. from
each population, whereas `sample_more_prefixes` *rejects duplicates* — it samples without
replacement.  The two agree except on collisions.  Sampling without replacement is
strictly *more* concentrated (Hoeffding 1963), so i.i.d. is conservative per round; but
the collision mass grows with the number of draws, so closing `validity_of_returned` at
unbounded budgets will need either the without-replacement concentration or a non-atomic
prefix distribution.  That is a gap in the *proof*, recorded rather than papered over by
weakening the claim.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S] [IsCancelMul S] [MeasurableMul S]
  [Countable S] [MeasurableSingletonClass S] [DecidableEq S]

/-! ## The derived configuration

`build_pst` does not take the family size; it *computes* it:

```python
n, eps = population_size_and_evidence_margin(
    signal_strength=min_signal_strength, acceptable_fpr=0.01, acceptable_fnr=0.01)
config = SearchConfig(suffix_family_size=n, evidence_margin=eps, ...)
```

and `min_signal_strength` is `½ − η`, already carried by the `Oracle`.  So the family
size and the evidence margin are **derived** from the oracle's signal together with the
two acceptable rates — they are not free parameters. -/

/-- Binomial CDF: `P[Bin(N,p) ≤ j]`. -/
noncomputable def binomCdf (N : ℕ) (p : ℝ) (j : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (j + 1), (N.choose i : ℝ) * p ^ i * (1 - p) ^ (N - i)

/-- `evidence_margin_for_population_size`: at population size `N`, the margin `eps`
around `center` is *admissible* when the binomial false-positive rate under the null and
false-negative rate under the signal are both within budget. -/
def admissibleMargin (s fpr accFnr center : ℝ) (N : ℕ) (eps : ℝ) : Prop :=
  0 < eps ∧ eps ≤ s ∧
    (binomCdf N center ⌊(N : ℝ) * (center - eps)⌋₊
        + (1 - binomCdf N center (⌈(N : ℝ) * (center + eps)⌉₊ - 1)) ≤ fpr) ∧
    (binomCdf N (s + center) (⌈(N : ℝ) * (center + eps)⌉₊ - 1)
        - binomCdf N (s + center) ⌊(N : ℝ) * (center - eps)⌋₊ ≤ accFnr)

/-- A large enough population always admits a margin, for any positive signal.  (The
binary search in `population_size_and_evidence_margin` terminates.) -/
theorem exists_admissibleMargin (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) :
    ∃ N, 0 < N ∧ ∃ eps, admissibleMargin s fpr accFnr center N eps :=
  sorry

open scoped Classical in
/-- **The suffix family size, derived.**  `population_size_and_evidence_margin` returns the
*least* population size admitting a margin; this is that `N`. -/
noncomputable def suffixFamilySize (s fpr accFnr center : ℝ) : ℕ :=
  if h : ∃ N, 0 < N ∧ ∃ eps, admissibleMargin s fpr accFnr center N eps then Nat.find h else 1

open scoped Classical in
/-- **The evidence margin, derived**: the margin admissible at that population size. -/
noncomputable def evidenceMargin (s fpr accFnr center : ℝ) : ℝ :=
  if h : ∃ eps, admissibleMargin s fpr accFnr center (suffixFamilySize s fpr accFnr center) eps
  then h.choose else 0

theorem suffixFamilySize_pos (s fpr accFnr center : ℝ) (hs : 0 < s)
    (hfpr : 0 < fpr) (haccFnr : 0 < accFnr) : 0 < suffixFamilySize s fpr accFnr center := by
  classical
  rw [suffixFamilySize, dif_pos (exists_admissibleMargin s fpr accFnr center hs hfpr haccFnr)]
  exact (Nat.find_spec (exists_admissibleMargin s fpr accFnr center hs hfpr haccFnr)).1

/-- The algorithm's family size, as `build_pst` computes it from the oracle's signal
`½ − η` and the two acceptable rates. -/
noncomputable def cfgK (O : Oracle μ S) (fpr accFnr center : ℝ) : ℕ :=
  suffixFamilySize (1 / 2 - O.η) fpr accFnr center

/-- The algorithm's evidence margin, likewise derived. -/
noncomputable def cfgMargin (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  evidenceMargin (1 / 2 - O.η) fpr accFnr center

/-- The gate's accept threshold: `decision_boundary + evidence_margin`. -/
noncomputable def cfgAcc (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  center + cfgMargin O fpr accFnr center

/-- The gate's reject threshold: `decision_boundary − evidence_margin`. -/
noncomputable def cfgRej (O : Oracle μ S) (fpr accFnr center : ℝ) : ℝ :=
  center - cfgMargin O fpr accFnr center

/-- The membership query the oracle actually answers: `MQ w = ℓ(w) ⊕ noise(w)`. -/
noncomputable def mq (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-- What a run draws, with its **exact joint law**.

Both streams are drawn *without replacement*: `_draw_cohort` skips suffixes already in
the table (`if self.table.contains_suffix(v): continue`) and `sample_more_prefixes` skips
prefixes already drawn.  So a block of `n` draws is i.i.d. **conditioned on being
distinct** — precisely rejection sampling, which is what `law_joint` says.

`law_joint` is a single equation fixing the whole joint law: the persistent noise, the
suffix block, and the per-population prefix blocks are jointly independent, each block
being its distribution's `n`-fold product conditioned on distinctness.  No independence
assumption is left implicit. -/
structure Draws {Ξ : Type*} [MeasurableSpace Ξ] (ν : Measure Ξ) (μ : Measure Ω)
    {J : Type*} [Fintype J] (D : J → Measure S) (Dsf : Measure S) where
  nz : Ξ → Ω
  sfx : ℕ → Ξ → S
  prf : J → ℕ → Ξ → S
  meas_nz : Measurable nz
  meas_sfx : ∀ i, Measurable (sfx i)
  meas_prf : ∀ j i, Measurable (prf j i)
  law_joint : ∀ n : ℕ,
    Measure.map (fun x => (nz x, (fun i : Fin n => sfx i.val x),
        (fun (j : J) (i : Fin n) => prf j i.val x))) ν
      = μ.prod
          ((ProbabilityTheory.cond (Measure.pi fun _ : Fin n => Dsf)
              {p : Fin n → S | Function.Injective p}).prod
            (Measure.pi (fun j : J => ProbabilityTheory.cond
              (Measure.pi fun _ : Fin n => D j) {p : Fin n → S | Function.Injective p})))

section Loop

variable {Ξ : Type*} [MeasurableSpace Ξ] {ν : Measure Ξ} {J : Type*} [Fintype J]
  {D : J → Measure S} {Dsf : Measure S}

/-! ## The loop, with the schedule left to the algorithm

The growth schedule is not a user parameter — it is something the algorithm optimizes as
it sees fit (`sample_suffix_family` alternates suffix- and prefix-growth based on whether
the FNR improved, and even the suffix increment is the *screened* `kept` count).  So the
statement fixes no schedule.  Instead a **history** records the sequence of budget states
the loop has passed through, and the guarantee is uniform over *all* histories and *all*
budgets — whatever the algorithm chooses, it is covered.  Histories are countable, so the
union bound still closes. -/

/-- The loop's growth history: the budget states it has passed through, in order.  The
algorithm picks this however it likes. -/
abbrev Hist := List (ℕ × ℕ)

/-- The candidate pool at a suffix budget: the first `M` suffixes drawn. -/
noncomputable def poolAt (dr : Draws ν μ D Dsf) (M : ℕ) (x : Ξ) : Finset S :=
  (Finset.range M).image (fun i => dr.sfx i x)

open scoped Classical in
/-- The representative prefixes at a prefix budget: every population's first `m` draws. -/
noncomputable def prefixesAt (dr : Draws ν μ D Dsf) (populations : Finset J) (m : ℕ)
    (x : Ξ) : Finset S :=
  populations.biUnion (fun j => (Finset.range m).image (fun i => dr.prf j i x))

/-- The family's vote on a prefix: the mean membership query over the family. -/
noncomputable def vote (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  (∑ v ∈ F, mq O (p * v) ω) / F.card

open scoped Classical in
/-- `identify_cluster_around`'s loss: the Hamming distance from a candidate's mask row to
the cluster's **own** thresholded mean (`masks[cluster].mean(0) > decision_boundary`). -/
noncomputable def hammingLoss (O : Oracle μ S) (F : Finset S) (b : ℝ) (P : Finset S)
    (ω : Ω) (v : S) : ℝ :=
  ((P.filter (fun p => ¬ ((mq O (p * v) ω = 1) ↔ b < vote O F p ω))).card : ℝ)

open scoped Classical in
/-- One Lloyd step: recentre on the current cluster, then retake the `k` least-loss
candidates. -/
noncomputable def lloydStep (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω) (k : ℕ)
    (F : Finset S) : Finset S :=
  leastLossSubset (hammingLoss O F b P ω) cands k

/-- `identify_cluster_around` iterated to its fixed point.  The loss is a natural number
bounded by `#P` that strictly decreases at each improving step, so `#P + 1` iterations
from the seed `ε` already sit at the fixed point — the bound is derived, not a knob. -/
noncomputable def clusterAround (O : Oracle μ S) (b : ℝ) (P cands : Finset S) (ω : Ω)
    (k : ℕ) : Finset S :=
  (lloydStep O b P cands ω k)^[P.card + 1] {(1 : S)}

open scoped Classical in
/-- The boundary update at the end of `identify_cluster_around`: the midpoint of the
accept-side and reject-side prefix means, falling back to whichever side is nonempty. -/
noncomputable def newBoundary (O : Oracle μ S) (F P : Finset S) (ω : Ω) (b : ℝ) : ℝ :=
  let acc := P.filter (fun p => b < vote O F p ω)
  let rej := P.filter (fun p => ¬ (b < vote O F p ω))
  let am := (∑ p ∈ acc, vote O F p ω) / acc.card
  let rm := (∑ p ∈ rej, vote O F p ω) / rej.card
  if acc.Nonempty then (if rej.Nonempty then (am + rm) / 2 else am)
  else (if rej.Nonempty then rm else b)

/-- The cluster at one budget state, at the boundary carried in. -/
noncomputable def clusterAt (O : Oracle μ S) (populations : Finset J)
    (dr : Draws ν μ D Dsf) (fpr accFnr : ℝ) (x : Ξ) (b : ℝ) (Mm : ℕ × ℕ) : Finset S :=
  clusterAround O b (prefixesAt dr populations Mm.2 x) (poolAt dr Mm.1 x) (dr.nz x)
    (cfgK O fpr accFnr b)

/-- The decision boundary carried along a history: it starts at `1/2`
(`decision_boundary : float = 0.5`) and each state replaces it with the boundary its own
cluster induces. -/
noncomputable def boundaryFold (O : Oracle μ S) (populations : Finset J)
    (dr : Draws ν μ D Dsf) (fpr accFnr : ℝ) (x : Ξ) : ℝ → Hist → ℝ
  | b, [] => b
  | b, Mm :: h =>
      boundaryFold O populations dr fpr accFnr x
        (newBoundary O (clusterAt O populations dr fpr accFnr x b Mm)
          (prefixesAt dr populations Mm.2 x) (dr.nz x) b) h

/-- The boundary after a history. -/
noncomputable def boundaryAfter (O : Oracle μ S) (populations : Finset J)
    (dr : Draws ν μ D Dsf) (fpr accFnr : ℝ) (x : Ξ) (h : Hist) : ℝ :=
  boundaryFold O populations dr fpr accFnr x (1 / 2) h

/-- The family the loop proposes at budget `Mm`, having come through history `h`. -/
noncomputable def famAt (O : Oracle μ S) (populations : Finset J)
    (dr : Draws ν μ D Dsf) (fpr accFnr : ℝ) (x : Ξ) (h : Hist) (Mm : ℕ × ℕ) : Finset S :=
  clusterAt O populations dr fpr accFnr x (boundaryAfter O populations dr fpr accFnr x h) Mm

/-- A prefix is *decided* when the family's vote clears the state's accept or reject
threshold; otherwise it lands in the indecisive band and counts towards the FNR. -/
def decided (O : Oracle μ S) (b fpr accFnr : ℝ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  cfgAcc O fpr accFnr b ≤ vote O F p ω ∨ vote O F p ω < cfgRej O fpr accFnr b

open scoped Classical in
/-- **The loop's return test** (PR #257: held per population, not over their union), at
the state's own boundary and margin. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J)
    (dr : Draws ν μ D Dsf) (fpr accFnr indecisionLimit : ℝ) (hM : Hist × (ℕ × ℕ)) : Set Ξ :=
  {x | ∀ j ∈ populations,
    (((Finset.range hM.2.2).filter (fun i => ¬ decided O
        (boundaryAfter O populations dr fpr accFnr x hM.1) fpr accFnr
        (famAt O populations dr fpr accFnr x hM.1 hM.2) (dr.prf j i x) (dr.nz x))).card : ℝ)
      ≤ indecisionLimit * hM.2.2}

/-- The family at a reachable state is **invalid**: on some population it fails to
preserve acceptance on a `1 − εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S)
    (dr : Draws ν μ D Dsf) (fpr accFnr εcov : ℝ) (hM : Hist × (ℕ × ℕ)) : Set Ξ :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | ∀ v ∈ famAt O populations dr fpr accFnr x hM.1 hM.2,
            O.label (p * v) = O.label p}}

/-- **Part 1 — whatever is returned is valid, whenever it is returned.**

Except with probability `δ/2`, the family is valid at **every reachable state** — every
history the algorithm might follow and every budget it might stop at.  So the loop may
grow and stop however it likes: neither its schedule nor its stopping rule has to be
modelled or itself proved correct.

Proof plan (pieces in `Distributional.lean`): at one state this is `clustering_budget` —
the selection avoids suffixes of high distributional flip-mass (`ploss_good_upper` /
`ploss_bad_lower`, two-level concentration against the *persistent* oracle), and
`coverage_of_summed_flip` turns per-suffix flip control into the per-population fraction.
States are countable, so union-bound them at a summable weight.  Because the draws are
*without replacement*, the per-state concentration needs Hoeffding's 1963 result that
sampling without replacement is at least as concentrated as with replacement (not in
Mathlib; to be proved here). -/
theorem validity_of_returned [IsProbabilityMeasure ν]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (dr : Draws ν μ D Dsf)
    (fpr accFnr : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) :
    ν.real (⋃ hM, FailAt O populations D dr fpr accFnr εcov hM) ≤ δ / 2 :=
  sorry

/-- **Part 2 — the loop terminates.**

Except with probability `δ/2`, some reachable state passes the FNR test, so the loop
returns.

Proof plan: each growth step draws fresh suffixes, and by findability a draw is
accept-preserving with probability `≥ pAP`; once the pool holds enough accept-preserving
suffixes and the prefix count is large enough, every population's vote is decisive on all
but `indecisionLimit` of its mass, so the per-population test passes.  The per-step trigger is
block-local, so `geometric_miss_triggered` gives `(1 − p)^N` and `geom_le` drives it under
`δ/2`.

`hslack : accFnr < indecisionLimit` is **necessary**, not decoration.  `accFnr` is the
binomial probability that one prefix's count lands inside the indecisive band, so it
bounds the *expected* indecision fraction; if the loop's limit were at or below it the
test could essentially never pass and the loop would not terminate.  The code keeps the
slack: `0.01 < 0.02` (`0.10` after PR #257). -/
theorem loop_terminates [IsProbabilityMeasure ν]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (dr : Draws ν μ D Dsf)
    (fpr accFnr indecisionLimit : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) :
    ν.real {x | ∀ hM, x ∉ ret O populations dr fpr accFnr indecisionLimit hM} ≤ δ / 2 :=
  sorry

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and the family it returns —
at whatever state it chooses to stop — preserves acceptance on `≥ 1 − εcov` of **each**
prefix population.

Nothing is fixed or idealised.  The growth schedule is not a parameter: the guarantee is
uniform over every history the algorithm might follow and every budget it might reach, so
it may optimize its own schedule.  The stopping time is likewise arbitrary.  The decision
boundary, evidence margin, thresholds and family size are recomputed at each state from
the oracle's signal, as `build_pst` computes them; the cluster is the Lloyd fixed point
against its own thresholded mean; and the draws are without replacement. -/
theorem clustering_correct [IsProbabilityMeasure ν]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (dr : Draws ν μ D Dsf)
    (fpr accFnr indecisionLimit : ℝ) (hfpr : 0 < fpr) (haccFnr : 0 < accFnr)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty)
    (pAP : ℝ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ) (hindLim : 0 < indecisionLimit)
    (hslack : accFnr < indecisionLimit) :
    1 - δ ≤ ν.real
      {x | (∃ hM, x ∈ ret O populations dr fpr accFnr indecisionLimit hM) ∧
        ∀ hM : Hist × (ℕ × ℕ), ∀ j ∈ populations, 1 - εcov
          ≤ (D j).real {p | ∀ v ∈ famAt O populations dr fpr accFnr x hM.1 hM.2,
              O.label (p * v) = O.label p}} := by
  have h := sound_and_terminating ν
    (FailAt O populations D dr fpr accFnr εcov)
    (ret O populations dr fpr accFnr indecisionLimit) δ
    (validity_of_returned O populations D Dsf dr fpr accFnr hfpr haccFnr hsig hpop
      pAP hpAPPositive hpAPBound εcov hεcov δ hδ)
    (loop_terminates O populations D Dsf dr fpr accFnr indecisionLimit hfpr haccFnr hsig hpop
      pAP hpAPPositive hpAPBound δ hδ hindLim hslack)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, FailAt, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
