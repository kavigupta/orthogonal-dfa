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

/-- The membership query the oracle actually answers: `MQ w = ℓ(w) ⊕ noise(w)`. -/
noncomputable def mq (O : Oracle μ S) (w : S) (ω : Ω) : ℝ :=
  O.label w + (1 - 2 * O.label w) * O.noise w ω

/-- What a run draws: the oracle's persistent noise sample, an unbounded i.i.d. stream of
candidate suffixes from `Dsf`, and for each population an unbounded i.i.d. stream of
prefixes from `D j` — with their laws and independence.  This is the algorithm's
randomness; nothing here is left unconstrained. -/
structure Draws {Ξ : Type*} [MeasurableSpace Ξ] (ν : Measure Ξ) (μ : Measure Ω)
    {J : Type*} (D : J → Measure S) (Dsf : Measure S) where
  nz : Ξ → Ω
  sfx : ℕ → Ξ → S
  prf : J → ℕ → Ξ → S
  meas_nz : Measurable nz
  meas_sfx : ∀ i, Measurable (sfx i)
  meas_prf : ∀ j i, Measurable (prf j i)
  law_nz : Measure.map nz ν = μ
  law_sfx : ∀ i, Measure.map (sfx i) ν = Dsf
  law_prf : ∀ j i, Measure.map (prf j i) ν = D j
  indep_sfx : iIndepFun sfx ν
  indep_prf : iIndepFun (fun q : J × ℕ => prf q.1 q.2) ν
  indep_nz_sfx : ∀ i, IndepFun nz (sfx i) ν
  indep_nz_prf : ∀ j i, IndepFun nz (prf j i) ν

section Loop

variable {Ξ : Type*} [MeasurableSpace Ξ] {ν : Measure Ξ} {J : Type*}
  {D : J → Measure S} {Dsf : Measure S}

/-- Round `t`'s accumulated candidate suffix pool: the first `Mc t` suffixes drawn. -/
noncomputable def pool (dr : Draws ν μ D Dsf) (Mc : ℕ → ℕ) (x : Ξ) (t : ℕ) : Finset S :=
  (Finset.range (Mc t)).image (fun i => dr.sfx i x)

/-- The family proposed at round `t`: the least-loss `k`-subset of the accumulated pool,
scored by the summed oracle reads over the round's prefixes across all populations. -/
noncomputable def famAt (O : Oracle μ S) (populations : Finset J) (k : ℕ)
    (dr : Draws ν μ D Dsf) (Mc mc : ℕ → ℕ) (x : Ξ) (t : ℕ) : Finset S :=
  leastLossSubset
    (fun v => ∑ j ∈ populations, ∑ i ∈ Finset.range (mc t),
      O.read (fun i' : ℕ => dr.prf j i' x) v i (dr.nz x))
    (pool dr Mc x t) k

/-- The family's vote on a prefix: the mean membership query over the family. -/
noncomputable def vote (O : Oracle μ S) (F : Finset S) (p : S) (ω : Ω) : ℝ :=
  (∑ v ∈ F, mq O (p * v) ω) / F.card

/-- A prefix is *decided* by the family when its vote clears one of the thresholds;
otherwise it lands in the indecisive band and counts towards the FNR. -/
def decided (O : Oracle μ S) (accThresh rejThresh : ℝ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  accThresh ≤ vote O F p ω ∨ vote O F p ω < rejThresh

open scoped Classical in
/-- **The loop's return test** (PR #257: held per population, not over their union).
The loop returns at round `t` when *every* population's indecisive fraction is within
`fnrLimit`. -/
noncomputable def ret (O : Oracle μ S) (populations : Finset J) (k : ℕ)
    (dr : Draws ν μ D Dsf) (Mc mc : ℕ → ℕ) (accThresh rejThresh fnrLimit : ℝ)
    (t : ℕ) : Set Ξ :=
  {x | ∀ j ∈ populations,
    ((((Finset.range (mc t)).filter (fun i => ¬ decided O accThresh rejThresh
        (famAt O populations k dr Mc mc x t) (dr.prf j i x) (dr.nz x))).card : ℝ))
      ≤ fnrLimit * mc t}

/-- Round `t`'s family is **invalid**: on some population it fails to preserve acceptance
on a `1 − εcov` fraction. -/
def FailAt (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (k : ℕ)
    (dr : Draws ν μ D Dsf) (Mc mc : ℕ → ℕ) (εcov : ℝ) (t : ℕ) : Set Ξ :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | ∀ v ∈ famAt O populations k dr Mc mc x t,
            O.label (p * v) = O.label p}}

/-- **Part 1 — whatever is returned is valid, whenever it is returned.**

Except with probability `δ/2`, *no* round's family is invalid.  Being uniform over the
round is the point: the loop may stop wherever it likes — by the FNR test or any other
rule — and the family it hands back is valid, so the stopping rule never has to be
modelled or itself proved correct.

Proof plan (pieces in `Distributional.lean`): per round this is `clustering_budget` at
that round's budget — the selection avoids suffixes of high distributional flip-mass
(`ploss_good_upper` / `ploss_bad_lower`: two-level concentration against the *persistent*
oracle, plus the collision term), and `coverage_of_summed_flip` turns per-suffix flip
control into the per-population fraction.  Union-bound the rounds at `δ/2^(t+2)`, which
is summable — this is what removes any bound on the number of rounds.  The open point is
the collision mass at large `mc t`; see the module note. -/
theorem validity_of_returned [IsProbabilityMeasure ν]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (k : ℕ) (dr : Draws ν μ D Dsf) (Mc mc : ℕ → ℕ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty) (hk : 0 < k)
    (pAP : ℝ) (hpAP : 0 < pAP)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ)
    (hMc : Filter.Tendsto Mc Filter.atTop Filter.atTop)
    (hmc : Filter.Tendsto mc Filter.atTop Filter.atTop) :
    ν.real (⋃ t, FailAt O populations D k dr Mc mc εcov t) ≤ δ / 2 :=
  sorry

/-- **Part 2 — the loop terminates.**

Except with probability `δ/2`, some round's FNR test passes and the loop returns.

Proof plan: each round draws fresh suffixes, and by findability a draw is
accept-preserving with probability `≥ pAP`; once the pool holds enough accept-preserving
suffixes and the prefix count is large enough, every population's vote is decisive on all
but `fnrLimit` of its mass, so the per-population test passes.  The per-round trigger is
block-local (it depends only on that round's fresh draws), so `geometric_miss_triggered`
gives `(1 − p)^N` and `geom_le` drives it under `δ/2`. -/
theorem loop_terminates [IsProbabilityMeasure ν]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (k : ℕ) (dr : Draws ν μ D Dsf) (Mc mc : ℕ → ℕ)
    (accThresh rejThresh fnrLimit : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty) (hk : 0 < k)
    (pAP : ℝ) (hpAP : 0 < pAP)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (δ : ℝ) (hδ : 0 < δ)
    (hthresh : rejThresh < accThresh) (hfnr : 0 < fnrLimit)
    (hMc : Filter.Tendsto Mc Filter.atTop Filter.atTop)
    (hmc : Filter.Tendsto mc Filter.atTop Filter.atTop) :
    ν.real {x | ∀ t, x ∉ ret O populations k dr Mc mc accThresh rejThresh fnrLimit t}
      ≤ δ / 2 :=
  sorry

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and **whatever** family it
returns preserves acceptance on `≥ 1 − εcov` of **each** prefix population.

No fixed budget: the rounds range over all of `ℕ`, the pool and the prefix counts grow
with the round, and the stopping time is data-dependent and arbitrary. -/
theorem clustering_correct [IsProbabilityMeasure ν]
    (O : Oracle μ S) (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (k : ℕ) (dr : Draws ν μ D Dsf) (Mc mc : ℕ → ℕ)
    (accThresh rejThresh fnrLimit : ℝ)
    (hsig : O.η < 1 / 2) (hpop : populations.Nonempty) (hk : 0 < k)
    (pAP : ℝ) (hpAP : 0 < pAP)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ)
    (hthresh : rejThresh < accThresh) (hfnr : 0 < fnrLimit)
    (hMc : Filter.Tendsto Mc Filter.atTop Filter.atTop)
    (hmc : Filter.Tendsto mc Filter.atTop Filter.atTop) :
    1 - δ ≤ ν.real
      {x | (∃ t, x ∈ ret O populations k dr Mc mc accThresh rejThresh fnrLimit t) ∧
        ∀ t, ∀ j ∈ populations, 1 - εcov
          ≤ (D j).real {p | ∀ v ∈ famAt O populations k dr Mc mc x t,
              O.label (p * v) = O.label p}} := by
  have h := sound_and_terminating ν
    (FailAt O populations D k dr Mc mc εcov)
    (ret O populations k dr Mc mc accThresh rejThresh fnrLimit) δ
    (validity_of_returned O populations D Dsf k dr Mc mc hsig hpop hk pAP hpAP hfind
      εcov hεcov δ hδ hMc hmc)
    (loop_terminates O populations D Dsf k dr Mc mc accThresh rejThresh fnrLimit
      hsig hpop hk pAP hpAP hfind δ hδ hthresh hfnr hMc hmc)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, FailAt, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
