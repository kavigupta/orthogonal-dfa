import OrthoDFA.Proofs.Liveness

/-!
# The distributional clustering guarantee (PR #257) — target statement + proof

The clustering algorithm's own guarantee, distributional and per population, as in
PR #257 ("Hold every prefix population to the FNR limit"):

* prefixes come from a collection of distributions `D : J → Measure S` (the uniform
  pool, the boundary set, one per state) held individually;
* candidate suffixes are drawn from a suffix distribution `Dsf`, and findability is a
  single number `pAP` — the probability a drawn suffix is accept-preserving on the true
  noiseless oracle (`∀ p, ℓ(p·v) = ℓ(p)`);
* the seed is `ε = 1` (so accept-preserving = the seed's Nerode class);
* the guarantee: w.p. `≥ 1 − δ`, the returned family preserves acceptance on `≥ 1 − εcov`
  of each population `D j`.

This file builds the proof from reusable pieces.  `coverage` is the first: if every
family member flips at most `β` of a population's mass, the family preserves `≥ 1 − #F·β`
of it — a clean union bound, the step that turns per-suffix flip control into the
per-population fraction.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S]

section Draws
variable [MeasurableMul S]

/-- The `Dj`-flip-mass of a suffix: the probability that `v` flips a `Dj`-drawn prefix.
This is the *distributional* quantity `good`/`bad` are defined by. -/
noncomputable def flipMass (O : Oracle μ S) (Dj : Measure S) (v : S) : ℝ :=
  ∫ p, O.flip v p ∂Dj

lemma flip_meas (O : Oracle μ S) (v : S) : Measurable (fun p => O.flip v p) := by
  have h1 : Measurable (fun p : S => O.label (p * v)) :=
    O.label_meas.comp (measurable_mul_const v)
  show Measurable (fun p => O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
  exact (h1.add O.label_meas).sub ((measurable_const.mul h1).mul O.label_meas)

lemma flip_icc (O : Oracle μ S) (v p : S) : O.flip v p ∈ Set.Icc (0 : ℝ) 1 := by
  rcases O.flip_bit v p with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num

section Assembly
variable [Countable S] [MeasurableSingletonClass S] [DecidableEq S] [IsCancelMul S]

/-- A failure bound gives the complementary success bound, with no measurability
needed: outer measure is subadditive and `s ∪ sᶜ = univ`. -/
theorem one_sub_le_compl_real {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] (s : Set α) (d : ℝ) (h : ν.real s ≤ d) : 1 - d ≤ ν.real sᶜ := by
  have hsub : (1 : ℝ≥0∞) ≤ ν s + ν sᶜ := by
    calc (1 : ℝ≥0∞) = ν Set.univ := measure_univ.symm
      _ = ν (s ∪ sᶜ) := by rw [Set.union_compl_self]
      _ ≤ ν s + ν sᶜ := measure_union_le _ _
  have hreal : (1 : ℝ) ≤ ν.real s + ν.real sᶜ := by
    have := ENNReal.toReal_mono (by finiteness) hsub
    rwa [ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _), ENNReal.toReal_one] at this
  linarith

#print axioms one_sub_le_compl_real

/-- Soundness + termination ⇒ correctness.  The two halves of a retry loop compose by
a union bound: if whatever is returned is valid except w.p. `δ/2` (uniformly over *when*
it is returned), and the loop returns at all except w.p. `δ/2`, then with probability
`≥ 1 − δ` the loop returns something *and* what it returns is valid. -/
theorem sound_and_terminating {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] {T : Type*} [Countable T]
    (Fail Ret : T → Set α) (δ : ℝ)
    (hvalid : ν.real (⋃ t, Fail t) ≤ δ / 2)
    (hterm : ν.real {y | ∀ t, y ∉ Ret t} ≤ δ / 2) :
    1 - δ ≤ ν.real {y | (∃ t, y ∈ Ret t) ∧ ∀ t, y ∉ Fail t} := by
  have hcompl : {y : α | (∃ t, y ∈ Ret t) ∧ ∀ t, y ∉ Fail t}
      = ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t})ᶜ := by
    ext y
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_union, Set.mem_iUnion, not_or,
      not_exists, not_forall, not_not]
    constructor
    · rintro ⟨⟨t, ht⟩, hF⟩
      exact ⟨hF, ⟨t, ht⟩⟩
    · rintro ⟨hF, ⟨t, ht⟩⟩
      exact ⟨⟨t, ht⟩, hF⟩
  rw [hcompl]
  refine one_sub_le_compl_real ν _ δ ?_
  calc ν.real ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t})
      ≤ ν.real (⋃ t, Fail t) + ν.real {y | ∀ t, y ∉ Ret t} := measureReal_union_le _ _
    _ ≤ δ / 2 + δ / 2 := add_le_add hvalid hterm
    _ = δ := by ring

#print axioms sound_and_terminating

end Assembly

end Draws

end OrthoDFA
