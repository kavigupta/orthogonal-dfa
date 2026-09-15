import OrthoDFA.Liveness
import OrthoDFA.Complexity
import OrthoDFA.Termination

/-!
# The distributional clustering guarantee (PR #257) — target statement + proof

The clustering algorithm's own guarantee, distributional and **per population**, as in
PR #257 ("Hold every prefix population to the FNR limit"):

* prefixes come from a **collection** of distributions `D : J → Measure S` (the uniform
  pool, the boundary set, one per state) held individually;
* candidate suffixes are drawn from a suffix distribution `Dsf`, and **findability** is a
  single number `pAP` — the probability a drawn suffix is accept-preserving on the true
  noiseless oracle (`∀ p, ℓ(p·v) = ℓ(p)`);
* the seed is `ε = 1` (so accept-preserving = the seed's Nerode class);
* the guarantee: w.p. `≥ 1 − δ`, the returned family preserves acceptance on `≥ 1 − εcov`
  of **each** population `D j`.

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

/-- **Coverage.**  If every suffix in the family `F` flips at most `β` of population
`Dj`'s prefix mass, then the family preserves acceptance on at least `1 − #F·β` of `Dj`:
a union bound over the family's flip sets. -/
theorem coverage (O : Oracle μ S)
    (Dj : Measure S) [IsProbabilityMeasure Dj]
    (F : Finset S) (β : ℝ)
    (hmeas : ∀ v ∈ F, MeasurableSet {p | O.label (p * v) ≠ O.label p})
    (hF : ∀ v ∈ F, Dj.real {p | O.label (p * v) ≠ O.label p} ≤ β) :
    1 - (F.card : ℝ) * β ≤ Dj.real {p | ∀ v ∈ F, O.label (p * v) = O.label p} := by
  classical
  -- the "some family member flips p" set is the finite union of the per-suffix flip sets
  have hunion : {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p}
      = ⋃ v ∈ F, {p | O.label (p * v) ≠ O.label p} := by
    ext p; simp only [Set.mem_setOf_eq, Set.mem_iUnion, exists_prop]
    push_neg; rfl
  have hbad : Dj.real {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} ≤ (F.card : ℝ) * β := by
    rw [hunion]
    calc Dj.real (⋃ v ∈ F, {p | O.label (p * v) ≠ O.label p})
        ≤ ∑ v ∈ F, Dj.real {p | O.label (p * v) ≠ O.label p} := measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ F, β := Finset.sum_le_sum hF
      _ = (F.card : ℝ) * β := by rw [Finset.sum_const, nsmul_eq_mul]
  -- the preservation set is the complement of the flip set
  have hcompl : Dj.real {p | ∀ v ∈ F, O.label (p * v) = O.label p}
      = 1 - Dj.real {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} := by
    have hmeasBad : MeasurableSet {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} := by
      rw [hunion]; exact F.measurableSet_biUnion (fun v hv => hmeas v hv)
    have huniv : Dj.real Set.univ = 1 := by simp [measureReal_def, measure_univ]
    have hset : {p | ∀ v ∈ F, O.label (p * v) = O.label p}
        = {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p}ᶜ := by
      ext p; simp
    rw [hset, measureReal_compl hmeasBad, huniv]
  rw [hcompl]; linarith [hbad]

#print axioms coverage

/-- **Findability.**  Over `M` i.i.d. suffix draws from `Dsf`, each accept-preserving with
probability `≥ pAP`, the probability that *none* is accept-preserving is `≤ (1−pAP)^M`.
A direct instance of the product-measure `geometric_miss`; `geom_le` then drives it below
any budget once `M ≥ log(1/·)/pAP`. -/
theorem findAP (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (pAP : ℝ) (AP : Set S) (hAPmeas : MeasurableSet AP)
    (hpAP1 : pAP ≤ 1) (hfind : pAP ≤ Dsf.real AP) :
    (Measure.pi (fun _ : Fin M => Dsf)).real {x | ∀ i, x i ∉ AP} ≤ (1 - pAP) ^ M :=
  geometric_miss (fun _ => Dsf) (fun _ => AP) (fun _ => hAPmeas) pAP hpAP1 (fun _ => hfind)

#print axioms findAP

end OrthoDFA
