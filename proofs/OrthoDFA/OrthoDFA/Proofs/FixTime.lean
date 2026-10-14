import OrthoDFA.Proofs.BinomLaw
import Mathlib.Probability.ProductMeasure

/-!
# The first draw of a product measure
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {X S H : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

omit [Countable X] [MeasurableSingletonClass X] in
theorem piFinSuccAbove_zero' {n : ℕ} (d : Fin (n + 1) → X) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) 0 d = (d 0, Fin.tail d) := by
  ext j
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove, Fin.tail]

/-- The chance of `E` over `T + 1` draws, as the first draw's average of its section. -/
theorem pi_succ_apply (D : Measure X) [IsProbabilityMeasure D] (T : ℕ)
    (E : Set (Fin (T + 1) → X)) :
    (Measure.pi fun _ : Fin (T + 1) => D) E
      = ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈ E} ∂D := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (T + 1) => D) 0
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (T + 1) => X) 0
  have hE : E = e ⁻¹' {p | Fin.cons p.1 p.2 ∈ E} := by
    ext d
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, e, piFinSuccAbove_zero', Fin.cons_self_tail]
  rw [hE, hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
    Measure.prod_apply (Set.to_countable _).measurableSet]
  rfl

variable (D : Measure X) [IsProbabilityMeasure D] (step : S → X → Option S) (key : S → H)
  (A : H → Set X) (cnt : S → ℕ) (sig : H → Prop) (q : ℝ) (m : ℕ)

end OrthoDFA
