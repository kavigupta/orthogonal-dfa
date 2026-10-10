import OrthoDFA.Proofs.BinomLaw
import Mathlib.Probability.ProductMeasure

/-!
# A significant edge is fixed in time

A round driven by fresh draws: `step s x` is the state after the draw `x`, or `none` once the
round has ended. Each state has a hypothesis `key s`. Where the hypothesis is significant, a
fresh draw lies in its record set `A (key s)` with chance at least `q`; each such draw raises the
counter `cnt` while the hypothesis stays, and the counter never reaches `m`, since `m` records
force a fix.

`fix_in_time`: over `T` draws, the chance that at some point a significant hypothesis outlasts
the next `n` draws, the round still running, is at most `T · P(Bin(n, q) < m)`.
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
