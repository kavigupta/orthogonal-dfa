import Mathlib.Probability.Independence.Basic
import Mathlib.MeasureTheory.Constructions.Pi

/-!
# One round given the rounds before it

`K` rounds' draws are independent. Round `r`'s draw splits, measure-preservingly, into a part the
rest of the round is handed and a fresh part. Where a countable summary of the earlier rounds and
the handed part has measurable level sets, and at each of its values the fresh part is bad with
chance at most `δ`, the round is bad with chance at most `δ` (`round_stage_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

/-- Round `r`'s draws split into the earlier rounds' with this round's first part, against this
round's second part. -/
theorem measurePreserving_stageSplit {K : ℕ} {X X₁ X₂ : Type*} [MeasurableSpace X]
    [MeasurableSpace X₁] [MeasurableSpace X₂] (ν : Measure X) [IsProbabilityMeasure ν]
    (ν₁ : Measure X₁) [IsProbabilityMeasure ν₁] (ν₂ : Measure X₂) [IsProbabilityMeasure ν₂]
    (σ : X → X₁ × X₂) (hσ : MeasurePreserving σ ν (ν₁.prod ν₂)) (r : Fin K) :
    MeasurePreserving (fun x : Fin K → X =>
        (((fun i : {i : Fin K // i < r} => x i), (σ (x r)).1), (σ (x r)).2))
      (Measure.pi fun _ : Fin K => ν)
      (((Measure.pi fun _ : {i : Fin K // i < r} => ν).prod ν₁).prod ν₂) := by
  set π := Measure.pi fun _ : {i : Fin K // i < r} => ν
  have hsplit : MeasurePreserving
      (fun x : Fin K → X => ((fun i : {i : Fin K // i < r} => x i), x r))
      (Measure.pi fun _ : Fin K => ν) (π.prod ν) :=
    ((MeasurePreserving.id π).prod
      (measurePreserving_eval (fun _ : {i : Fin K // ¬ i < r} => ν) ⟨r, lt_irrefl r⟩)).comp
      (measurePreserving_piEquivPiSubtypeProd (fun _ : Fin K => ν) (fun i : Fin K => i < r))
  have hσ' := (MeasurePreserving.id π).prod hσ
  have e := (measurePreserving_prodAssoc π ν₁ ν₂).symm
  exact e.comp (hσ'.comp hsplit)

/-- Round `r` is bad with chance at most `δ`, where the earlier rounds and its handed part are
summed up countably and measurably and each value leaves its fresh part bad at most `δ`. -/
theorem round_stage_le {K : ℕ} {X X₁ X₂ : Type*} [MeasurableSpace X] [MeasurableSpace X₁]
    [MeasurableSpace X₂] (ν : Measure X) [IsProbabilityMeasure ν] (ν₁ : Measure X₁)
    [IsProbabilityMeasure ν₁] (ν₂ : Measure X₂) [IsProbabilityMeasure ν₂]
    (σ : X → X₁ × X₂) (hσ : MeasurePreserving σ ν (ν₁.prod ν₂)) (τ : X₁ × X₂ → X)
    (hτ : Measurable τ) (hστ : ∀ w, σ (τ w) = w) (r : Fin K) {Y : Type*} [Countable Y]
    (stage : (Fin K → X) → Y)
    (hdep : ∀ x x' : Fin K → X, (∀ i, i < r → x i = x' i) → (σ (x r)).1 = (σ (x' r)).1 →
      stage x = stage x')
    (hmeas : ∀ y, MeasurableSet {x | stage x = y})
    (Bad : Y → Set X₂) (δ : ℝ≥0∞) (hBad : ∀ y, ν₂ (Bad y) ≤ δ) :
    (Measure.pi fun _ : Fin K => ν) {x | (σ (x r)).2 ∈ Bad (stage x)} ≤ δ := by
  classical
  have : Nonempty X₂ := nonempty_of_isProbabilityMeasure ν₂
  obtain ⟨b₀⟩ := this
  set π := Measure.pi fun _ : {i : Fin K // i < r} => ν
  set Φ := fun x : Fin K → X => (((fun i : {i : Fin K // i < r} => x i), (σ (x r)).1),
    (σ (x r)).2)
  have hΦ := measurePreserving_stageSplit ν ν₁ ν₂ σ hσ r
  -- rebuild a draw from its earlier rounds and handed part
  set ext : ({i : Fin K // i < r} → X) × X₁ → Fin K → X := fun w i =>
    if h : i < r then w.1 ⟨i, h⟩ else τ (w.2, b₀)
  have hext : Measurable ext := by
    refine measurable_pi_lambda _ fun i => ?_
    by_cases h : i < r
    · simp only [ext, dif_pos h]
      exact (measurable_pi_apply _).comp measurable_fst
    · simp only [ext, dif_neg h]
      exact hτ.comp (measurable_snd.prodMk measurable_const)
  have hextr : ∀ x : Fin K → X, stage (ext (Φ x).1) = stage x := by
    intro x
    refine hdep _ _ (fun i hi => ?_) ?_
    · simp only [ext, Φ, dif_pos hi]
    · simp only [ext, Φ, dif_neg (lt_irrefl r), hστ]
  set A : Y → Set (({i : Fin K // i < r} → X) × X₁) := fun y => {w | stage (ext w) = y}
  have hA : ∀ y, MeasurableSet (A y) := fun y => hext (hmeas y)
  set B' : Y → Set X₂ := fun y => toMeasurable ν₂ (Bad y)
  have hsub : {x : Fin K → X | (σ (x r)).2 ∈ Bad (stage x)}
      ⊆ ⋃ y, Φ ⁻¹' (A y ×ˢ B' y) := by
    intro x hx
    refine Set.mem_iUnion.2 ⟨stage x, ?_, subset_toMeasurable _ _ hx⟩
    exact hextr x
  calc (Measure.pi fun _ : Fin K => ν) {x | (σ (x r)).2 ∈ Bad (stage x)}
      ≤ (Measure.pi fun _ : Fin K => ν) (⋃ y, Φ ⁻¹' (A y ×ˢ B' y)) := measure_mono hsub
    _ ≤ ∑' y, (Measure.pi fun _ : Fin K => ν) (Φ ⁻¹' (A y ×ˢ B' y)) := measure_iUnion_le _
    _ = ∑' y, (π.prod ν₁) (A y) * ν₂ (Bad y) := by
        refine tsum_congr fun y => ?_
        rw [hΦ.measure_preimage ((hA y).prod (measurableSet_toMeasurable _ _)).nullMeasurableSet,
          Measure.prod_prod, measure_toMeasurable]
    _ ≤ ∑' y, (π.prod ν₁) (A y) * δ := ENNReal.tsum_le_tsum fun y => mul_le_mul_of_nonneg_left (hBad y) zero_le
    _ = (π.prod ν₁) (⋃ y, A y) * δ := by
        rw [ENNReal.tsum_mul_right, measure_iUnion]
        · intro y y' hyy'
          refine Set.disjoint_left.2 fun w hw hw' => hyy' ?_
          exact hw.symm.trans hw'
        · exact hA
    _ ≤ δ := by
        calc (π.prod ν₁) (⋃ y, A y) * δ ≤ 1 * δ := mul_le_mul_of_nonneg_right prob_le_one zero_le
          _ = δ := one_mul δ

end OrthoDFA
