import OrthoDFA.Proofs.LoopMeasurable
import OrthoDFA.Proofs.LoopIndep

/-!
# One round given the rounds before it, on one noise

The whole run reads one noise. Where what the earlier rounds hand on is fixed by the oracle's
answers at a finite set `W` of strings, and round `r`'s bad event by its answers at a finite set
`Z`, the two are independent wherever `W` and `Z` are disjoint (`indep_noiseAlg`). So round `r` is
bad off a collision of the two with chance at most the bound its bad event has on fresh noise
(`persist_stage_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- An event the oracle's answers at the strings of `A` fix is in the σ-algebra they generate. -/
theorem measurableSet_noiseAlg_of_reads (O : Oracle μ (FreeMonoid α)) (A : Finset (FreeMonoid α))
    (E : Set Ω) (hE : ∀ ω ω', (∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) → ω ∈ E → ω' ∈ E) :
    MeasurableSet[noiseAlg O ↑A] E := by
  classical
  have hset : E = {ω | (fun U => ∃ ω' ∈ E, A.filter (fun p => O.mq p ω' = 1) = U)
      (A.filter (fun p => O.mq p ω = 1))} := by
    ext ω
    constructor
    · intro h; exact ⟨ω, h, rfl⟩
    · rintro ⟨ω', hω', hU⟩
      refine hE ω' ω (fun p hp => ?_) hω'
      have := Finset.ext_iff.1 hU p
      simp only [Finset.mem_filter, hp, true_and] at this
      exact this
  rw [hset]
  exact measurableSet_filter_pred O (T := ↑A) (A := A) subset_rfl
    (fun U => ∃ ω' ∈ E, A.filter (fun p => O.mq p ω' = 1) = U)

/-- Two events the answers at disjoint finite sets fix are independent. -/
theorem measure_inter_of_reads (O : Oracle μ (FreeMonoid α)) {W Z : Finset (FreeMonoid α)}
    (hWZ : Disjoint W Z) (H B : Set Ω)
    (hH : ∀ ω ω', (∀ p ∈ W, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) → ω ∈ H → ω' ∈ H)
    (hB : ∀ ω ω', (∀ p ∈ Z, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) → ω ∈ B → ω' ∈ B) :
    μ (H ∩ B) = μ H * μ B :=
  ((indep_noiseAlg O (Finset.disjoint_coe.2 hWZ)).indepSet_of_measurableSet
    (measurableSet_noiseAlg_of_reads O W H hH) (measurableSet_noiseAlg_of_reads O Z B hB)
    ).measure_inter_eq_mul

/-- Round `r` is bad off a collision with chance at most `δ`. The earlier rounds and the round's
handed part are summed up countably by `stage`, which the answers at `W` fix; the round's fresh
part is bad by `Bad`, which the answers at `Z` fix, with chance at most `δ` on fresh noise. -/
theorem persist_stage_le (O : Oracle μ (FreeMonoid α)) {K : ℕ} {X X₁ X₂ : Type*}
    [MeasurableSpace X] [MeasurableSpace X₁] [MeasurableSpace X₂] (ν : Measure X)
    [IsProbabilityMeasure ν] (ν₁ : Measure X₁) [IsProbabilityMeasure ν₁] (ν₂ : Measure X₂)
    [IsProbabilityMeasure ν₂] (σ : X → X₁ × X₂) (hσ : MeasurePreserving σ ν (ν₁.prod ν₂))
    (τ : X₁ × X₂ → X) (hτ : Measurable τ) (hστ : ∀ w, σ (τ w) = w) (r : Fin K) {Y : Type*}
    [Countable Y] (stage : Ω → (Fin K → X) → Y)
    (hdep : ∀ ω (x x' : Fin K → X), (∀ i, i < r → x i = x' i) → (σ (x r)).1 = (σ (x' r)).1 →
      stage ω x = stage ω x')
    (hmeas : ∀ y, MeasurableSet {p : Ω × (Fin K → X) | stage p.1 p.2 = y})
    (W : Y → (Fin K → X) → Finset (FreeMonoid α)) (G : Y → Prop)
    (hW : ∀ y x ω ω', G y → stage ω x = y → (∀ w ∈ W y x, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
      stage ω' x = y)
    (Bad : Y → Set (Ω × X₂)) (hG : ∀ y, ¬ G y → Bad y = ∅) (hmBad : ∀ y, MeasurableSet (Bad y))
    (Z : Y → X₂ → Finset (FreeMonoid α))
    (hZ : ∀ y x₂ ω ω', (∀ w ∈ Z y x₂, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) → (ω, x₂) ∈ Bad y →
      (ω', x₂) ∈ Bad y)
    (Dj : Y → Set (Fin K → X)) (hmDj : ∀ y, MeasurableSet (Dj y))
    (hDj : ∀ y x, G y → x ∈ Dj y → Disjoint (W y x) (Z y (σ (x r)).2))
    (δ : ℝ≥0∞) (hBad : ∀ y, (μ.prod ν₂) (Bad y) ≤ δ) :
    (μ.prod (Measure.pi fun _ : Fin K => ν))
      {p | (p.1, (σ (p.2 r)).2) ∈ Bad (stage p.1 p.2) ∧ p.2 ∈ Dj (stage p.1 p.2)} ≤ δ := by
  classical
  have : Nonempty X₂ := nonempty_of_isProbabilityMeasure ν₂
  obtain ⟨b₀⟩ := this
  set π := Measure.pi fun _ : Fin K => ν
  set π' := Measure.pi fun _ : {i : Fin K // i < r} => ν
  have hσm : Measurable σ := hσ.measurable
  -- the round's fresh part and the rest
  set Φ := fun x : Fin K → X => (((fun i : {i : Fin K // i < r} => x i), (σ (x r)).1),
    (σ (x r)).2)
  have hΦ : MeasurePreserving Φ π ((π'.prod ν₁).prod ν₂) :=
    measurePreserving_stageSplit ν ν₁ ν₂ σ hσ r
  set ext : ({i : Fin K // i < r} → X) × X₁ → Fin K → X := fun w i =>
    if h : i < r then w.1 ⟨i, h⟩ else τ (w.2, b₀)
  have hext : Measurable ext := by
    refine measurable_pi_lambda _ fun i => ?_
    by_cases h : i < r
    · simp only [ext, dif_pos h]
      exact (measurable_pi_apply _).comp measurable_fst
    · simp only [ext, dif_neg h]
      exact hτ.comp (measurable_snd.prodMk measurable_const)
  have hextr : ∀ ω (x : Fin K → X), stage ω (ext (Φ x).1) = stage ω x := by
    intro ω x
    refine hdep ω _ _ (fun i hi => ?_) ?_
    · simp only [ext, Φ, dif_pos hi]
    · simp only [ext, Φ, dif_neg (lt_irrefl r), hστ]
  -- the sets
  set T : Y → Set (Ω × (Fin K → X)) := fun y =>
    {p | stage p.1 p.2 = y} ∩ ({p | (p.1, (σ (p.2 r)).2) ∈ Bad y} ∩ {p | p.2 ∈ Dj y}) with hTd
  have hTm : ∀ y, MeasurableSet (T y) := fun y =>
    (hmeas y).inter ((measurable_fst.prodMk ((measurable_snd.comp hσm).comp
      ((measurable_pi_apply r).comp measurable_snd)) (hmBad y)).inter
      (measurable_snd (hmDj y)))
  have hsub : {p : Ω × (Fin K → X) | (p.1, (σ (p.2 r)).2) ∈ Bad (stage p.1 p.2)
        ∧ p.2 ∈ Dj (stage p.1 p.2)} ⊆ ⋃ y, T y := by
    rintro p ⟨h1, h2⟩
    exact Set.mem_iUnion.2 ⟨stage p.1 p.2, rfl, h1, h2⟩
  -- the stage's chance, as a function of the draws
  set f : Y → (Fin K → X) → ℝ≥0∞ := fun y x => μ {ω | stage ω x = y}
  have hfm : ∀ y, Measurable (f y) := fun y => measurable_measure_prodMk_right (hmeas y)
  set g : Y → X₂ → ℝ≥0∞ := fun y x₂ => μ {ω | (ω, x₂) ∈ Bad y}
  have hgm : ∀ y, Measurable (g y) := fun y => measurable_measure_prodMk_right (hmBad y)
  have hTy : ∀ y, (μ.prod π) (T y) ≤ ∫⁻ x, f y x * g y (σ (x r)).2 ∂π := by
    intro y
    rw [Measure.prod_apply_symm (hTm y)]
    refine lintegral_mono fun x => ?_
    by_cases hGy : G y
    swap
    · have hsec : (fun ω => (ω, x)) ⁻¹' T y = ∅ := by
        ext ω
        simp only [hTd, Set.mem_preimage, Set.mem_inter_iff, Set.mem_setOf_eq, hG y hGy,
          Set.mem_empty_iff_false, false_and, and_false, iff_false, not_false_eq_true]
      rw [hsec, measure_empty]
      exact zero_le
    by_cases hd : x ∈ Dj y
    · have hsec : (fun ω => (ω, x)) ⁻¹' T y
          = {ω | stage ω x = y} ∩ {ω | (ω, (σ (x r)).2) ∈ Bad y} := by
        ext ω
        simp only [hTd, Set.mem_preimage, Set.mem_inter_iff, Set.mem_setOf_eq]
        tauto
      rw [hsec]
      refine le_of_eq (measure_inter_of_reads O (hDj y x hGy hd) _ _ (fun ω ω' h hω => hW y x ω ω' hGy hω h)
        (fun ω ω' h hω => hZ y _ ω ω' h hω))
    · have hsec : (fun ω => (ω, x)) ⁻¹' T y = ∅ := by
        ext ω
        simp only [hTd, Set.mem_preimage, Set.mem_inter_iff, Set.mem_setOf_eq,
          Set.mem_empty_iff_false, iff_false]
        tauto
      rw [hsec, measure_empty]
      exact zero_le
  -- the product splits over the earlier rounds and the fresh part
  have hsplit : ∀ y, ∫⁻ x, f y x * g y (σ (x r)).2 ∂π
      = (∫⁻ w, f y (ext w) ∂(π'.prod ν₁)) * ∫⁻ x₂, g y x₂ ∂ν₂ := by
    intro y
    have hfx : ∀ x, f y x = f y (ext (Φ x).1) := fun x => by
      simp only [f]
      congr 1
      ext ω
      simp only [Set.mem_setOf_eq, hextr]
    calc ∫⁻ x, f y x * g y (σ (x r)).2 ∂π
        = ∫⁻ x, (fun z => f y (ext z.1) * g y z.2) (Φ x) ∂π := by
          refine lintegral_congr fun x => ?_
          simp only [hfx x]
          rfl
      _ = ∫⁻ z, f y (ext z.1) * g y z.2 ∂((π'.prod ν₁).prod ν₂) := by
          rw [← hΦ.lintegral_comp]
          exact ((hfm y).comp (hext.comp measurable_fst)).mul ((hgm y).comp measurable_snd)
      _ = _ := lintegral_prod_mul ((hfm y).comp hext).aemeasurable (hgm y).aemeasurable
  have hgδ : ∀ y, ∫⁻ x₂, g y x₂ ∂ν₂ ≤ δ := fun y => by
    have := Measure.prod_apply_symm (μ := μ) (ν := ν₂) (hmBad y)
    calc ∫⁻ x₂, g y x₂ ∂ν₂ = (μ.prod ν₂) (Bad y) := by rw [this]; rfl
      _ ≤ δ := hBad y
  set A : Y → Set (Ω × (({i : Fin K // i < r} → X) × X₁)) := fun y =>
    {q | stage q.1 (ext q.2) = y}
  have hAm : ∀ y, MeasurableSet (A y) := fun y =>
    (measurable_fst.prodMk (hext.comp measurable_snd)) (hmeas y)
  have hfA : ∀ y, ∫⁻ w, f y (ext w) ∂(π'.prod ν₁) = (μ.prod (π'.prod ν₁)) (A y) := by
    intro y
    rw [Measure.prod_apply_symm (hAm y)]
    rfl
  calc (μ.prod π) {p | (p.1, (σ (p.2 r)).2) ∈ Bad (stage p.1 p.2) ∧ p.2 ∈ Dj (stage p.1 p.2)}
      ≤ (μ.prod π) (⋃ y, T y) := measure_mono hsub
    _ ≤ ∑' y, (μ.prod π) (T y) := measure_iUnion_le _
    _ ≤ ∑' y, (μ.prod (π'.prod ν₁)) (A y) * δ := by
        refine ENNReal.tsum_le_tsum fun y => (hTy y).trans ?_
        rw [hsplit y, hfA y]
        exact mul_le_mul_of_nonneg_left (hgδ y) zero_le
    _ = (μ.prod (π'.prod ν₁)) (⋃ y, A y) * δ := by
        rw [ENNReal.tsum_mul_right, measure_iUnion _ hAm]
        intro y y' hyy'
        refine Set.disjoint_left.2 fun q hq hq' => hyy' ?_
        exact hq.symm.trans hq'
    _ ≤ 1 * δ := mul_le_mul_of_nonneg_right prob_le_one zero_le
    _ = δ := one_mul δ

end OrthoDFA
