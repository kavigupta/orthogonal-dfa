import OrthoDFA.Proofs.EpsSteps

/-!
# Level 4's runs, step by step

The state before each probe while the round runs (`stateAt`), what a split re-sifts at most
(`resifts_card`), and a mean over fresh draws bounded point by point (`lintegral_le_of_points`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (C : Level3Settings)

/-- The state before the `j`-th probe of `xs`, from `s`, while the round runs. -/
noncomputable def stateAt (read : FreeMonoid α → ARU) :
    Level3State α → List (FreeMonoid α) → ℕ → Option (Level3State α)
  | s, _, 0 => some s
  | _, [], _ + 1 => none
  | s, x :: xs, j + 1 =>
    match applyProbe read C s x with
    | .inl s' => stateAt read s' xs j
    | .inr _ => none

omit [Fintype σ] in
theorem stateAt_succ (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α) (j : ℕ),
      stateAt C read s xs (j + 1) = (stateAt C read s xs j).bind fun s' =>
        xs[j]?.bind fun x => (applyProbe read C s' x).elim some fun _ => none
  | [], s, j => by cases j <;> simp [stateAt]
  | x :: xs, s, 0 => by
    simp only [stateAt, Option.bind_some, List.getElem?_cons_zero]
    rcases applyProbe read C s x with s' | e <;> rfl
  | x :: xs, s, j + 1 => by
    simp only [stateAt, List.getElem?_cons_succ]
    rcases applyProbe read C s x with s' | e
    · exact stateAt_succ read xs s' j
    · rfl

omit [Fintype σ] [DecidableEq α] in
theorem resifts_card (s : Level3State α) (T' : DTree α) :
    (resifts s T').card ≤ s.tree.leaves.length * Fintype.card α * (mids T').card := by
  classical
  unfold resifts
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α), (mids T').card :=
        Finset.sum_le_sum fun e _ => by
          split
          · exact Finset.card_image_le
          · simp
    _ ≤ _ := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_product, Finset.card_univ]
        gcongr
        exact List.toFinset_card_le _

instance : MeasurableSingletonClass ARU := ⟨fun _ => trivial⟩

/-- A measure on a countable type is the sum of its points'. -/
theorem measure_eq_tsum {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (ν : Measure X) (A : Set X) : ν A = ∑' d, ν {d} * A.indicator 1 d := by
  have hA : MeasurableSet A := (Set.to_countable A).measurableSet
  conv_lhs => rw [← Measure.sum_smul_dirac ν]
  rw [Measure.sum_apply _ hA]
  congr 1
  funext d
  rw [Measure.smul_apply, Measure.dirac_apply' _ hA, smul_eq_mul]

/-- Over fresh draws `d`, each drawn point bounding the chance of `{ω | d ∈ S ω}` by `c`, the mean
mass of `S` is at most `c`. -/
theorem lintegral_le_of_points {Ω X : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X] (ν : Measure X)
    [IsProbabilityMeasure ν] (S : Ω → Set X) (c : ENNReal)
    (h : ∀ d, ν {d} ≠ 0 → μ {ω | d ∈ S ω} ≤ c) :
    ∫⁻ ω, ν (S ω) ∂μ ≤ c := by
  set E : X → Set Ω := fun d => toMeasurable μ {ω | d ∈ S ω}
  have hpt : ∀ ω, ν (S ω) ≤ ∑' d, ν {d} * (E d).indicator 1 ω := fun ω => by
    rw [measure_eq_tsum ν]
    refine ENNReal.tsum_le_tsum fun d => ?_
    gcongr
    by_cases hd : d ∈ S ω
    · rw [Set.indicator_of_mem hd,
        Set.indicator_of_mem (show ω ∈ E d from subset_toMeasurable _ _ hd)]
      simp
    · rw [Set.indicator_of_notMem hd]
      exact bot_le
  calc _ ≤ ∫⁻ ω, ∑' d, ν {d} * (E d).indicator 1 ω ∂μ := lintegral_mono hpt
    _ = ∑' d, ν {d} * μ (E d) := by
        rw [lintegral_tsum fun d => ((measurable_one.indicator
          (measurableSet_toMeasurable _ _)).const_mul _).aemeasurable]
        congr 1
        funext d
        rw [lintegral_const_mul _ (measurable_one.indicator (measurableSet_toMeasurable _ _)),
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ ∑' d, ν {d} * c := ENNReal.tsum_le_tsum fun d => by
        by_cases hd : ν {d} = 0
        · rw [hd, zero_mul, zero_mul]
        · rw [measure_toMeasurable]
          gcongr
          exact h d hd
    _ = c := by
        have h1 : ∑' d, ν {d} = 1 := by
          rw [← measure_univ (μ := ν), measure_eq_tsum ν Set.univ]
          simp
        rw [ENNReal.tsum_mul_right, h1, one_mul]

end Eps

end Random

end OrthoDFA
