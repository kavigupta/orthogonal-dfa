import OrthoDFA.RoundEnd

/-!
# A sampler's law

A sampler that draws `x` from `D` and finds the strings `f x` puts on a set `A` the mean count of
`f x`'s strings in `A`. It is spread from `k` when each draw finds a string at most once and only
one that starts as the draw does for `k` letters.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Countable α]

theorem diracSum_apply (l : List (FreeMonoid α)) (A : Set (FreeMonoid α))
    [DecidablePred (· ∈ A)] :
    (l.map Measure.dirac).sum A = ((l.filter (· ∈ A)).length : ℝ≥0∞) := by
  induction l with
  | nil => simp
  | cons z l ih =>
    rw [List.map_cons, List.sum_cons, Measure.add_apply, ih,
      Measure.dirac_apply' _ (Set.to_countable A).measurableSet]
    by_cases hz : z ∈ A
    · simp [hz, add_comm]
    · simp [hz]

theorem bind_diracSum_apply (D : Measure (FreeMonoid α))
    (f : FreeMonoid α → List (FreeMonoid α)) (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    (D.bind fun x => ((f x).map Measure.dirac).sum) A
      = ∫⁻ x, (((f x).filter (· ∈ A)).length : ℝ≥0∞) ∂D := by
  rw [Measure.bind_apply (Set.to_countable A).measurableSet
    (measurable_of_countable _).aemeasurable]
  simp only [diracSum_apply]

theorem bind_diracSum_of_forall (D : Measure (FreeMonoid α))
    (f : FreeMonoid α → List (FreeMonoid α)) (A : Set (FreeMonoid α)) (h : ∀ x, ∀ z ∈ f x, z ∈ A) :
    (D.bind fun x => ((f x).map Measure.dirac).sum) A
      = (D.bind fun x => ((f x).map Measure.dirac).sum) Set.univ := by
  classical
  refine measure_congr (ae_eq_univ.2 ?_)
  rw [bind_diracSum_apply]
  refine lintegral_eq_zero_iff (measurable_of_countable _) |>.2 (ae_of_all _ fun x => ?_)
  simp only [Pi.zero_apply, Nat.cast_eq_zero, List.length_eq_zero_iff, List.filter_eq_nil_iff,
    Set.mem_compl_iff, decide_eq_true_eq, not_not]
  exact h x

/-- Each draw finds a string at most once, and only one that starts as the draw does for `k`
letters. -/
theorem spread_of_fresh (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (k : ℕ)
    (f : FreeMonoid α → List (FreeMonoid α)) (hnd : ∀ x, (f x).Nodup)
    (hpre : ∀ x, ∀ z ∈ f x, z.toList.take k = x.toList.take k) :
    Spread D k (D.bind fun x => ((f x).map Measure.dirac).sum) := by
  classical
  intro t
  set B := {p : FreeMonoid α | p.toList.take k = t.toList.take k}
  have hle : (D.bind fun x => ((f x).map Measure.dirac).sum) {t} ≤ D B := by
    rw [bind_diracSum_apply, ← lintegral_indicator_one (Set.to_countable B).measurableSet]
    refine lintegral_mono fun x => ?_
    by_cases ht : t ∈ f x
    · have hlen : ((f x).filter (· ∈ ({t} : Set _))).length ≤ 1 := by
        rw [← List.toFinset_card_of_nodup ((hnd x).filter _)]
        refine (Finset.card_le_card (t := {t}) fun a ha => ?_).trans (by simp)
        simp only [List.mem_toFinset, List.mem_filter, Set.mem_singleton_iff,
          decide_eq_true_eq] at ha
        simp [ha.2]
      have hx : x ∈ B := (hpre x t ht).symm
      rw [Set.indicator_of_mem hx, Pi.one_apply]
      calc _ ≤ ((1 : ℕ) : ℝ≥0∞) := Nat.cast_le.2 hlen
        _ = 1 := Nat.cast_one
    · have : (f x).filter (· ∈ ({t} : Set _)) = [] := by
        refine List.filter_eq_nil_iff.2 fun a ha => ?_
        simp only [Set.mem_singleton_iff, decide_eq_true_eq]
        rintro rfl
        exact ht ha
      rw [this]
      simp
  exact ENNReal.toReal_mono (measure_ne_top D B) hle

end OrthoDFA
