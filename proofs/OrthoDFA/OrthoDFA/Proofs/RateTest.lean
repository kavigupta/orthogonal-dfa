import OrthoDFA.RateTest
import OrthoDFA.Proofs.BinomLaw

/-!
# The rate test is sound

The test says above exactly at the counts from some least `K` up, where `P(Bin(n, θ) ≥ K) < a`;
at a rate `p ≤ θ` the count reaches `K` with chance `P(Bin(n, p) ≥ K)`, which is no more. Below
is the mirror image, from the greatest count whose lower tail under `θ` is below `a`.
-/

namespace OrthoDFA

open MeasureTheory

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) (θ a : ℝ) (n₀ n : ℕ)

open scoped Classical in
theorem rateSide_above_le (hθ1 : θ ≤ 1) (ha : 0 ≤ a) (hp : D.real {x | P x} ≤ θ) :
    (Measure.pi fun _ : Fin n => D).real
      {xs | rateSide θ a n₀ n (Finset.univ.filter fun i => P (xs i)).card = some true} ≤ a := by
  set ν := Measure.pi fun _ : Fin n => D
  set E := {xs : Fin n → X |
    rateSide θ a n₀ n (Finset.univ.filter fun i => P (xs i)).card = some true}
  by_cases hfire : ∃ k, binomSfGe n θ k < a
  swap
  · simp only [not_exists, not_lt] at hfire
    have : E = ∅ := by
      refine Set.eq_empty_of_forall_notMem fun xs hxs => ?_
      simp only [E, Set.mem_ofPred_eq, rateSide] at hxs
      split_ifs at hxs with h1 h2
      · exact (not_lt.2 (hfire _)) h2
      all_goals simp at hxs
    rw [this, measureReal_empty]; exact ha
  set K := Nat.find hfire
  have hK : binomSfGe n θ K < a := Nat.find_spec hfire
  have hsub : E ⊆ {xs | K ≤ (Finset.univ.filter fun i => P (xs i)).card} := by
    intro xs hxs
    simp only [E, Set.mem_ofPred_eq, rateSide] at hxs
    split_ifs at hxs with h1 h2
    · exact Nat.find_min' hfire h2
    all_goals simp at hxs
  have hcount := pi_count_ge D P (Finset.univ : Finset (Fin n)) K
  rw [Finset.card_univ, Fintype.card_fin] at hcount
  calc ν.real E ≤ ν.real {xs | K ≤ (Finset.univ.filter fun i => P (xs i)).card} :=
        measureReal_mono hsub
    _ = binomSfGe n (D.real {x | P x}) K := hcount
    _ ≤ binomSfGe n θ K := binomSfGe_mono measureReal_nonneg hθ1 hp n K
    _ ≤ a := hK.le

open scoped Classical in
theorem rateSide_below_le (hθ0 : 0 ≤ θ) (ha : 0 ≤ a) (hp : θ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin n => D).real
      {xs | rateSide θ a n₀ n (Finset.univ.filter fun i => P (xs i)).card = some false} ≤ a := by
  set ν := Measure.pi fun _ : Fin n => D
  set E := {xs : Fin n → X |
    rateSide θ a n₀ n (Finset.univ.filter fun i => P (xs i)).card = some false}
  set Q : ℕ → Prop := fun k => 1 - binomSfGe n θ (k + 1) < a
  set K := Nat.findGreatest Q n
  have hcard : ∀ xs : Fin n → X, (Finset.univ.filter fun i => P (xs i)).card ≤ n := fun xs =>
    (Finset.card_filter_le _ _).trans (by simp)
  have hsub : E ⊆ {xs | K + 1 ≤ (Finset.univ.filter fun i => P (xs i)).card}ᶜ := by
    intro xs hxs
    simp only [E, Set.mem_ofPred_eq, rateSide] at hxs
    simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_le, Nat.lt_succ_iff]
    split_ifs at hxs with h1 h2 h3
    · simp at hxs
    · exact Nat.le_findGreatest (hcard xs) h3
  by_cases hE : E = ∅
  · rw [hE, measureReal_empty]; exact ha
  obtain ⟨xs, hxs⟩ := Set.nonempty_iff_ne_empty.2 hE
  have hQ : Q K := by
    have h := hxs
    simp only [E, Set.mem_ofPred_eq, rateSide] at h
    split_ifs at h with h1 h2 h3
    · simp at h
    · exact Nat.findGreatest_spec (hcard xs) h3
  have hcount := pi_count_ge D P (Finset.univ : Finset (Fin n)) (K + 1)
  rw [Finset.card_univ, Fintype.card_fin] at hcount
  have hmeas : MeasurableSet
      {xs : Fin n → X | K + 1 ≤ (Finset.univ.filter fun i => P (xs i)).card} :=
    (Set.to_countable _).measurableSet
  calc ν.real E ≤ ν.real {xs | K + 1 ≤ (Finset.univ.filter fun i => P (xs i)).card}ᶜ :=
        measureReal_mono hsub
    _ = 1 - binomSfGe n (D.real {x | P x}) (K + 1) := by
        rw [measureReal_compl hmeas, probReal_univ, hcount]
    _ ≤ 1 - binomSfGe n θ (K + 1) := by
        have := binomSfGe_mono hθ0 (by
          have := measureReal_mono (μ := D) (Set.subset_univ {x | P x})
          rwa [probReal_univ] at this) hp n (K + 1)
        linarith
    _ ≤ a := hQ.le

theorem rateTestSound : RateTestSound := by
  intro X _ _ _ D _ P θ a n₀ n hθ0 hθ1 ha
  exact ⟨rateSide_above_le D P θ a n₀ n hθ1 ha, rateSide_below_le D P θ a n₀ n hθ0 ha⟩

end OrthoDFA
