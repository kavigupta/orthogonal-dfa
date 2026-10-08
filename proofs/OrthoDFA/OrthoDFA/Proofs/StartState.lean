import OrthoDFA.StartState
import OrthoDFA.Proofs.GateFlip

/-! # `StartExists`: along a run that stays where `H` agrees, `H` tracks the target -/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Q P : Type*}

theorem step_tracks {A : DFA (FreeMonoid α) Q} {H : DFA (FreeMonoid α) P} {S : Set Q}
    {h : Q → P} (hm : MatchesOn A H S h) {q : Q} {x : FreeMonoid α} (hs : StaysIn A S q x) :
    ∀ i ≤ x.toList.length, H.step (h q) (prefixOf x i) = h (A.step q (prefixOf x i))
  | 0, _ => by simp [prefixOf_zero, A.step_one, H.step_one]
  | i + 1, hi => by
    have hix : i < x.toList.length := by omega
    rw [prefixOf_succ hix, A.step_mul, H.step_mul, step_tracks hm hs i hix.le]
    refine hm.1 _ (hs i hix.le) _ ?_
    rw [← A.step_mul, ← prefixOf_succ hix]
    exact hs (i + 1) hi

theorem start_exists_holds : StartExists := by
  intro α _ _ Q P A H D _ S h q η hm hcov
  have hgood : {x | StaysIn A S q x ∧ (A.step q x ∈ A.accept ↔ A.state x ∈ A.accept)}
      ⊆ {x | ¬ (H.step (h q) x ∈ H.accept ↔ A.state x ∈ A.accept)}ᶜ := by
    rintro x ⟨hs, hag⟩
    have hend := step_tracks hm hs x.toList.length le_rfl
    rw [prefixOf_length] at hend
    intro hn
    apply hn
    have hS : A.step q x ∈ S := by simpa [prefixOf_length] using hs x.toList.length le_rfl
    rw [hend, hm.2 _ hS]
    exact hag
  have h1 := measureReal_mono (μ := D) hgood (measure_ne_top D _)
  rw [measureReal_compl (Set.to_countable _).measurableSet, probReal_univ] at h1
  linarith
end OrthoDFA
