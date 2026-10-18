import OrthoDFA.Proofs.LoopDFA

/-!
# The learner's DFA under the relative rule: the proof

At `c = min c₀ (ξ/(2C|Q|))` the states covered at `c₀` are covered. The dominated states visited
from `⌈L/2⌉` on by fewer than `c` of the strings are visited from `q` by at most `|Q| C c ≤ ξ/2` of
them, and off those runs the relative rule's good strings are the covered rule's
(`covered_of_dominated`). The claims are `LearnerLoopDFASimple` and `LearnerLoopDFA` at this `c`,
at `η + ξ/2` and `ξ/2`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]

omit [DecidableEq α] in
theorem covered_of_dominated {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (L : ℕ)
    {c₀ C η ξ : ℝ} (hc₀ : 0 < c₀) (hC : 0 < C) (hξ : 0 < ξ) {q : σ}
    (hq : 1 - η ≤ rerootAccuracy M L (coveredStates M L c₀ ∪ dominatedStates M L C q) q) :
    1 - (η + ξ / 2)
      ≤ rerootAccuracy M L (coveredStates M L (min c₀ (ξ / (2 * C * Fintype.card σ)))) q := by
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ : (1 : ℝ) ≤ Fintype.card σ := by exact_mod_cast Fintype.card_pos
  set c := min c₀ (ξ / (2 * C * Fintype.card σ)) with hc
  have hc0 : 0 < c := lt_min hc₀ (div_pos hξ (mul_pos (mul_pos two_pos hC) (by linarith)))
  set D := dominatedStates M L C q
  set μL : Measure (FreeMonoid α) := uniformStrings L
  let T : Finset σ := Finset.univ.filter fun q' => q' ∈ D ∧ windowVisitRate M L q' < c
  let B : Set (FreeMonoid α) :=
    ⋃ q' ∈ T, {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList = q'}
  let GD : Set (FreeMonoid α) :=
    {x | (∀ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ coveredStates M L c₀ ∪ D)
      ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}
  let GC : Set (FreeMonoid α) :=
    {x | (∀ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ coveredStates M L c)
      ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}
  have hsub : GD ⊆ GC ∪ B := by
    rintro x ⟨hD, hacc⟩
    by_cases hx : x ∈ B
    · exact Or.inr hx
    · refine Or.inl ⟨fun i hi => ?_, hacc⟩
      change c ≤ windowVisitRate M L _
      rcases hD i hi with hcov | hDi
      · exact (min_le_left _ _).trans hcov
      by_contra hlt
      exact hx (Set.mem_iUnion₂.2 ⟨_, Finset.mem_filter.2
        ⟨Finset.mem_univ _, hDi, not_le.1 hlt⟩, i, hi, rfl⟩)
  have hB : μL.real B ≤ ξ / 2 := by
    refine (measureReal_biUnion_finset_le _ _).trans ?_
    calc ∑ q' ∈ T, μL.real {x | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList = q'}
        ≤ ∑ _q' ∈ T, C * c := Finset.sum_le_sum fun q' hq' => by
          obtain ⟨-, hD', hlt⟩ := Finset.mem_filter.1 hq'
          exact hD'.trans (mul_le_mul_of_nonneg_left hlt.le hC.le)
      _ ≤ Fintype.card σ * (C * (ξ / (2 * C * Fintype.card σ))) := by
          rw [Finset.sum_const, nsmul_eq_mul]
          exact mul_le_mul (by exact_mod_cast Finset.card_le_univ T)
            (mul_le_mul_of_nonneg_left (min_le_right _ _) hC.le) (mul_pos hC hc0).le
            (by linarith)
      _ = ξ / 2 := by field_simp
  have hmono : μL.real GD ≤ μL.real (GC ∪ B) := measureReal_mono hsub
  have hu : μL.real (GC ∪ B) ≤ μL.real GC + μL.real B := measureReal_union_le _ _
  have hq' : 1 - η ≤ μL.real GD := hq
  change 1 - (η + ξ / 2) ≤ μL.real GC
  linarith

theorem learner_loop_dfa_relative_simple_holds : LearnerLoopDFARelativeSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c₀ C η δ pAP ξ Q A c k L₁ N εd εcov δ' ι ρ s τ J V Kp G PB
    RB cap n hL hη hη₀ hc₀ hC hδ hδ1 hpAP hξ hA2 hpAPu hdom hk1 hk2 hk3
  obtain ⟨q, hcov, hq⟩ := hdom
  have : Nonempty σ := ⟨M.start⟩
  have hQ : (1 : ℝ) ≤ Q := by
    change (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hc : 0 < c := lt_min hc₀ (div_pos hξ (mul_pos (mul_pos two_pos hC) (by linarith)))
  have hq₁ : q ∈ coveredStates M L c := by
    change c ≤ _
    exact (min_le_left c₀ _).trans hcov
  have h := learner_loop_dfa_simple_holds M O L η₀ c (η + ξ / 2) δ pAP (ξ / 2)
  dsimp only at h
  rw [show ∀ x : ℝ, ξ / 2 / (8 * x) = ξ / (16 * x) from fun x => by ring,
    show ∀ x : ℝ, 32 * x / (ξ / 2) ^ 2 = 128 * x / ξ ^ 2 from fun x => by ring,
    show η + ξ / 2 + ξ / 2 = η + ξ by ring] at h
  exact h hL hη hη₀ hc hδ hδ1 hpAP (by positivity) hA2 hpAPu
    ⟨q, hq₁, covered_of_dominated M L hc₀ hC hξ hq⟩ hk1 hk2 hk3

theorem learner_loop_dfa_relative_holds : LearnerLoopDFARelative := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c₀ C η δ pAP ξ Q A c k L₁ N εd δ' s V Kp G n Ce Θ K₁ B₁ P₀
    R₀ E₁ E E₂ L₀ hL hη hη₀ hc₀ hC hδ hδ1 hpAP hξ hA2 hpAPu hdom hL₀
  obtain ⟨q, hcov, hq⟩ := hdom
  have : Nonempty σ := ⟨M.start⟩
  have hQ : (1 : ℝ) ≤ Q := by
    change (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hc : 0 < c := lt_min hc₀ (div_pos hξ (mul_pos (mul_pos two_pos hC) (by linarith)))
  have hq₁ : q ∈ coveredStates M L c := by
    change c ≤ _
    exact (min_le_left c₀ _).trans hcov
  have h := learner_loop_dfa_holds M O L η₀ c (η + ξ / 2) δ pAP (ξ / 2)
  dsimp only at h
  rw [show (8 : ℝ) / (ξ / 2) = 16 / ξ by ring,
    show ∀ x : ℝ, 32 * x / (ξ / 2) ^ 2 = 128 * x / ξ ^ 2 from fun x => by ring,
    show η + ξ / 2 + ξ / 2 = η + ξ by ring] at h
  exact h hL hη hη₀ hc hδ hδ1 hpAP (by positivity) hA2 hpAPu
    ⟨q, hq₁, covered_of_dominated M L hc₀ hC hξ hq⟩ hL₀

end OrthoDFA

end
