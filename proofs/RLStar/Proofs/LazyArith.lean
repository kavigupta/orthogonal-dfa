import RLStar.Proofs.RandomTree
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-! Facts the lazy loop's DFA and arithmetic share: a grown tree's root reads the string itself,
and the prefix count is nonnegative. -/

noncomputable section

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The tree's root reads the string itself. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

theorem rootOne_splitAt {T : DTree α} (h : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := h
  rcases p with _ | ⟨_ | _, p⟩ <;> exact ⟨_, _, rfl⟩

theorem rootOne_of_classSet : ∀ (n : ℕ) (T : DTree α), T ∈ Random.classSet n → RootOne T
  | 0, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_singleton] at h
    exact ⟨_, _, h⟩
  | n + 1, T, h => by
    classical
    simp only [Random.classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T₀, hT₀, _, _, rfl⟩
    · exact rootOne_of_classSet n T h
    · exact rootOne_splitAt (rootOne_of_classSet n T₀ hT₀) _ _

lemma rb_nonneg {Q s pAP δ' ι τ cL : ℝ} (hQ : 0 ≤ Q) (hs0 : 0 < s) (hs1 : s ≤ 1 / 2)
    (hp : 0 < pAP) (hp1 : pAP ≤ 1) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hι0 : 0 < ι) (hτ0 : 0 < τ)
    (hτ1 : τ ≤ 1) (hcL : 0 ≤ cL) :
    0 ≤ 2 ^ 22 * (Q + 2) * Real.log (128 * ((50 * (max (Real.log (20 / τ)) cL + 3) / s ^ 2 + 20 / s
          + 28) / pAP + 120 * Real.log (3 / (δ' / 2)) / pAP + 1) / δ') / (s ^ 4 * ι ^ 2)
      + 2 ^ 27 * Real.log ((Q + 2 + 2)
        * ((50 * (max (Real.log (20 / τ)) cL + 3) / s ^ 2 + 20 / s + 28) / pAP
        + 120 * Real.log (3 / (δ' / 2)) / pAP)
        / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / τ ^ 2 := by
  have hm0 : 0 ≤ max (Real.log (20 / τ)) cL := le_max_of_le_right hcL
  have hN0 : 0 ≤ 50 * (max (Real.log (20 / τ)) cL + 3) / s ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) (by linarith)) (sq_nonneg s)
  have hN1 : 0 ≤ 20 / s := by positivity
  have h1 : 1 ≤ (50 * (max (Real.log (20 / τ)) cL + 3) / s ^ 2 + 20 / s + 28) / pAP := by
    rw [le_div_iff₀ hp]
    linarith
  have h2 : 0 ≤ 120 * Real.log (3 / (δ' / 2)) / pAP := by
    refine div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg ?_)) (by positivity)
    rw [le_div_iff₀ (by positivity)]
    linarith
  have hPB1 : 1 ≤ (50 * (max (Real.log (20 / τ)) cL + 3) / s ^ 2 + 20 / s + 28) / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP := by linarith
  generalize (50 * (max (Real.log (20 / τ)) cL + 3) / s ^ 2 + 20 / s + 28) / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP = PB at hPB1 ⊢
  have hl1 : 0 ≤ Real.log (128 * (PB + 1) / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
  have hD1 : δ' / 2 * (δ' / 4) * (pAP / 2) * τ ≤ 1 := by
    have : δ' / 2 * (δ' / 4) * (pAP / 2) ≤ 1 := by nlinarith
    nlinarith
  have hl2 : 0 ≤ Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ (by positivity), one_mul]
    nlinarith
  exact add_nonneg (div_nonneg (mul_nonneg (by positivity) hl1) (by positivity))
    (div_nonneg (mul_nonneg (by norm_num) hl2) (by positivity))

end OrthoDFA

end
