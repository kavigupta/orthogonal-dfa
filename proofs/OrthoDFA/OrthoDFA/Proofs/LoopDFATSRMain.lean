import OrthoDFA.Proofs.TSRMainBridge

/-!
# The learner's DFA under typical-start replication, probes as on main: the proof

The claims are `learner_loop_dfa_dist_core` and `learner_loop_dfa_dist_tol` at `D` uniform, the
misled probes counted among each state's strings shorter than `L + |Q| + 3`, the coverage
`c = 2ξm/(5|Σ|^(|Q|-1))` and edges crossed at `|Σ| + |Q| + 2`, so
`ε = ξm/(10|Σ|^(|Q|-1)(|Σ|+|Q|+3)N)`, with `ξ/2` and `γ + ξ/2` in place of `ξ` and `γ`: the last
round's leaf misjudges at most `γ + ξ/2` (`tsrm_bridge`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

theorem learner_loop_dfa_tsrm_simple_holds : LearnerLoopDFATSRMainSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ m γ δ pAP ξ Q A k L₁ N ε εcov δ' ι s τ J S B₃ Wm W PB RB
    cap Kc n hL hη hη₀ hm hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hstart hk1 hk2 hk3 hk4
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hQ0 : (0 : ℝ) ≤ Q := Nat.cast_nonneg _
  have hm1 : m ≤ 1 := by
    obtain ⟨q, hq, -⟩ := hstart
    exact hq.trans measureReal_le_one
  have hG1 : (1 : ℝ) ≤ A ^ (Fintype.card σ - 1) := one_le_pow₀ (by linarith)
  have hc1 : 2 * ξ * m / (5 * A ^ (Fintype.card σ - 1)) ≤ 1 := by
    rw [div_le_one (by positivity)]; nlinarith
  have hpL : prefixLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3)) L
      = uniformStrings L := prefixLaw_uniformStrings (by omega)
  have hpS : prefixLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3))
      (L + Fintype.card σ + 3 + Fintype.card σ + 3)
      = uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3) :=
    prefixLaw_uniformStrings le_rfl
  have hwS : windowLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3))
      ((L + 1) / 2) (L + Fintype.card σ + 3)
      = windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3) := windowLaw_uniformStrings (by omega)
  have h := learner_loop_dfa_dist_core M O L
    (uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3)) η₀
    (2 * ξ * m / (5 * A ^ (Fintype.card σ - 1))) (γ + ξ / 2) δ pAP (ξ / 2) (A + Q + 2)
    (A ^ k)⁻¹ hL hη hη₀ (by positivity) hc1 hδ hδ1 hpAP (by positivity) hA2 (by linarith)
    (uniformStrings_long _) (fun u => uniformStrings_prefix_le (by omega) u)
    (by rw [hpS]; exact hpAPu) (tsrm_bridge M O hL L hm hξ hstart)
  rw [show ∀ x : ℝ, 2 * ξ * m / (5 * A ^ (Fintype.card σ - 1)) / (4 * (A + Q + 2 + 1) * x)
        = ξ * m / (10 * A ^ (Fintype.card σ - 1) * (A + Q + 3) * x) from
      fun x => by field_simp; ring,
    show ∀ x : ℝ, ξ / 2 / (8 * x) = ξ / (16 * x) from fun x => by ring,
    show ∀ x : ℝ, 32 * x / (ξ / 2) ^ 2 = 128 * x / ξ ^ 2 from fun x => by ring,
    show γ + ξ / 2 + ξ / 2 = γ + ξ by ring] at h
  obtain ⟨settings, h1, h2, h3, -, h4⟩ := h hk1 hk2 hk3 hk4
  rw [hpL, hpS, hwS] at h4
  exact ⟨settings, h1, h2, h3, h4⟩

set_option maxHeartbeats 4000000 in -- matching the claim's `L₀` against `loopMassInv`
theorem learner_loop_dfa_tsrm_holds : LearnerLoopDFATSRMain := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ m γ δ pAP ξ Q A k L₁ N ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hm hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hstart hL₀
  have : Nonempty σ := ⟨M.start⟩
  have hQ0 : (0 : ℝ) ≤ Q := Nat.cast_nonneg _
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hm1 : m ≤ 1 := by
    obtain ⟨q, hq, -⟩ := hstart
    exact hq.trans measureReal_le_one
  have hG1 : (1 : ℝ) ≤ A ^ (Fintype.card σ - 1) := one_le_pow₀ (by linarith)
  have hc0 : 0 < 2 * ξ * m / (5 * A ^ (Fintype.card σ - 1)) := by positivity
  have hc1 : 2 * ξ * m / (5 * A ^ (Fintype.card σ - 1)) ≤ 1 := by
    rw [div_le_one (by positivity)]; nlinarith
  have hMI : loopMassInv Q A (A + Q + 2) (Fintype.card σ) (Fintype.card α) δ
      (2 * ξ * m / (5 * A ^ (Fintype.card σ - 1))) (ξ / 2) pAP η₀ = E + E₂ + E₃ := by
    unfold loopMassInv loopΘ
    rw [show 4 * (A + Q + 2 + 1) / (2 * ξ * m / (5 * A ^ (Fintype.card σ - 1))) = aε by
        simp only [aε]; field_simp; ring,
      show (8 : ℝ) / (ξ / 2) = 16 / ξ by ring,
      show ∀ x : ℝ, 32 * x / (ξ / 2) ^ 2 = 128 * x / ξ ^ 2 from fun x => by ring]
    rfl
  have hEs1 : 1 ≤ E + E₂ + E₃ := by
    rw [← hMI]
    exact loopMassInv_ge_one _ _ hQ0 (by linarith) (by linarith) hδ hδ1 hc0 hc1 (by positivity)
      hpAP hp1 hη0 hη₀
  have hpow : ((L : ℝ) + Q + 4) ^ 6 * (E + E₂ + E₃) ≤ A ^ k :=
    dfa_pow hA rfl hQ0 (Nat.cast_nonneg L) hEs1
      (by exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)) rfl hL₀
  have hp₀ : (A ^ k)⁻¹ ≤ 1 / (((L : ℝ) + Q + 4) ^ 6 * loopMassInv Q A (A + Q + 2)
      (Fintype.card σ) (Fintype.card α) δ (2 * ξ * m / (5 * A ^ (Fintype.card σ - 1))) (ξ / 2)
      pAP η₀) := by
    have hL0 : (0 : ℝ) ≤ L := Nat.cast_nonneg L
    rw [hMI, one_div]
    exact inv_anti₀ (mul_pos (pow_pos (by linarith) 6) (by linarith)) hpow
  have hpL : prefixLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3)) L
      = uniformStrings L := prefixLaw_uniformStrings (by omega)
  have hpS : prefixLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3))
      (L + Fintype.card σ + 3 + Fintype.card σ + 3)
      = uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3) :=
    prefixLaw_uniformStrings le_rfl
  have hwS : windowLaw (uniformStrings (α := α) (L + Fintype.card σ + 3 + Fintype.card σ + 3))
      ((L + 1) / 2) (L + Fintype.card σ + 3)
      = windowStrings ((L + 1) / 2) (L + Fintype.card σ + 3) := windowLaw_uniformStrings (by omega)
  obtain ⟨settings, h1, h2, h3, -, h4⟩ := learner_loop_dfa_dist_tol M O L
    (uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3)) η₀
    (2 * ξ * m / (5 * A ^ (Fintype.card σ - 1))) (γ + ξ / 2) δ pAP (ξ / 2) (A + Q + 2) (A ^ k)⁻¹
    hL hη hη₀ hc0 hc1 hδ hδ1 hpAP (by positivity) hA2 (by linarith) (uniformStrings_long _)
    (fun u => uniformStrings_prefix_le (by show (L + 1) / 2 ≤ _; omega) u)
    (by rw [hpS]; exact hpAPu) (tsrm_bridge M O hL L hm hξ hstart) hp₀
  have hε : ∀ x : ℝ, 2 * ξ * m / (5 * A ^ (Fintype.card σ - 1)) / (4 * (A + Q + 2 + 1) * x)
      = ξ * m / (10 * A ^ (Fintype.card σ - 1) * (A + Q + 3) * x) :=
    fun x => by field_simp; ring
  rw [hε] at h2 h3
  rw [hpL, hpS, hwS, show ∀ x : ℝ, 32 * x / (ξ / 2) ^ 2 = 128 * x / ξ ^ 2 from fun x => by ring,
    show γ + ξ / 2 + ξ / 2 = γ + ξ by ring] at h4
  exact ⟨settings, h1, h2, h3, h4⟩

end OrthoDFA

end
