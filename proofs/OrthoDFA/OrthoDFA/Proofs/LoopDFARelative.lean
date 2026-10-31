import OrthoDFA.Proofs.LoopDFASummed

/-!
# The learner's DFA under the relative rule: the proof

The claims are `learner_loop_dfa_dist_core` and `learner_loop_dfa_dist_tol` at `D` uniform, the
coverage `ξ/(2R)` and edges crossed at `|Σ| + |Q|`, so `ε = m/(3(|Σ|+|Q|+1)N)` at `m = 3ξ/(8R)`,
with `ξ/4` and `γ + 3ξ/4` in place of `ξ` and `γ`: the last round's leaf misjudges at most
`γ + 3ξ/4` (`summed_bridge`). A start in its own zone gives `R ≥ 1/2`, so the coverage is at most
`1`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

theorem learner_loop_dfa_relative_simple_holds : LearnerLoopDFARelativeSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ R γ δ pAP ξ Q A k L₁ N m ε εcov δ' ι s τ J S B₃ Wm W PB RB
    cap Kc n hL hη hη₀ hR hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hdom hk1 hk2 hk3 hk4
  obtain ⟨q, hself, hq⟩ := hdom
  have hR2 := half_le_of_self_nonStartZone (uniformStrings L) M L hR hself
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hQ0 : (0 : ℝ) ≤ Q := Nat.cast_nonneg _
  have hc1 : ξ / (2 * R) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
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
    (uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3)) η₀ (ξ / (2 * R))
    (γ + 3 * ξ / 4) δ pAP (ξ / 4) (A + Q) (A ^ k)⁻¹ hL hη hη₀ (by positivity) hc1 hδ hδ1 hpAP
    (by positivity) hA2 (by linarith) (uniformStrings_long _)
    (fun u => uniformStrings_prefix_le (by omega) u) (by rw [hpS]; exact hpAPu)
    (summed_bridge M O hL L _ hR hξ hA ((uniformStrings_long _).mono fun x hx => by omega)
      (fun i _ hi y a => uniformStrings_floor (by omega) y a) ⟨q, by rw [hpL]; exact hq⟩)
  rw [show ∀ x : ℝ, ξ / (2 * R) / (4 * (A + Q + 1) * x) = m / (3 * (A + Q + 1) * x) from
      fun x => by simp only [m]; field_simp; ring,
    show ∀ x : ℝ, ξ / (2 * R) / (10 * x) = ξ / (20 * R * x) from fun x => by ring,
    show ∀ x : ℝ, ξ / 4 / (8 * x) = ξ / (32 * x) from fun x => by ring,
    show ∀ x : ℝ, 32 * x / (ξ / 4) ^ 2 = 512 * x / ξ ^ 2 from fun x => by ring,
    show γ + 3 * ξ / 4 + ξ / 4 = γ + ξ by ring] at h
  obtain ⟨settings, h1, h2, h3, -, h4⟩ := h hk1 hk2 hk3 hk4
  rw [hpL, hpS, hwS] at h4
  exact ⟨settings, h1, h2, h3, h4⟩

set_option maxHeartbeats 4000000 in -- matching the claim's `L₀` against `loopMassInv`
theorem learner_loop_dfa_relative_holds : LearnerLoopDFARelative := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ R γ δ pAP ξ Q A k L₁ N m ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hR hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hdom hL₀
  obtain ⟨q, hself, hq⟩ := hdom
  have hR2 := half_le_of_self_nonStartZone (uniformStrings L) M L hR hself
  have : Nonempty σ := ⟨M.start⟩
  have hQ0 : (0 : ℝ) ≤ Q := Nat.cast_nonneg _
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hc0 : 0 < ξ / (2 * R) := by positivity
  have hc1 : ξ / (2 * R) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
  have hMI : loopMassInv Q A (A + Q) (Fintype.card σ) (Fintype.card α) δ (ξ / (2 * R)) (ξ / 4)
      pAP η₀ = E + E₂ + E₃ := by
    unfold loopMassInv loopMassTerms loopΘ
    rw [show 4 * (A + Q + 1) / (ξ / (2 * R)) = aε by
        simp only [aε, m]; field_simp; ring,
      show (10 : ℝ) / (ξ / (2 * R)) = 20 * R / ξ by rw [div_div_eq_mul_div]; ring,
      show (8 : ℝ) / (ξ / 4) = 32 / ξ by ring,
      show ∀ x : ℝ, 32 * x / (ξ / 4) ^ 2 = 512 * x / ξ ^ 2 from fun x => by ring]
    rfl
  have hEs1 : 1 ≤ E + E₂ + E₃ := by
    rw [← hMI]
    exact loopMassInv_ge_one _ _ hQ0 (by linarith) (by linarith) hδ hδ1 hc0 hc1 (by positivity)
      hpAP hp1 hη0 hη₀
  have hpow : ((L : ℝ) + Q + 4) ^ 5 * (E + E₂ + E₃) ≤ A ^ k :=
    dfa_pow hA rfl hQ0 (Nat.cast_nonneg L) hEs1
      (by exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)) rfl hL₀
  have hp₀ : (A ^ k)⁻¹ ≤ 1 / loopTolInv ((L : ℝ) + Q + 4) Q A (A + Q) (Fintype.card σ)
      (Fintype.card α) δ (ξ / (2 * R)) (ξ / 4) pAP η₀ := by
    have hL0 : (0 : ℝ) ≤ L := Nat.cast_nonneg L
    obtain ⟨hT1, hTle⟩ := loopTolInv_le (ℓ := (L : ℝ) + Q + 4) (Q := Q) (A := A) (Ae := A + Q)
      (c := ξ / (2 * R)) (ξ := ξ / 4) (Fintype.card σ) (Fintype.card α) (by linarith) hQ0
      (by linarith) (by linarith) hδ hδ1 hc0 hc1 (by positivity) hpAP hp1 hη0 hη₀
    rw [hMI] at hTle
    rw [one_div]
    exact inv_anti₀ (by linarith) (hTle.trans hpow)
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
    (uniformStrings (L + Fintype.card σ + 3 + Fintype.card σ + 3)) η₀ (ξ / (2 * R))
    (γ + 3 * ξ / 4) δ pAP (ξ / 4) (A + Q) (A ^ k)⁻¹ hL hη hη₀ hc0 hc1 hδ hδ1 hpAP
    (by positivity) hA2 (by linarith) (uniformStrings_long _)
    (fun u => uniformStrings_prefix_le (by show (L + 1) / 2 ≤ _; omega) u)
    (by rw [hpS]; exact hpAPu)
    (summed_bridge M O hL L _ hR hξ hA ((uniformStrings_long _).mono fun x hx => by omega)
      (fun i _ hi y a => uniformStrings_floor (by omega) y a) ⟨q, by rw [hpL]; exact hq⟩) hp₀
  have hε : ∀ x : ℝ, ξ / (2 * R) / (4 * (A + Q + 1) * x) = m / (3 * (A + Q + 1) * x) :=
    fun x => by simp only [m]; field_simp; ring
  rw [hε] at h2 h3
  rw [hpL, hpS, hwS, show ∀ x : ℝ, 32 * x / (ξ / 4) ^ 2 = 512 * x / ξ ^ 2 from fun x => by ring,
    show γ + 3 * ξ / 4 + ξ / 4 = γ + ξ by ring] at h4
  exact ⟨settings, h1, h2, h3, h4⟩

end OrthoDFA

end
