import OrthoDFA.Claims.Loop
import OrthoDFA.Proofs.LoopDFARelative
import OrthoDFA.Proofs.ClaimLoopUniform

/-!
# The learner's costs, asymptotically: the proof

`learner_loop_dfa_dist_tol` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) loopCount` prefixes and `loopPool + 1` suffixes, and its probes by
`(|Q| + 1) loopBudget`, all at `ℓ = L + |Q| + 4`, once `D` holds at most
`1/(ℓ⁶ loopMassInv)` on any first `k` letters. The claim runs it at `c = ξ/(2C|Q|)`, `ξ/2` and
`1/β` in place of `|Σ|` where edges are crossed, admitted by `covered_of_self_nonStartZone` and
`covered_of_nonStartZone`; `ClaimLoopArith` bounds those shapes. `LoopClaimUniform` and
`LoopClaimSimple` follow (`ClaimLoopUniform`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

theorem loopClaim_holds : LoopClaim := by
  refine ⟨distP₀Tol, distPref, distSuff, distProbes, distHeldOut, distP₀Tol_isBigO,
    distPref_isBigO, distSuff_isBigO, distProbes_isBigO, distHeldOut_isBigO, distP₀Tol_pos, ?_⟩
  intro α _ _ _ σ _ Ω _ μ _ x
  obtain ⟨⟨Q, A, δ, R, ξ, pAP, s₀⟩, β⟩ := x
  intro M O L γ D _ k L₁ Ls hx hQ hA hL hηle hlong hpre hfl hpAPu hdom
  have hr : ({ toLoopDistInputs := ⟨⟨Q, A, δ, R, ξ, pAP, s₀⟩, β⟩, L } : LoopDistRun)
      ∈ loopDistRunRegion := hx
  obtain ⟨-, -, -, hAAe, -, -, -, -, -, -, -, hc0, hc1⟩ := dist_coords hr
  obtain ⟨⟨-, hA2, hδ, hδ2, hR, hξ, hξ1, hpAP, hpAP1, hs, hs1⟩, hβ, -⟩ := hx
  subst hQ hA
  have hη₀ : 1 / 2 - s₀ < 1 / 2 := by linarith
  have : Nonempty σ := ⟨M.start⟩
  set C : ℝ := R * (L + 1) / ((L + 1 - k : ℕ) : ℝ) with hCdef
  set c : ℝ := ξ / (2 * C * Fintype.card σ) with hcdef
  have hC : R ≤ C ∧ C ≤ 2 * R := visitC_bounds L (by linarith)
  have hC0 : 0 < C := by linarith
  obtain ⟨q, hself, hq⟩ := hdom
  have hcov := covered_of_self_nonStartZone (prefixLaw D L) M L hC0 rfl hξ1 hself
  rw [← Set.insert_eq_of_mem hself] at hq
  have hceil : ∃ q ∈ coveredStatesOn (prefixLaw D L) M L c,
      1 - (γ + ξ / 2) ≤ rerootAccuracyOn (prefixLaw D L) M L
        (coveredStatesOn (prefixLaw D L) M L c) q :=
    ⟨q, hcov, covered_of_nonStartZone (prefixLaw D L) M L hC0 rfl hξ hcov hq⟩
  have hfl' : ∀ i, k ≤ i → i ≤ L → ∀ y a, D.real {w | prefixOf w i = y}
      ≤ 1 / β * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a} := fun i hi hi' y a => by
    have h := hfl i hi hi' y a
    have e : 1 / β * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a}
        = D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a} / β := by ring
    rw [e, le_div_iff₀ hβ]
    linarith
  have hM1 : 1 ≤ loopMassInv (Fintype.card σ) (Fintype.card α) (1 / β) (Fintype.card σ)
      (Fintype.card α) δ c (ξ / 2) pAP (1 / 2 - s₀) :=
    loopMassInv_ge_one _ _ (Nat.cast_nonneg _) (Nat.cast_nonneg _)
      ((show (2 : ℝ) ≤ Fintype.card α by exact_mod_cast hA2).trans hAAe) hδ (by linarith) hc0 hc1
      (by positivity) hpAP hpAP1 (by linarith) hη₀
  have hp₀ : distP₀Tol ⟨⟨⟨Fintype.card σ, Fintype.card α, δ, R, ξ, pAP, s₀⟩, β⟩, L⟩
      ≤ 1 / ((L + (Fintype.card σ : ℝ) + 4) ^ 6 * loopMassInv (Fintype.card σ)
        (Fintype.card α) (1 / β) (Fintype.card σ) (Fintype.card α) δ c (ξ / 2) pAP
        (1 / 2 - s₀)) := by
    have e : distP₀Tol ⟨⟨⟨Fintype.card σ, Fintype.card α, δ, R, ξ, pAP, s₀⟩, β⟩, L⟩
        = 1 / ((L + (Fintype.card σ : ℝ) + 4) ^ 6 * max 1 (loopMassInv (Fintype.card σ)
          (Fintype.card α) (1 / β) (Fintype.card σ) (Fintype.card α) δ c (ξ / 2) pAP
          (1 / 2 - s₀))) := rfl
    rw [e, max_eq_right hM1]
  obtain ⟨settings, hw, -, -, hcnt, hprob⟩ := learner_loop_dfa_dist_tol M O L D (1 / 2 - s₀) c
    (γ + ξ / 2) δ pAP (ξ / 2) (1 / β) _ hL hηle hη₀ hc0 hδ (by linarith) hpAP (by positivity)
    hA2 hAAe hlong hfl' hpre hpAPu hceil hp₀
  refine ⟨settings, hw, hcnt, ?_⟩
  rw [show γ + ξ / 2 + ξ / 2 = γ + ξ by ring] at hprob
  exact hprob

theorem loopClaimUniform_holds : LoopClaimUniform := loopClaimUniform_of_loopClaim loopClaim_holds

theorem loopClaimSimple_holds : LoopClaimSimple := loopClaimSimple_of_claim loopClaimUniform_holds

end OrthoDFA

end
