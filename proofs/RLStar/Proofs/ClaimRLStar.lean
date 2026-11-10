import RLStar.Claims.RLStar
import RLStar.Proofs.TrieTol
import RLStar.Proofs.ClaimRLStarUniform
import RLStar.Proofs.ClaimReadsArith

/-!
# The learner's costs, asymptotically: the proof

`trie_loop_tol` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) loopCount` prefixes and `loopPool + 1` suffixes, and its probes by
`(|Q| + 1) loopBudget`, all at `ℓ = L + |Q| + 4`, once `D` holds at most `1/(2³⁰ loopTolInv⁹ ℓ)` on
any first `k` letters; but for chance `δ` the run succeeds, asking the oracle at most
`learnerReads` strings with no held-out ones.
The claim runs it at main's rates at `R = 1` and `β = 1/|Σ|`: `c = ξ/2`, `ξ/4`, edges crossed at
`|Σ|`; `ClaimRLStarArith` and `ClaimReadsArith` bound those shapes.
`RLStarClaimUniform` and `RLStarClaimUniformSimple` follow (`ClaimRLStarUniform`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

theorem rlStarClaim_holds : RLStarClaim := by
  have hmapsR : Set.MapsTo (fun r => toDistRun (ratioOneRun r)) rlStarRunRegion
      rlStarDistRunRegion := fun r hr => toDist_mem (ratioOne_mem hr)
  refine ⟨fun r => distP₀Tol (toDistRun (ratioOneRun r)) ^ 9 / 2 ^ 30,
    fun r => distPref (toDistRun (ratioOneRun r)), fun r => distSuff (toDistRun (ratioOneRun r)),
    fun r => distProbes (toDistRun (ratioOneRun r)),
    fun r => distLoopReads (toDistRun (ratioOneRun r)), ?_, ?_, ?_, ?_, ?_,
    fun r => div_pos (pow_pos (distP₀Tol_pos _) 9) (by norm_num), ?_⟩
  · refine ((((isBigO_comp_on distP₀Tol_isBigO hmapsR).pow 9).const_mul_left (2 ^ 30)).congr'
      (eventually_principal.2 fun r _ => ?_) (eventually_principal.2 fun r hr => ?_))
    · simp only [Function.comp_apply]
      rw [one_div_pow, mul_one_div, one_div_div]
    · simp only [Function.comp_apply]
      rw [p₀Shape_one hr]
  · refine (isBigO_comp_on distPref_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, one_div_one_div]
    obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs
    rw [e, e']
    simp only [mul_one, one_pow]
  · refine (isBigO_comp_on distSuff_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist]
    rw [(uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs).2]
    simp only [mul_one]
  · refine (isBigO_comp_on distProbes_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, one_div_one_div]
    ring
  · refine (isBigO_comp_on distLoopReads_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, toDistRun, ratioOneRun, ratioOne, toDist, shCluster, shSearch,
      shHarvest, shPref, shSuff, shProbe, shFam, shRho, shLg, one_div_one_div]
    obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs
    rw [e, e']
    ring
  -- the run
  intro α _ _ _ σ _ Ω _ μ _ x
  obtain ⟨Q, A, δ, ξ, pAP, s₀⟩ := x
  intro M O L k D _ L₁ Ls hx hQ hA hLang hη hlong hkL hpre hpAPu
  subst hQ hA
  obtain ⟨hQ1, hA2, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  have hA' : (2 : ℝ) ≤ Fintype.card α := by exact_mod_cast hA2
  have hη₀ : 1 / 2 - s₀ < 1 / 2 := by linarith
  have hc0 : 0 < ξ / (2 * 1) := by positivity
  have hc1 : ξ / (2 * 1) ≤ 1 := by linarith
  set T := loopTolInv ((L : ℝ) + Fintype.card σ + 4) (Fintype.card σ)
      (Fintype.card α) (1 / (1 / (Fintype.card α : ℝ))) (Fintype.card σ) (Fintype.card α) δ
      (ξ / (2 * 1)) (ξ / 4) pAP (1 / 2 - s₀) with hTd
  have hT1 : 1 ≤ T := (loopTolInv_le _ _
    (by have : (0 : ℝ) ≤ L := Nat.cast_nonneg _; linarith) (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    (by rw [one_div_one_div]; exact hA') hδ (by linarith) hc0 hc1 (by positivity) hp hp1
    (by linarith) hη₀).1
  have hTm : max 1 T = T := max_eq_right hT1
  -- `D`'s first `k` letters weigh at most `1/Ak`
  have hpre' : ∀ u, D.real {w | prefixOf w k = u} ≤ (2 ^ 30 * T ^ 9)⁻¹ := by
    intro u
    refine (hpre u).trans (le_of_eq ?_)
    change (1 / max 1 T) ^ 9 / 2 ^ 30 = _
    rw [hTm, one_div_pow, div_div, one_div, mul_comm]
  obtain ⟨settings, hwk, hprobe, hprob⟩ := trie_loop_tol (Ω := Ω) (μ := μ) M O L k k
    (1 / 2 - s₀) (ξ / (2 * 1)) δ pAP (ξ / 4) ξ (1 / (1 / (Fintype.card α : ℝ)))
    (2 ^ 30 * T ^ 9) D hLang hη hη₀ hc0 hc1 hδ (by linarith) hp (by positivity) hA2
    (le_of_eq (one_div_one_div _).symm) hkL hlong hpre' hpAPu le_rfl
    (by linarith)
  exact ⟨settings, hwk, hprobe, hprob⟩

theorem rlStarClaimUniform_holds : RLStarClaimUniform :=
  rlStarClaimUniform_of_rlStarClaim rlStarClaim_holds

theorem rlStarClaimUniformSimple_holds : RLStarClaimUniformSimple :=
  rlStarClaimUniformSimple_of_claim rlStarClaimUniform_holds

end OrthoDFA

end
