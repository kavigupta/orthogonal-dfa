import OrthoDFA.Claims.Loop
import OrthoDFA.Proofs.LoopDFASummed
import OrthoDFA.Proofs.ClaimLoopUniform
import OrthoDFA.Proofs.ClaimReadsArith

/-!
# The learner's costs, asymptotically: the proof

`learner_loop_dfa_dist_tol` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) loopCount` prefixes and `loopPool + 1` suffixes, its probes by `(|Q| + 1) loopBudget`,
and the strings the learner asks the oracle by `learnerReads` where the loop's harvested draws
walk at most `B` probes a round, all at `ℓ = L + |Q| + 4`, once `D` holds at most
`1/loopTolInv ℓ` on any first `k` letters; `Query.eval_congr` is the rest of the reads conjunct,
and `indecisiveLStar_eq` turns its error event into the learner's. The claim runs it at
`c = ξ/(2R)`, `ξ/4` and `γ + 3ξ/4`, edges crossed at `1/β + |Q|`, its last round's leaf from
`summed_bridge`; `ClaimLoopArith` and `ClaimReadsArith` bound those shapes. `LoopClaimUniform` and
`LoopClaimSimple` follow (`ClaimLoopUniform`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

theorem loopClaim_holds : LoopClaim := by
  refine ⟨distP₀Tol, distPref, distSuff, distProbes, distReads, distHeldOut, distP₀Tol_isBigO,
    distPref_isBigO, distSuff_isBigO, distProbes_isBigO, distReads_isBigO, distHeldOut_isBigO,
    distP₀Tol_pos, ?_⟩
  intro α _ _ _ σ _ Ω _ μ _ x
  obtain ⟨⟨Q, A, δ, R, ξ, pAP, s₀⟩, β⟩ := x
  intro M O L γ D _ k L₁ Ls hx hQ hA hL hηle hlong hpre hfl hpAPu hdom
  have hr : ({ toLoopDistInputs := ⟨⟨Q, A, δ, R, ξ, pAP, s₀⟩, β⟩, L } : LoopDistRun)
      ∈ loopDistRunRegion := hx
  obtain ⟨-, -, -, hAAe, -, -, -, -, -, hc0, hc1⟩ := dist_coords hr
  obtain ⟨⟨-, hA2, hδ, hδ2, hR, hξ, hξ1, hpAP, hpAP1, hs, hs1⟩, hβ, -⟩ := hx
  subst hQ hA
  have hη₀ : 1 / 2 - s₀ < 1 / 2 := by linarith
  have : Nonempty σ := ⟨M.start⟩
  obtain ⟨q, -, hq⟩ := hdom
  have hAe : (2 : ℝ) ≤ 1 / β :=
    (show (2 : ℝ) ≤ Fintype.card α by exact_mod_cast hA2).trans hAAe
  have hfl' : ∀ i, k ≤ i → i ≤ L → ∀ y a, D.real {w | prefixOf w i = y}
      ≤ 1 / β * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a} := fun i hi hi' y a => by
    have h := hfl i hi hi' y a
    have e : 1 / β * D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a}
        = D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a} / β := by ring
    rw [e, le_div_iff₀ hβ]
    linarith
  have hM1 : 1 ≤ loopTolInv (L + Fintype.card σ + 4) (Fintype.card σ) (Fintype.card α)
      (1 / β + Fintype.card σ) (Fintype.card σ) (Fintype.card α) δ (ξ / (2 * R)) (ξ / 4) pAP
      (1 / 2 - s₀) :=
    (loopTolInv_le _ _ (by have : (0 : ℝ) ≤ L := Nat.cast_nonneg _; linarith)
      (Nat.cast_nonneg _) (Nat.cast_nonneg _)
      (by have : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _; linarith) hδ (by linarith) hc0
      hc1 (by positivity) hpAP hpAP1 (by linarith) hη₀).1
  have hp₀ : distP₀Tol ⟨⟨⟨Fintype.card σ, Fintype.card α, δ, R, ξ, pAP, s₀⟩, β⟩, L⟩
      ≤ 1 / loopTolInv (L + Fintype.card σ + 4) (Fintype.card σ) (Fintype.card α)
        (1 / β + Fintype.card σ) (Fintype.card σ) (Fintype.card α) δ (ξ / (2 * R)) (ξ / 4) pAP
        (1 / 2 - s₀) := by
    have e : distP₀Tol ⟨⟨⟨Fintype.card σ, Fintype.card α, δ, R, ξ, pAP, s₀⟩, β⟩, L⟩
        = 1 / max 1 (loopTolInv (L + Fintype.card σ + 4) (Fintype.card σ) (Fintype.card α)
          (1 / β + Fintype.card σ) (Fintype.card σ) (Fintype.card α) δ (ξ / (2 * R)) (ξ / 4) pAP
          (1 / 2 - s₀)) := rfl
    rw [e, max_eq_right hM1]
  obtain ⟨settings, B, hw, -, -, -, -, -, hleaf, hcnt, hcalls, hprob⟩ :=
    learner_loop_dfa_dist_tol M O L D
    (1 / 2 - s₀)
    (ξ / (2 * R)) (γ + 3 * ξ / 4) δ pAP (ξ / 4) (1 / β + Fintype.card σ) _ hL hηle hη₀ hc0 hc1 hδ
    (by linarith) hpAP (by positivity) hA2
    (by have : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _; linarith) hlong hpre hpAPu
    (summed_bridge M O hL L D (by linarith) hξ hAe (hlong.mono fun x hx => le_trans (by omega) hx)
      hfl' ⟨q, hq⟩) hp₀
  refine ⟨settings, B, hw, hleaf, hcnt, fun p xs out hout hW => ⟨?_, fun o ho => ?_⟩, ?_⟩
  · rw [indecisiveLStar_eq] at hout
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 hout
    exact hcalls p xs rs hrs hW
  · obtain ⟨e1, e2⟩ := Query.eval_congr _ _ ho
    exact ⟨e1.trans hout, e2⟩
  rw [show γ + 3 * ξ / 4 + ξ / 4 = γ + ξ by ring] at hprob
  refine le_trans (measureReal_mono (fun q hq => ?_) (measure_ne_top _ _)) hprob
  simp only [Set.mem_setOf_eq] at hq ⊢
  rintro ⟨rounds, hr, ⟨hlen, o, ho, hend, herr⟩, hW⟩
  refine hq ⟨rounds, startLeaf O q.1.1 rounds q.2, ?_, ⟨hlen, o, ho, hend, ?_⟩, hW⟩
  · rw [indecisiveLStar_eq, hr]
    rfl
  · rw [Option.mem_def] at ho
    simp only [startLeaf, ho]
    exact herr

theorem loopClaimUniform_holds : LoopClaimUniform := loopClaimUniform_of_loopClaim loopClaim_holds

theorem loopClaimSimple_holds : LoopClaimSimple := loopClaimSimple_of_claim loopClaimUniform_holds

end OrthoDFA

end
