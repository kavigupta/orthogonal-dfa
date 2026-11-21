import RLStar.Proofs.ClaimRLStarTol
import RLStar.Proofs.ClaimRLStarUniform

/-!
# The headline claims

`RLStarClaim` at the knobs of `ClaimRLStarArith`: the run is `lazy_within`, the totals' shapes
are `ClaimReadsArith`'s.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

theorem rlStarClaim_holds : RLStarClaim.{u, v, w} := by
  refine ⟨knTol, knPref, knSuff, knProbes, knTrie, knReads, ?_, ?_, ?_, ?_, ?_, ?_, knTol_pos,
    lazy_within⟩
  · refine isBigO_on (2 ^ 155) fun r hr => ?_
    have h := knP₀_inv_le hr
    have hp := (knP₀_bounds hr).1
    have hS := (tolShape_le hr).2.1
    have e : knTol r = knP₀ r := if_pos hr
    refine ⟨?_, ?_, ?_⟩
    · show 0 ≤ 1 / knTol r
      rw [e]; positivity
    · show 0 ≤ tolShape r
      linarith
    · show 1 / knTol r ≤ 2 ^ 155 * tolShape r
      rw [e]; exact h
  · refine isBigO_on (2 ^ 89) fun r hr => ⟨?_, (shape_pos hr).2.2.1, knPref_le hr⟩
    obtain ⟨hRB0, -⟩ := knRB_bounds hr
    unfold knPref
    positivity
  · refine isBigO_on (2 ^ 21) fun r hr => ⟨?_, (shape_pos hr).2.2.2.1, knSuff_le hr⟩
    obtain ⟨hPB, -⟩ := knPB_bounds hr
    unfold knSuff
    positivity
  · refine isBigO_on (2 ^ 163) fun r hr => ⟨?_, (shape_pos hr).2.2.2.2.2.1, knProbes_le hr⟩
    obtain ⟨hB₃, -⟩ := knB₃_bounds hr
    unfold knProbes
    positivity
  · refine isBigO_on (2 ^ 150) fun r hr => ⟨?_, nTrie_nonneg hr, knTrie_le hr⟩
    obtain ⟨hRB0, -⟩ := knRB_bounds hr
    obtain ⟨hθ, -⟩ := knθ_bounds hr
    unfold knTrie
    positivity
  · refine isBigO_on (2 ^ 171) fun r hr => ⟨?_, (shape_pos hr).2.2.2.2.2.2, knReads_le hr⟩
    obtain ⟨hRB0, -⟩ := knRB_bounds hr
    obtain ⟨hθ, -⟩ := knθ_bounds hr
    obtain ⟨hPB, -, hFV, -⟩ := knPB_bounds hr
    obtain ⟨hSC, -⟩ := knSC_bounds hr
    obtain ⟨hHR, -⟩ := knHR_bounds hr
    unfold knReads lazyReads knTrie
    positivity

theorem rlStarClaimUniform_holds : RLStarClaimUniform.{u, v, w} :=
  rlStarClaimUniform_of_rlStarClaim rlStarClaim_holds

theorem rlStarClaimUniformSimple_holds : RLStarClaimUniformSimple.{u, v, w} :=
  rlStarClaimUniformSimple_of_claim rlStarClaimUniform_holds

end OrthoDFA

end
