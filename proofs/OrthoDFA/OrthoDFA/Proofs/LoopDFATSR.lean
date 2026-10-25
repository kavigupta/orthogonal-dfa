import OrthoDFA.Proofs.StartReplication
import OrthoDFA.Proofs.LoopDFA

/-!
# The learner's DFA under typical-start replication: the proof

The start `q` is in `coveredStates M L c` and re-roots the target with accuracy at least `1 − γ`
(`reroot_accuracy_of_tsr`), so the claims are `LearnerLoopDFASimple` and `LearnerLoopDFA` at
`(c, γ)`.
-/

noncomputable section

namespace OrthoDFA

theorem learner_loop_dfa_tsr_simple_holds : LearnerLoopDFATSRSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c γ δ pAP ξ Q A k L₁ N ε εcov δ' ι s τ J S B₃ Wm W PB RB
    cap Kc n hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu htsr hk1 hk2 hk3 hk4
  obtain ⟨q, hqc, hq⟩ := htsr
  exact learner_loop_dfa_simple_holds M O L η₀ c γ δ pAP ξ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu
    ⟨q, hqc, (sub_le_sub_left hq 1).trans (reroot_accuracy_of_tsr M L hc hqc)⟩ hk1 hk2 hk3 hk4

theorem learner_loop_dfa_tsr_holds : LearnerLoopDFATSR := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c γ δ pAP ξ Q A k L₁ N ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu htsr hL₀
  obtain ⟨q, hqc, hq⟩ := htsr
  exact learner_loop_dfa_holds M O L η₀ c γ δ pAP ξ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu
    ⟨q, hqc, (sub_le_sub_left hq 1).trans (reroot_accuracy_of_tsr M L hc hqc)⟩ hL₀

end OrthoDFA

end
