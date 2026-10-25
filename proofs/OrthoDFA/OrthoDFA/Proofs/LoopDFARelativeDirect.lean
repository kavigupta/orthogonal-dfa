import OrthoDFA.Proofs.LoopDFA
import OrthoDFA.Proofs.VisitedStart

/-!
# The learner's DFA from a start the window visits: the proof

`ceil_of_visited` turns the relative precondition into `LearnerLoopDFASimple`'s at `c`: every state
the start reaches is covered at `c`, and a run from the start never leaves them.
-/

noncomputable section

namespace OrthoDFA

theorem learner_loop_dfa_relative_direct_simple_holds : LearnerLoopDFARelativeDirectSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c₀ c R γ δ pAP ξ Q A k ρ L₁ N ε εcov δ' ι s τ J S B₃ Wm W
    PB RB cap Kc n hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hcc hstart hk1 hk2 hk3 hk4
  obtain ⟨q₀, hq₀, hacc⟩ := hstart
  exact learner_loop_dfa_simple_holds M O L η₀ c γ δ pAP ξ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu
    (ceil_of_visited M _ hc hcc hq₀ hacc) hk1 hk2 hk3 hk4

theorem learner_loop_dfa_relative_direct_holds : LearnerLoopDFARelativeDirect := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c₀ c R γ δ pAP ξ Q A k ρ L₁ N ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hcc hstart hL₀
  obtain ⟨q₀, hq₀, hacc⟩ := hstart
  exact learner_loop_dfa_holds M O L η₀ c γ δ pAP ξ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu
    (ceil_of_visited M _ hc hcc hq₀ hacc) hL₀

end OrthoDFA

end
