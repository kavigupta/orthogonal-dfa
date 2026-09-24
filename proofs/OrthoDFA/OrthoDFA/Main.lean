import OrthoDFA.Adaptive
import OrthoDFA.Budget

/-!
# The theorem

`ClusteringGuarantee`, stated in `OrthoDFA.Model`, holds.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- From `clustering_correct`, with the schedule and collision cap it names as the witnesses,
and `prefCount_le_poly` for the cost. -/
theorem clustering_guarantee : ClusteringGuarantee := by
  refine ⟨524288, ?_⟩
  intro Ω _ μ _ S _ J _ O populations Pre η₀ indecisionLimit εcov α δ pAP
    hηle hη₀ hpop hflat hpAPPositive hindLim hind1 hαpos hα hεcov hε1 hδ hδ1
  classical
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  refine ⟨collisionCap η₀ populations indecisionLimit εcov δ α pAP, ?_, ?_⟩
  · simp only [collisionCap]
    positivity
  intro D Dsf hD hDsf hsupp hpAPBound ρ hρ hρcap hρsf
  haveI := hD
  haveI := hDsf
  refine ⟨stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf), ?_, ?_⟩
  · -- Every rung's count is the top one halved, and the top one is what `budgetCap` bounds.
    intro B hB
    have hpAP1 : pAP ≤ 1 := le_trans hpAPBound measureReal_le_one
    have hsched : B ∈ schedule η₀ populations indecisionLimit εcov δ α pAP :=
      Finset.mem_of_mem_filter _ hB
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 hsched
    refine le_trans ?_ (prefCount_le_poly populations η₀ indecisionLimit εcov δ α pAP
      hsig hη0 hpop hindLim hεcov hε1 hδ hδ1 hαpos (by linarith) hpAPPositive hpAP1)
    exact_mod_cast Nat.div_le_self _ _
  exact clustering_correct O populations D Dsf Pre η₀ indecisionLimit εcov α δ ρ pAP 524288
    hηle hη₀ hpop hflat hsupp hρ hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov hε1 hδ
    (prefCount_le_poly populations η₀ indecisionLimit εcov δ α pAP hsig hη0 hpop
      hindLim hεcov hε1 hδ hδ1 hαpos (by linarith) hpAPPositive
      (le_trans hpAPBound measureReal_le_one))
    hρcap hρsf

#print axioms clustering_guarantee

end OrthoDFA
