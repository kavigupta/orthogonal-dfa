import OrthoDFA.Adaptive
import OrthoDFA.Budget

/-!
# The theorem

`ClusteringCorrect`, stated in `OrthoDFA.Model` alongside the definitions it is built from,
holds.  Those two files are the whole claim; the rest of the development is lemmas.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- `validity_of_returned` (whatever the loop returns is good, whenever it is returned) and
`loop_terminates` (it returns), each except w.p. `δ/2`, glued by `sound_and_terminating`. -/
theorem clustering_correct : ClusteringCorrect := by
  intro Ω _ μ _ S _ J _ O populations D Dsf _ _ Pre η₀ indecisionLimit εcov α δ ρ pAP k
    hηle hη₀ hpop hflat hsupp hρ hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov
    hε1 hδ _hbudget hρcap hρsf
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ ≤ 0) measureReal_nonneg
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hfind : Real.exp (-2 * (poolCount η₀ populations indecisionLimit εcov δ pAP : ℝ)
      * (pAP / 2) ^ 2) ≤ δ / 4 :=
    le_trans (solved_findability η₀ populations hδ hpAPPositive hcard1) (by linarith)
  have h := sound_and_terminating (runMeasure μ D Dsf)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf)} =>
      ret O.mq populations indecisionLimit α B.val ∩ FailAt O populations D εcov B.val)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP ρ (collisionMass Dsf)} =>
      ret O.mq populations indecisionLimit α B.val) δ
    (validity_of_returned hflat O populations D Dsf hsupp indecisionLimit εcov α hindLim hsig hpop
      hηle hη₀ ρ (collisionMass Dsf) hρ le_rfl (tsum_nonneg (fun a => sq_nonneg _))
      hεcov δ hδ pAP hpAPPositive.le hpAPBound hfind)
    (loop_terminates hflat O populations D Dsf hsupp indecisionLimit εcov α ρ pAP
      δ hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1
      hpAPPositive
      hpAPBound hρ (le_trans (tsum_nonneg (fun a => sq_nonneg _))
        (hρ hpop.choose hpop.choose_spec)) hρcap hρsf)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_inter_iff, FailAt, not_and, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

#check @ClusteringCorrect

/-- The audit-sized statement, from the concrete one: the schedule, the collision cap and the
budget are supplied here rather than named in the claim.  `prefCount_le_poly` is what says the
budget supplied is within `budgetCap`. -/
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
