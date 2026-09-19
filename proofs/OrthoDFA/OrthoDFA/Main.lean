import OrthoDFA.Adaptive

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
  intro Ω _ μ _ S _ J _ O populations D Dsf _ _ Pre η₀ indecisionLimit εcov α δ ρ pAP
    hηle hη₀ hpop hflat hsupp hρ hpAPPositive hpAPBound hindLim hind1 hαpos hα hεcov
    hε1 hδ hcutlim hρcap hρsf
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ ≤ 0) measureReal_nonneg
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hfind : Real.exp (-2 * (poolCount η₀ populations εcov δ pAP : ℝ) * (pAP / 2) ^ 2)
      ≤ δ / 4 :=
    le_trans (solved_findability η₀ populations hδ hpAPPositive hcard1) (by linarith)
  have h := sound_and_terminating (runMeasure μ D Dsf)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations εcov δ α pAP ρ (collisionMass Dsf)} =>
      ret O.mq populations indecisionLimit α B.val ∩ FailAt O populations D εcov B.val)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations εcov δ α pAP ρ (collisionMass Dsf)} =>
      ret O.mq populations indecisionLimit α B.val) δ
    (validity_of_returned hflat O populations D Dsf hsupp indecisionLimit εcov α hsig hpop
      hηle hη₀ ρ (collisionMass Dsf) hρ le_rfl (tsum_nonneg (fun a => sq_nonneg _))
      hεcov δ hδ pAP hpAPPositive.le hpAPBound hfind)
    (loop_terminates hflat O populations D Dsf hsupp indecisionLimit εcov α ρ pAP
      δ hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1 hcutlim
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

end OrthoDFA
