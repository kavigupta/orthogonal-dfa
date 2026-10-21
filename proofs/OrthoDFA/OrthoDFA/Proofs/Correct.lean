import OrthoDFA.Proofs.RawValid

/-!
# The clustering is correct

`raw_valid` (whatever the loop returns is good, except with chance `δ/2 + α`) and
`raw_terminates` (it returns, except with chance `δ/2`), together.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem clustering_correct : ClusteringCorrect := by
  intro Ω _ μ _ S _ J _ O populations tame uni H D Dsf _ _ Pre Suf η₀ indecisionLimit εcov
    α δ ρ pAP crossLimit k hηle hη₀ huni htame hflat hH hsupp hsuppSf hρ hpAPPositive hpAPBound
    hindLim hind1 hαpos hα hεcov hε1 hδ _hbudget hρcap hρsf
  classical
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact le_trans (by linarith [not_le.1 hδ1] : (1 : ℝ) - δ - α ≤ 0) measureReal_nonneg
  have hvalid := raw_valid (rule := nearestClusterer) (crossLimit := crossLimit) hflat O
    populations tame uni
    huni htame H D Dsf hsupp hH hsuppSf εcov α δ ρ pAP hηle hη₀ hindLim hind1 hαpos
    (by linarith) hεcov hδ hδ1 hpAPPositive hpAPBound hρ hρcap hρsf
  have hterm := raw_terminates hflat O populations uni (htame huni) H D Dsf hsupp
    hH hsuppSf εcov α δ ρ pAP crossLimit hsig hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1
    hpAPPositive hpAPBound (hρ uni huni) hρcap hρsf
  set st := stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit (allHarvest H).card ρ
    with hst
  set Bad := streams H uni ⁻¹' ⋃ B : {B : State // B ∈ st},
      (retBy nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α B.val
        ∩ {y | ¬ ∀ j ∈ tame, goodD (rule := nearestClusterer) O populations uni D εcov
          (rejFrac η₀ indecisionLimit εcov) indecisionLimit y B.val j}) with hBad
  set NoRet := {x : Draws Ω S J | ∀ B : {B : State // B ∈ st},
      streams H uni x ∉ retAll nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α
        B.val}
    with hNoRet
  set ν := runMeasure μ D Dsf with hν
  have hgood : (Bad ∪ NoRet)ᶜ ⊆ {x | (∃ B : {B : State // B ∈ st},
        streams H uni x ∈ retBy nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α
          B.val) ∧
      ∀ B : {B : State // B ∈ st},
        streams H uni x ∈ retBy nearestClusterer O.mq populations uni η₀ indecisionLimit εcov α
          B.val →
        ∀ j ∈ tame, 1 - gateLimit uni εcov j - 2 * rejFrac η₀ indecisionLimit εcov
            - (if j = uni then 0 else εcov / 1024)
          ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
              (familyBy nearestClusterer O.mq populations (streams H uni x) B.val) p (oracleNoise
                x.1)}
          ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
              (familyBy nearestClusterer O.mq populations (streams H uni x) B.val) p (oracleNoise
                x.1)}
            ≤ 2 * indecisionLimit} := by
    intro x hx
    simp only [Set.mem_compl_iff, Set.mem_union, not_or] at hx
    obtain ⟨hxB, hxN⟩ := hx
    refine ⟨?_, fun B hB j hj => ?_⟩
    · simp only [hNoRet, Set.mem_ofPred_eq, not_forall, not_not] at hxN
      obtain ⟨B, hB⟩ := hxN
      exact ⟨B, retAll_subset_retBy nearestClusterer O.mq η₀ indecisionLimit εcov α B.val hB⟩
    · by_contra hc
      apply hxB
      refine Set.mem_preimage.2 (Set.mem_iUnion.2 ⟨B, hB, fun hall => hc ?_⟩)
      exact hall j hj
  have h1 : (1 : ℝ) ≤ ν.real (Bad ∪ NoRet) + ν.real (Bad ∪ NoRet)ᶜ := by
    have := measureReal_union_le (μ := ν) (Bad ∪ NoRet) (Bad ∪ NoRet)ᶜ
    rwa [Set.union_compl_self, probReal_univ] at this
  have h2 := measureReal_union_le (μ := ν) Bad NoRet
  have h3 := measureReal_mono (μ := ν) hgood (measure_ne_top _ _)
  linarith

end OrthoDFA
