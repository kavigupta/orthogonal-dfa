import OrthoDFA.Proofs.Budget

/-!
# The clustering's failure, as a bound on its bad runs

`clustering_correct` bounds below the chance its runs are good; its proof bounds above the chance
they are bad, which is what a union over the loop's rounds needs.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem sound_and_terminating_bad {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] {T : Type*} [Countable T] (Fail Ret : T → Set α) (δ : ℝ)
    (hvalid : ν.real (⋃ t, Fail t) ≤ δ / 2) (hterm : ν.real {y | ∀ t, y ∉ Ret t} ≤ δ / 2)
    (Good : α → Prop) (hG : ∀ y, (∃ t, y ∈ Ret t) → (∀ t, y ∉ Fail t) → Good y) :
    ν.real {y | ¬ Good y} ≤ δ := by
  have hsub : {y | ¬ Good y} ⊆ (⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t} := by
    intro y hy
    by_contra hc
    simp only [Set.mem_union, Set.mem_iUnion, Set.mem_setOf_eq, not_or, not_exists,
      not_forall, not_not] at hc
    exact hy (hG y hc.2 hc.1)
  calc ν.real {y | ¬ Good y} ≤ ν.real ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ν.real (⋃ t, Fail t) + ν.real {y | ∀ t, y ∉ Ret t} := measureReal_union_le _ _
    _ ≤ δ := by linarith

/-- `ClusteringCorrect`, as a bound on the runs that are not good. -/
theorem clustering_correct_bad {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (rule : Clusterer S) (populations : Finset J) (uni : J)
    (D : J → Measure S)
    (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (Pre Suf : Set S) (η₀ indecisionLimit εcov α δ ρ pAP crossLimit k : ℝ)
    (hηle : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (huni : uni ∈ populations) (hflat : Flat Pre Suf)
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (hρ : ∀ j ∈ populations, collisionMass (D j) ≤ ρ) (hpAPPositive : 0 < pAP)
    (hpAPBound : pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L})
    (hindLim : 0 < indecisionLimit) (hind1 : indecisionLimit ≤ 1 / 2) (hαpos : 0 < α)
    (hα : α < 1 / 2) (hεcov : 0 < εcov) (hε1 : εcov ≤ 1) (hδ : 0 < δ)
    (_hbudget : (prefCount η₀ populations indecisionLimit εcov δ α pAP crossLimit : ℝ)
      ≤ budgetCap populations η₀ indecisionLimit εcov δ α pAP crossLimit k)
    (hρcap : ρ ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP crossLimit)
    (hρsf : collisionMass Dsf ≤ collisionCap η₀ populations indecisionLimit εcov δ α pAP
      crossLimit) :
    (runMeasure μ D Dsf).real
      {x | ¬ ((∃ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP
          crossLimit ρ},
            x ∈ retBy rule O.mq populations uni indecisionLimit α B.val) ∧
          ∀ B : {B : State // B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP
            crossLimit ρ},
            x ∈ retBy rule O.mq populations uni indecisionLimit α B.val →
            ∀ j ∈ populations, 1 - εcov
              ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                  (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
              ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                  (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
                ≤ 2 * indecisionLimit)} ≤ δ := by
  have hsig : O.η < 1 / 2 := lt_of_le_of_lt hηle hη₀
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  rw [O.apSet_eq] at hpAPBound
  by_cases hδ1 : δ ≤ 1
  case neg =>
    exact measureReal_le_one.trans (by linarith [not_le.1 hδ1])
  have hcard1 : (1 : ℝ) ≤ (populations.card : ℝ) := by
    exact_mod_cast Finset.card_pos.2 hpop
  have hfind : Real.exp (-2
      * (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ)
      * (pAP / 2) ^ 2) ≤ δ / 16 :=
    le_trans (solved_findability η₀ populations hδ hpAPPositive hcard1) (by linarith)
  have hρsf0 : (0 : ℝ) ≤ collisionMass Dsf := tsum_nonneg (fun a => sq_nonneg _)
  have hcoll : (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) ^ 2
      * collisionMass Dsf ≤ δ / 16 := by
    obtain ⟨-, -, -, -, -, h2⟩ := solved_tails η₀ populations hη₀ hεcov hindLim hδ hpAPPositive
      (by linarith) (le_trans (tsum_nonneg (fun a => sq_nonneg _))
        (hρ hpop.choose hpop.choose_spec)) hρsf0 hρcap hρsf
    have hnn : (0 : ℝ) ≤ (poolCount η₀ populations indecisionLimit εcov δ pAP crossLimit : ℝ) ^ 2
        * collisionMass Dsf := by positivity
    nlinarith [mul_le_mul_of_nonneg_right hcard1 hnn]
  have h := sound_and_terminating_bad (runMeasure μ D Dsf)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ} =>
      retBy rule O.mq populations uni indecisionLimit α B.val
        ∩ FailAt rule O populations D indecisionLimit εcov B.val)
    (fun B : {B : State //
        B ∈ stoppable η₀ populations indecisionLimit εcov δ α pAP crossLimit ρ} =>
      retBy rule O.mq populations uni indecisionLimit α B.val) δ
    (validity_of_returned hflat O populations D Dsf hsupp hsuppSf indecisionLimit εcov α hindLim
      hsig hpop hηle hη₀ ρ (collisionMass Dsf) hρ le_rfl
      hρsf0 hεcov δ hδ pAP hpAPPositive.le hpAPBound hfind hcoll)
    (le_trans (measureReal_mono (fun x hx B hB => hx B
        (retAll_subset_retBy rule O.mq huni indecisionLimit α B.val hB)) (measure_ne_top _ _))
      (loop_terminates hflat O populations D Dsf hsupp hsuppSf indecisionLimit εcov α ρ pAP
        δ hsig hpop hηle hη₀ hεcov hε1 hδ hδ1 hαpos hα hindLim hind1
        hpAPPositive
        hpAPBound hρ (le_trans (tsum_nonneg (fun a => sq_nonneg _))
          (hρ hpop.choose hpop.choose_spec)) hρcap hρsf))
  refine h _ fun x hex hall => ?_
  simp only [Set.mem_ofPred_eq, Set.mem_inter_iff, FailAt, not_and,
    not_not] at hex hall ⊢
  refine ⟨hex, fun B hB j hj => ?_⟩
  obtain ⟨hcov, hund⟩ := hall B hB j hj
  refine ⟨le_trans hcov (measureReal_mono (fun p hp => cutCorrect_insert_one O _ _ _
    (one_not_mem_clusterAt O populations B.val x) p _ hp) (measure_ne_top _ _)), hund⟩

end OrthoDFA
