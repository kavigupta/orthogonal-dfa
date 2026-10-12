import OrthoDFA.Proofs.TallyEvent
import OrthoDFA.Proofs.TallySpurNoise
import OrthoDFA.Proofs.TallyKeep
import OrthoDFA.Proofs.TallyRace
import OrthoDFA.Proofs.TallySuccess
import OrthoDFA.Proofs.Yield

/-!
# The tally round's risk

`TallyRoundGiven` with its noise event's chance bounded by `tallyE_le` and `spurious_le`, the
records field at the coefficient `κs` and the read model's own `κ`, gives `TallyRound` at every
admissible choice of knobs.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

theorem le_prefixMax (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (u : FreeMonoid α) : D.real {x | prefixOf x k = u} ≤ prefixMax D k :=
  le_ciSup (f := fun u => D.real {x | prefixOf x k = u}) ⟨1, by
    rintro _ ⟨v, rfl⟩; exact measureReal_le_one⟩ u

theorem prefixMax_pos (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) :
    0 < prefixMax D k := by
  by_contra h
  push_neg at h
  have h0 : ∀ u : FreeMonoid α, D {x | prefixOf x k = u} = 0 := by
    intro u
    have h1 := (le_prefixMax D k u).trans h
    have h2 : D.real {x | prefixOf x k = u} = 0 := le_antisymm h1 measureReal_nonneg
    rwa [measureReal_eq_zero_iff] at h2
  have : (Set.univ : Set (FreeMonoid α)) = ⋃ u, {x | prefixOf x k = u} := by
    ext x; simp
  have hu := measure_iUnion_null (μ := D) h0
  rw [← this, measure_univ] at hu
  exact one_ne_zero hu

omit [Fintype σ] in
theorem leafOf_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided)) :
    ∀ (T : DTree α) (q : σ), ({ G with κ := κ', rare := h } : ReadModel α σ).leafOf T q
      = G.leafOf T q
  | .leaf, _ => rfl
  | .node m r a, q => by
    simp only [ReadModel.leafOf, ReadModel.side, ReadModel.at', leafOf_with G κ' h r q,
      leafOf_with G κ' h a q]
    congr

omit [Fintype σ] in
theorem grown_of_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    {T : DTree α} {f : ℕ} (hg : ({ G with κ := κ', rare := h } : ReadModel α σ).Grown T f) :
    G.Grown T f := by
  induction hg with
  | start => exact .start
  | real _ ht ht₀ hs ih =>
    refine .real ih ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 8 <;> exact decide_eq_decide.2 Iff.rfl
  | fake _ hp ht ht₀ hs ih =>
    refine .fake ih hp ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 9 <;> exact decide_eq_decide.2 Iff.rfl

omit [Fintype σ] in
theorem untrueAt_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α) :
    ({ G with κ := κ', rare := h } : ReadModel α σ).untrueAt rd k T edges
      = G.untrueAt rd k T edges := by
  simp only [ReadModel.untrueAt, ReadModel.TrueRec, leafOf_with]

/-- `TallyRoundRisk` over the proofs' view of the oracle. -/
theorem tally_round_model {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ) (read : FreeMonoid α → Ω → ARU)
    (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r)
    (P : PrefixDistribution α) (C : TallyCfg) (κs δ : ℝ) (hκ : 2 * G.κ ≤ κs) (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => P.D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)
          (TallyEndsWell G.M (fun q => ¬ G.Good q) P.D C P.firstAnchor P.L
            ((undecidedCount P G.M G.dist
              - 2 * Real.sqrt (P.prefixBound * undecidedCount P G.M G.dist * Real.log (1 / δ)))
              / (P.L + 1 - P.firstAnchor : ℕ)) (read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ roundRisk σ G.κ G.ε G.θ P.D P.L C κs + ENNReal.ofReal δ := by
  obtain ⟨D, _, L, st, hst, hlen, q, hq, hpreq⟩ := P
  simp only [roundRisk, ENNReal.iInf_add]
  refine le_iInf₂ fun kn hA => ?_
  have hold : ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => D)
      {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)
        (TallyEndsWell G.M (fun q => ¬ G.Good q) D C 0 L 0 (read · ω))
        tallyStart (List.ofFn xs)} ∂μ ≤ roundBound σ G.κ G.ε G.θ D L C kn := by
    have hp₀ := prefixMax_pos D C.k
    have hpmax := le_prefixMax D C.k
    have hrare : ∀ r, min (G.dist r .accept) (G.dist r .reject)
        ≤ max G.ε (κs * G.dist r .undecided) := fun r =>
      (G.rare r).trans (max_le_max le_rfl (mul_le_mul_of_nonneg_right
        (by linarith [hA.κ_pos]) (G.nonneg r _)))
    set G₂ : ReadModel α σ := { G with κ := κs, rare := hrare }
    have hθg0 : 0 ≤ kn.θg := le_trans (by linarith [hA.θ_nonneg]) hA.θg_ge
    have hθgpt0 : 0 ≤ kn.θgpt := le_trans (by have := hA.θ_nonneg; positivity) hA.θgpt_ge
    have hρ0 : 0 ≤ kn.ρ := le_trans (by have := hA.ε_nonneg; positivity) hA.records_slack
    have hTR := tally_round_of sub_round_holds fake_race_holds harvest_good_holds
      success_sound_holds (μ := μ) G₂ read D C kn.S kn.nEnd kn.nRec kn.hP kn.hS L C.budget kn.ρ
      kn.θg kn.θgs kn.θgpt kn.θpt' kn.θr kn.εd' kn.Xe kn.θs' kn.Xs kn.η kn.ν
      hlen hρ0 hA.m_pos hA.Lmax_ge hA.n₀_le_nEnd hA.n₀_le_hS hA.θpt_nonneg hA.θpt_le
      hA.εd_nonneg hA.εd_le hA.θpt'_nonneg hA.θpt'_le hA.θr_nonneg hA.θr_le_one hA.εd'_nonneg
      hA.εd'_le hA.θr_le hA.pt_level hA.success_level hA.success_reach hA.φpt_nEnd hA.L_pos
      hA.a_pos hA.a_le hθg0 hA.θe_ge hA.φe_nonneg hA.exc_gt hA.θgs_nonneg hA.θgs_le hA.start_test
      hθgpt0 hA.θgpt_le hA.pt_test hA.φpt_nonneg hA.φpt_le hA.φpt_test (by linarith [hA.θe_ge])
      hA.Xe_nonneg hA.exc_le hA.θs'_nonneg hA.n₀_le_Xs hA.start_lin hA.η_nonneg hA.ν_nonneg
      hA.race_side hA.budget_ge
    have hE := tallyE_le (ρ := kn.ρ) (θgs := kn.θgs) G₂ read hmeas hind hlaw hA.θ_nonneg D C kn.S L
      hlen hp₀ hpmax hA.l_nonneg hA.φe_nonneg hA.φpt_nonneg hA.θe_ge hA.θg_ge hA.θgpt_ge
    have hSp := spurious_le G read hmeas hind hlaw hA.κ_pos hA.ε_nonneg D C.k L kn.S hlen hp₀ hpmax
      hκ hA.records_slack
    have hcard : ((classSet (Fintype.card σ + kn.S) : Finset (DTree α)).card : ℝ)
        ≤ classCount (α := α) σ kn.S := by
      unfold classCount
      exact_mod_cast classSet_card (Fintype.card σ + kn.S)
    refine hTR.trans ?_
    unfold roundBound testLevels
    refine add_le_add (add_le_add (add_le_add (hE.trans ?_) le_rfl) le_rfl) le_rfl
    unfold recordsTail edgeTail startTail middlesTail edgeMapCount
    refine add_le_add (add_le_add (add_le_add ((measure_mono ?_).trans (hSp.trans ?_)) ?_) ?_) ?_
    · intro ω hω
      simp only [Set.mem_setOf_eq, not_forall] at hω ⊢
      obtain ⟨T, edges, ⟨f, hf, hg⟩, he, hlt⟩ := hω
      refine ⟨T, edges, ⟨f, hf, grown_of_with G κs hrare hg⟩, he, ?_⟩
      rwa [untrueAt_with] at hlt
    all_goals
      refine ENNReal.ofReal_le_ofReal ?_
    · gcongr ?_ * _ * _
    · gcongr ?_ * _ * _
    · gcongr ?_ * _
    · simp only [mul_one]
      gcongr ?_ * _ * _
  set n : ℝ := ((L + 1 - st : ℕ) : ℝ)
  have hn : (0 : ℝ) < n := by simp only [n]; exact_mod_cast anchorCount_pos hst
  set V := undecidedCount ⟨D, inferInstance, L, st, hst, hlen, q, hq, hpreq⟩ G.M G.dist
  set lg := Real.log (1 / δ)
  have hlg : 0 ≤ lg := Real.log_nonneg (by rw [le_div_iff₀ hδ0]; linarith)
  set Y : Ω → ℝ := fun ω => (D.prod (uniformAnchor st L)).real {r | r.2 ≤ r.1.toList.length ∧
      read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided}
  set y := (V - 2 * Real.sqrt (q * V * lg)) / n
  set Ny := {ω | Y ω < y}
  have hV : V = n * ∑ u ∈ anchorStrs st L, anchorWeight D st L u
      * μ.real {ω | read u ω = .undecided} := by
    simp only [V, undecidedCount]
    rw [tsum_eq_sum (s := anchorStrs st L) fun u hu => if_neg fun h => hu (mem_anchorStrs.2 h),
      Finset.mul_sum]
    refine Finset.sum_congr rfl fun u hu => ?_
    rw [if_pos (mem_anchorStrs.1 hu), hlaw, anchorWeight]
    have hn0 := hn.ne'
    simp only [n] at hn0 ⊢
    field_simp
  have hV0 : 0 ≤ V := by
    rw [hV]
    exact mul_nonneg hn.le (Finset.sum_nonneg fun u _ =>
      mul_nonneg (anchorWeight_nonneg D st L u) measureReal_nonneg)
  have hNy : μ Ny ≤ ENNReal.ofReal δ := by
    rcases hV0.lt_or_eq with hVp | hVz
    · rw [← ENNReal.ofReal_toReal (measure_ne_top μ Ny)]
      refine ENNReal.ofReal_le_ofReal ((measureReal_mono (fun ω (h : Y ω < y) => h.le)).trans
        ((anchor_reads read hmeas hind D hst q hpreq y
          (n * Real.sqrt (q * V * lg) / (q * V)) (by positivity)).trans_eq ?_))
      have hr : Real.sqrt (q * V * lg) ^ 2 = q * V * lg := Real.sq_sqrt (by positivity)
      have hm : ∑ u ∈ anchorStrs st L, anchorWeight D st L u * μ.real {ω | read u ω = .undecided}
          = V / n := by rw [hV]; field_simp
      have hexp : n * Real.sqrt (q * V * lg) / (q * V) * y
          - n * Real.sqrt (q * V * lg) / (q * V)
            * (1 - n * Real.sqrt (q * V * lg) / (q * V) * (q / n)) * (V / n) = -lg := by
        have hq0 : q ≠ 0 := hq.ne'
        have hV0' : V ≠ 0 := hVp.ne'
        have hn0 : n ≠ 0 := hn.ne'
        simp only [y]
        field_simp
        linear_combination (-1 : ℝ) * hr
      rw [hm, hexp, show lg = Real.log (1 / δ) from rfl, one_div, Real.log_inv, neg_neg,
        Real.exp_log hδ0]
    · have : Ny = ∅ := by
        refine Set.eq_empty_of_forall_notMem fun ω (h : Y ω < y) => ?_
        simp only [y, ← hVz, mul_zero, zero_mul, Real.sqrt_zero, sub_zero, zero_div] at h
        exact absurd h (not_lt.2 measureReal_nonneg)
      rw [this, measure_empty]
      exact zero_le
  set B := toMeasurable μ Ny
  have hpt : ∀ ω, (Measure.pi fun _ : Fin C.budget => D)
      {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)

          (TallyEndsWell G.M (fun q => ¬ G.Good q) D C st L y (read · ω)) tallyStart
          (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)
          (TallyEndsWell G.M (fun q => ¬ G.Good q) D C 0 L 0 (read · ω))
          tallyStart (List.ofFn xs)} + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · rw [Set.indicator_of_notMem hω, add_zero]
      have hωY : ¬ ω ∈ Ny := fun h => hω (subset_toMeasurable _ _ h)
      simp only [Ny, Set.mem_ofPred_eq, not_lt] at hωY
      refine measure_mono fun xs hxs h => hxs ?_
      have h' := runEnds_inv (fun s : TState α => RootOne s.tree)
        (fun s x hs => tallyStep_rootOne C _ hs x) tallyStart _ ⟨_, _, rfl⟩ h
      refine runEnds_mono (fun e s' hs => ?_) _ _ h'
      obtain ⟨hQ, hroot⟩ := hs
      refine EndsWell.respread (tallyReplay_spread _ _ _ D st L) ?_ hQ
      refine hωY.trans (measureReal_mono ?_ (measure_ne_top _ _))
      rintro ⟨w, e⟩ ⟨he, hu⟩
      exact tallyReplay_root _ hroot _ he hu
  calc _ ≤ ∫⁻ ω, ((Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)
          (TallyEndsWell G.M (fun q => ¬ G.Good q) D C 0 L 0 (read · ω))
          tallyStart (List.ofFn xs)} + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)
          (TallyEndsWell G.M (fun q => ¬ G.Good q) D C 0 L 0 (read · ω))
          tallyStart (List.ofFn xs)} ∂μ + μ B := by
        rw [lintegral_add_right _ (measurable_one.indicator (measurableSet_toMeasurable _ _)),
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ _ := by
        rw [measure_toMeasurable]
        exact add_le_add hold hNy

theorem tally_round_holds : TallyRoundRisk := by
  intro α _ _ σ _ Ω _ μ _ M O θ P C κs δ hκ hδ0 hδ1
  exact tally_round_model ⟨M, O.dist, θ, O.κ, O.ε, O.nonneg, O.rare⟩ O.read O.measurable O.indep
    O.law P C κs δ hκ hδ0 hδ1

end OrthoDFA
