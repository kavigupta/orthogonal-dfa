import OrthoDFA.Proofs.LoopTermination
import OrthoDFA.Proofs.Level4Correct
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.BandWide
import OrthoDFA.Proofs.LoopIndep
import OrthoDFA.Proofs.LoopLaw
import OrthoDFA.Proofs.Level4Short
import Mathlib.Data.List.GetD

/-!
# The loop's theorem

Each round meets the clustering's claim but for chance `δc` and, given that, level 4's but for
chance `2δ`; on the rounds' draws these are the one-round claims, at the populations and the
family that round was handed (`loop_cluster_round_le`, `loop_level4_round_le`). A round's harvest
is at least `⌈1/cap⌉` strings long and has no repeats unless two of its probes share their
length-`k` prefix, so its collision mass is within the clustering's cap but for chance
`T² prefixMax` (`loop_harvest_spread_le`). Where every one of the first `|Q| + 1` rounds meets all
three, the loop stops in agreement (`loop_agrees_of_specs`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## The one-round claims, uniform in what the earlier rounds handed on -/

/-- One schedule and collision cap serve every set of the loop's populations, and every state of
the schedule reads a family of at most `k - 1` members at a band whose far side `crossLimit`
bounds, whatever the read's two rates. -/
theorem clustering_schedule_exists {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (K : ℕ) (Pre Suf : Set (FreeMonoid α))
    (η₀ indecisionLimit gateLevel crossLimit εcov δc slack pAP : ℝ)
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hflat : Flat Pre Suf) (hpAP : 0 < pAP)
    (hι : 0 < indecisionLimit) (hι1 : indecisionLimit ≤ 1 / 2) (hg : 0 < gateLevel)
    (hg1 : gateLevel < 1 / 2) (hcross : 0 < crossLimit) (hε : 0 < εcov) (hε1 : εcov ≤ 1)
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hslack : 0 < slack) :
    ∃ cap : ℝ, 0 < cap ∧ ∃ schedule : Finset (Option (Fin K)) → Finset State,
      (∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ,
        0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
      ∧ ∀ pops : Finset (Option (Fin K)), none ∈ pops →
        ∀ (D : Option (Fin K) → Measure (FreeMonoid α)) (Dsf : Measure (FreeMonoid α)),
          (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
          (∀ j ∈ pops, D j Preᶜ = 0) → Dsf Sufᶜ = 0 →
          pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
          (∀ j ∈ pops, collisionMass (D j) ≤ cap) → collisionMass Dsf ≤ cap →
          (runMeasure μ D Dsf).real
            {x | ¬ ((∃ B ∈ schedule pops, x ∈ ret O.mq pops none indecisionLimit gateLevel B)
              ∧ ∀ B ∈ schedule pops, x ∈ ret O.mq pops none indecisionLimit gateLevel B →
                ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
                    ≤ εcov + slack
                  ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
                    ∂(D j) ≤ 2 * indecisionLimit + slack)} ≤ δc := by
  classical
  obtain ⟨c, hc⟩ : ∃ c : ℝ, c = Real.exp (-2 / crossLimit) := ⟨_, rfl⟩
  have hc0 : 0 < c := hc ▸ Real.exp_pos _
  choose capOf hcapOf using fun (pops : Finset (Option (Fin K))) (h : none ∈ pops) =>
    clustering_quality_bad O pops none Pre Suf η₀ indecisionLimit εcov gateLevel δc pAP c slack
      hη hη₀ h hflat hpAP hι hι1 hg hg1 hε hε1 hδc hδc1 hc0 hslack
  set capAt : Finset (Option (Fin K)) → ℝ := fun pops =>
    if h : none ∈ pops then capOf pops h else 1 with hcapAt
  have hcapAt0 : ∀ pops, 0 < capAt pops := by
    intro pops
    simp only [hcapAt]
    split_ifs with h
    · exact (hcapOf pops h).1
    · norm_num
  obtain ⟨cap, hcapd⟩ : ∃ cap : ℝ,
      cap = (Finset.univ : Finset (Finset (Option (Fin K)))).inf' ⟨∅, Finset.mem_univ _⟩ capAt :=
    ⟨_, rfl⟩
  have hcap0 : 0 < cap := by
    rw [hcapd]; exact (Finset.lt_inf'_iff _).2 fun pops _ => hcapAt0 pops
  have hcap_le : ∀ pops (h : none ∈ pops), cap ≤ capOf pops h := by
    intro pops h
    have := Finset.inf'_le capAt (Finset.mem_univ pops)
    simp only [hcapAt, dif_pos h] at this
    rw [hcapd]; exact this
  refine ⟨cap, hcap0, fun pops =>
    stoppable η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP c cap, ?_, ?_⟩
  · intro pops B hB
    obtain ⟨hlohi, -, hband⟩ := band_of_mem_schedule (le_trans (eta_nonneg O) hη) hη₀
      (Finset.mem_of_mem_filter _ hB)
    refine ⟨hlohi, fun N hN p r ε₂ κ hp0 hp1 hr0 hr1 _ =>
      bandPasses_of_wide hp0 hp1 hr0 hr1 hlohi ?_⟩
    have hlog : Real.log (1 / c) = 2 / crossLimit := by
      rw [hc, one_div, ← Real.exp_neg, Real.log_exp]; ring
    rw [hlog, mul_div_assoc', div_le_iff₀ hcross] at hband
    have hNk : (N : ℝ) + 1 ≤ B.k := by exact_mod_cast hN
    have hlh : (B.lo : ℝ) ≤ B.hi := by exact_mod_cast hlohi
    have h2 : ((B.hi : ℝ) - B.lo) ^ 2 ≤ ((B.hi : ℝ) + 1 - B.lo) ^ 2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h2 hcross.le]
  · intro pops hpops D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolD hcolSf
    obtain ⟨-, hbad⟩ := (hcapOf pops hpops).2 D Dsf hD hDsf hsupp hsuppSf hpAPDsf cap hcolD
      (hcap_le pops hpops) (hcolSf.trans (hcap_le pops hpops))
    refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hbad
    intro x hx hgood
    apply hx
    obtain ⟨⟨B, hret⟩, hall⟩ := hgood
    refine ⟨⟨B.val, B.property, hret⟩, fun B' hB' hret' => ?_⟩
    exact ⟨(hall ⟨B', hB'⟩ hret' none hpops).1, fun j hj => (hall ⟨B', hB'⟩ hret' j hj).2⟩

/-! ## The one-round claims on the loop's draws -/

section Rounds

variable {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

/-- The harvests of the first `r` rounds are a measurable function of the rounds' draws. -/
theorem measurableSet_harvestsOf (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (r : ℕ) (H : List (List (FreeMonoid α))) :
    MeasurableSet {draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget |
      harvestsOf (loopRounds settings O draws r) = H} := by
  sorry

lemma lt_of_some_mem_loopPopulations {n : ℕ} {i : Fin K} (h : some i ∈ loopPopulations K n) :
    i.val < n := by
  simp only [loopPopulations, Finset.mem_insert, reduceCtorEq, false_or, Finset.mem_image,
    Finset.mem_filter, Finset.mem_univ, true_and, Option.some.injEq] at h
  obtain ⟨a, ha, rfl⟩ := h
  exact ha

lemma pickFrom_mem_or_one (zs : List (FreeMonoid α)) (u : unitInterval) :
    pickFrom zs u ∈ zs ∨ pickFrom zs u = 1 := by
  unfold pickFrom
  by_cases h : ⌊(u : ℝ) * zs.length⌋₊ < zs.length
  · left
    rw [List.getD_eq_getElem (hn := h)]
    exact List.getElem_mem h
  · right
    exact List.getD_eq_default _ _ (not_lt.1 h)

lemma harvestMeasure_null {zs : List (FreeMonoid α)} {A : Set (FreeMonoid α)}
    (hzs : ∀ z ∈ zs, z ∉ A) (h1 : (1 : FreeMonoid α) ∉ A) : harvestMeasure zs A = 0 := by
  rw [← map_pickFrom, Measure.map_apply (measurable_pickFrom zs) (measurableSet_of_countable A)]
  convert measure_empty (μ := (volume : Measure unitInterval))
  refine Set.eq_empty_of_forall_notMem fun u hu => ?_
  rcases pickFrom_mem_or_one zs u with h | h
  · exact hzs _ h hu
  · rw [Set.mem_preimage, h] at hu
    exact h1 hu

omit [IsProbabilityMeasure μ] in
lemma loop_long_null [IsProbabilityMeasure μ] (P : PrefixDistribution α)
    (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf] (T : ℕ) :
    loopMeasure μ P.D Dsf K T {draws | ∃ i j, P.L < ((draws i).2.2 j).toList.length} = 0 := by
  have : IsProbabilityMeasure (clusterLaw μ P.D Dsf K) := by unfold clusterLaw; infer_instance
  have hset : {draws : Fin K → LoopRoundDraws Ω α K T | ∃ i j,
      P.L < ((draws i).2.2 j).toList.length}
      = ⋃ i, ⋃ j, (fun draws : Fin K → LoopRoundDraws Ω α K T => (draws i).2.2 j) ⁻¹'
        {x | P.L < x.toList.length} := by
    ext draws; simp
  rw [hset]
  refine measure_iUnion_null fun i => measure_iUnion_null fun j => ?_
  have hf : MeasurePreserving (fun draws : Fin K → LoopRoundDraws Ω α K T => (draws i).2.2 j)
      (Measure.pi fun _ : Fin K =>
        (clusterLaw μ P.D Dsf K).prod (μ.prod (Measure.pi fun _ : Fin T => P.D))) P.D :=
    (measurePreserving_eval (fun _ : Fin T => P.D) j).comp
      ((measurePreserving_snd.comp measurePreserving_snd).comp
        (measurePreserving_eval (fun _ : Fin K =>
          (clusterLaw μ P.D Dsf K).prod (μ.prod (Measure.pi fun _ : Fin T => P.D))) i))
  unfold loopMeasure
  rw [hf.measure_preimage (measurableSet_of_countable _).nullMeasurableSet]
  refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
  simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
  exact hx

/-- Round `r` meets the clustering's claim but for chance `δc`, unless an earlier round harvested
a list too concentrated for it. Given the earlier rounds, the round's run is drawn as
`runMeasure` draws it at their harvests, whose strings are shorter than `Ls` unless a probe was
longer than `L`. -/
theorem loop_cluster_round_le (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (settings : LoopSettings K) (S Ls : ℕ) (εcov δc slack cap : ℝ) (hδc : 0 ≤ δc)
    (hmax : settings.level4.maxLeaves = Fintype.card σ + S + 3)
    (hLs : P.L + Fintype.card σ + S + 3 ≤ Ls)
    (hcolD : collisionMass P.D ≤ cap)
    (hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (D : Option (Fin K) → Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (D j)) →
        (∀ j ∈ pops, D j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, collisionMass (D j) ≤ cap) →
        (runMeasure μ D Dsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops,
                x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops,
              x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
                  ≤ εcov + slack
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
                  ∂(D j) ≤ 2 * settings.indecisionLimit + slack)} ≤ δc)
    (r : ℕ) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {draws | ∃ hr : r < (loopRounds settings O draws K).length,
        (∀ o ∈ (loopRounds settings O draws K).take r, ∀ zs, o.result.ending = .harvest zs →
          collisionMass (harvestMeasure zs) ≤ cap)
        ∧ ¬ ClusterSpec settings O P.D εcov slack
          (harvestsOf ((loopRounds settings O draws K).take r))
          ((loopRounds settings O draws K)[r]'hr).run} ≤ δc := by
  classical
  have : IsProbabilityMeasure (clusterLaw μ P.D Dsf K) := by unfold clusterLaw; infer_instance
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K settings.level4.probeBudget) := by
    unfold loopMeasure; infer_instance
  set T := settings.level4.probeBudget with hT
  by_cases hrK : r < K
  swap
  · have hempty : {draws : Fin K → LoopRoundDraws Ω α K T |
        ∃ hr : r < (loopRounds settings O draws K).length,
          (∀ o ∈ (loopRounds settings O draws K).take r, ∀ zs, o.result.ending = .harvest zs →
            collisionMass (harvestMeasure zs) ≤ cap)
          ∧ ¬ ClusterSpec settings O P.D εcov slack
            (harvestsOf ((loopRounds settings O draws K).take r))
            ((loopRounds settings O draws K)[r]'hr).run} = ∅ := by
      refine Set.eq_empty_of_forall_notMem fun draws hd => ?_
      obtain ⟨hr, -⟩ := hd
      have := length_loopRounds_le settings O draws K
      omega
    rw [hempty, measureReal_empty]
    exact hδc
  set rK : Fin K := ⟨r, hrK⟩
  set ν : Measure (LoopRoundDraws Ω α K T) :=
    (clusterLaw μ P.D Dsf K).prod (μ.prod (Measure.pi fun _ : Fin T => P.D))
  have hLs' : P.L < Ls := by omega
  -- the round's bad draws at given harvests
  set Bad : List (List (FreeMonoid α)) → Set (LoopRoundDraws Ω α K T) := fun H =>
    {d | H.length = r ∧ (∀ zs ∈ H, collisionMass (harvestMeasure zs) ≤ cap)
      ∧ (∀ zs ∈ H, ∀ z ∈ zs, z.toList.length < Ls)
      ∧ ¬ ClusterSpec settings O P.D εcov slack H (clusterRun H d.1)} with hBadd
  have hBad : ∀ H, ν (Bad H) ≤ ENNReal.ofReal δc := by
    intro H
    by_cases hH : (∀ zs ∈ H, collisionMass (harvestMeasure zs) ≤ cap)
      ∧ (∀ zs ∈ H, ∀ z ∈ zs, z.toList.length < Ls)
    swap
    · have : Bad H = ∅ := Set.eq_empty_of_forall_notMem fun d hd => hH ⟨hd.2.1, hd.2.2.1⟩
      rw [this, measure_empty]
      exact zero_le
    obtain ⟨hcol, hshort⟩ := hH
    set Tset := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      ¬ ClusterSpec settings O P.D εcov slack H x}
    have hcr := measurePreserving_clusterRun (μ := μ) (K := K) P.D Dsf H
    have hgetD : ∀ i : Fin K, some i ∈ loopPopulations K H.length → H.getD i [] ∈ H := by
      intro i hi
      have := lt_of_some_mem_loopPopulations hi
      rw [List.getD_eq_getElem (hn := this)]
      exact List.getElem_mem _
    have hsupp : ∀ j ∈ loopPopulations K H.length,
        loopDistributions P.D H j {p | Ls ≤ p.toList.length} = 0 := by
      intro j hj
      cases j with
      | none =>
        refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
        simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
        omega
      | some i =>
        refine harvestMeasure_null (fun z hz hzA => ?_) (by simp; omega)
        have := hshort _ (hgetD i hj) z hz
        simp only [Set.mem_ofPred_eq] at hzA
        omega
    have hcolp : ∀ j ∈ loopPopulations K H.length,
        collisionMass (loopDistributions P.D H j) ≤ cap := by
      intro j hj
      cases j with
      | none => exact hcolD
      | some i => exact hcol _ (hgetD i hj)
    have hc := hclust (loopPopulations K H.length) (Finset.mem_insert_self _ _)
      (loopDistributions P.D H) (fun j => inferInstance) hsupp hcolp
    calc ν (Bad H) ≤ ν (Prod.fst ⁻¹' (clusterRun H ⁻¹' Tset)) :=
          measure_mono fun d hd => hd.2.2.2
      _ ≤ (ν.map Prod.fst) (clusterRun H ⁻¹' Tset) :=
          Measure.le_map_apply measurable_fst.aemeasurable _
      _ = clusterLaw μ P.D Dsf K (clusterRun H ⁻¹' Tset) := by
          rw [Measure.map_fst_prod, measure_univ, one_smul]
      _ ≤ ((clusterLaw μ P.D Dsf K).map (clusterRun H)) Tset :=
          Measure.le_map_apply hcr.measurable.aemeasurable _
      _ = runMeasure μ (loopDistributions P.D H) Dsf Tset := by rw [hcr.map_eq]
      _ ≤ ENNReal.ofReal δc := by
          rw [← ofReal_measureReal]
          exact ENNReal.ofReal_le_ofReal hc
  have hσ : MeasurePreserving (fun d : LoopRoundDraws Ω α K T => ((() : Unit), d)) ν
      ((Measure.dirac ()).prod ν) :=
    ⟨measurable_const.prodMk measurable_id, by rw [Measure.dirac_prod]⟩
  have hmain := round_stage_le ν (Measure.dirac ()) ν (fun d => ((() : Unit), d)) hσ Prod.snd
    measurable_snd (fun w => Prod.ext rfl rfl) rK
    (fun draws => harvestsOf (loopRounds settings O draws r))
    (fun x x' hxx' _ => by
      rw [loopRounds_congr settings O x x' r fun i hi => hxx' i hi])
    (measurableSet_harvestsOf settings O r) Bad (ENNReal.ofReal δc) hBad
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K T |
        ∃ hr : r < (loopRounds settings O draws K).length,
          (∀ o ∈ (loopRounds settings O draws K).take r, ∀ zs, o.result.ending = .harvest zs →
            collisionMass (harvestMeasure zs) ≤ cap)
          ∧ ¬ ClusterSpec settings O P.D εcov slack
            (harvestsOf ((loopRounds settings O draws K).take r))
            ((loopRounds settings O draws K)[r]'hr).run}
      ⊆ {draws | ((() : Unit), draws rK).2 ∈ Bad (harvestsOf (loopRounds settings O draws r))}
        ∪ {draws | ∃ i j, P.L < ((draws i).2.2 j).toList.length} := by
    rintro draws ⟨hr, hcolE, hspec⟩
    by_cases hlong : ∃ i j, P.L < ((draws i).2.2 j).toList.length
    · exact Or.inr hlong
    left
    push Not at hlong
    have htake := take_loopRounds settings O draws (le_of_lt hrK)
    simp only [Set.mem_ofPred_eq]
    rw [← htake]
    set rounds := loopRounds settings O draws K
    have hmem : ∀ zs ∈ harvestsOf (rounds.take r), ∃ i, ∃ hi : i < rounds.length, i < r
        ∧ rounds[i].result.ending = .harvest zs := by
      intro zs hzs
      obtain ⟨o, ho, hoz⟩ := List.mem_filterMap.1 hzs
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
      simp only [List.length_take] at hi
      refine ⟨i, by omega, by omega, ?_⟩
      rw [List.getElem_take] at hoz
      revert hoz
      cases rounds[i].result.ending <;> simp [RoundEnd.harvested]
    refine ⟨?_, fun zs hzs => ?_, fun zs hzs z hz => ?_, ?_⟩
    · have hall : AllHarvested (rounds.take r) := by
        intro o ho
        obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
        simp only [List.length_take] at hi
        rw [List.getElem_take]
        exact harvested_of_lt settings O draws K i (by omega)
      rw [length_harvestsOf hall, List.length_take]
      omega
    · obtain ⟨i, hi, hir, hz⟩ := hmem zs hzs
      exact hcolE _ (List.mem_take_iff_getElem.2 ⟨i, by omega, rfl⟩) zs
        hz
    · obtain ⟨i, hi, -, hzi⟩ := hmem zs hzs
      obtain ⟨hK', e⟩ := getElem_loopRounds settings O draws K i hi
      rw [e] at hzi
      have := harvest_short settings.level4 P.L (by rw [hmax]; omega) _ _
        (fun j => hlong ⟨i, hK'⟩ j) hzi z hz
      rw [hmax] at this
      omega
    · obtain ⟨hK', e⟩ := getElem_loopRounds settings O draws K r hr
      rw [e] at hspec
      exact hspec
  calc (loopMeasure μ P.D Dsf K T).real _
      ≤ (loopMeasure μ P.D Dsf K T).real
          ({draws | ((() : Unit), draws rK).2 ∈ Bad (harvestsOf (loopRounds settings O draws r))}
            ∪ {draws | ∃ i j, P.L < ((draws i).2.2 j).toList.length}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (loopMeasure μ P.D Dsf K T).real
          {draws | ((() : Unit), draws rK).2 ∈ Bad (harvestsOf (loopRounds settings O draws r))}
        + (loopMeasure μ P.D Dsf K T).real
          {draws | ∃ i j, P.L < ((draws i).2.2 j).toList.length} := measureReal_union_le _ _
    _ ≤ δc + 0 := by
        refine add_le_add (ENNReal.toReal_le_of_le_ofReal hδc hmain) ?_
        rw [Measure.real, loop_long_null P Dsf T, ENNReal.toReal_zero]
    _ = δc := add_zero δc

/-- Round `r`, meeting the clustering's claim, meets level 4's but for chance `2δ`. Given the
earlier rounds and the round's clustering, level 4 reads a suffix family drawn afresh. -/
theorem loop_level4_round_le [Nonempty α] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (settings : LoopSettings K) (S k Ls : ℕ) (εcov slack εd δ crossLimit : ℝ)
    (hLs : P.L + Fintype.card σ + S + 3 ≤ Ls)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hwalk : settings.level4.walkStart = k) (hdis : settings.level4.disagreementRate = εd)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hεd : 0 < εd) (hεd1 : εd ≤ 1) (hL1 : 1 ≤ P.L)
    (hcross : 0 < crossLimit)
    (hθ : 384 * (P.L + 1) * (Fintype.card σ + S + 1)
      * loopTheta σ settings.indecisionLimit slack ≤ 1)
    (hθ0 : 0 ≤ loopTheta σ settings.indecisionLimit slack)
    (hε : (P.L + 1) * (Fintype.card σ + S + 1) * crossLimit
      + 3 * unionLog (α := α) σ S δ * prefixMax P.D k
        ≤ (S + 1) * εd / (256 * (Fintype.card σ + S + 1)
          * (Fintype.card σ * Fintype.card α + S + 1)))
    (hlevel4 : ∀ (O' : TriOracle μ M) (θ : ℝ),
      let n : ℝ := Fintype.card σ + S
      let A : ℝ := Fintype.card α
      let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
      let V := undecidedCount P M O'.readDist
      let bad := fun q => ¬ O'.Good θ q
      0 ≤ θ → 384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
      0 < O'.κ → 4 * O'.κ * (8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1) ≤ ρ →
      0 ≤ O'.ε → (P.L + 1) * (n + 1) * O'.ε + 3 * unionLog (α := α) σ S δ * prefixMax P.D k
        ≤ ρ →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.level4.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel4 settings.level4 (O'.read · ω) draws
            ¬ EndsWell M bad P.D settings.level4.disagreementRate
              (searchingProbes (O'.read · ω) settings.level4.walkStart result.hypothesis)
              P.firstAnchor P.L (level4Replay (O'.read · ω) result.hypothesis)
              ((V - 2 * Real.sqrt (P.prefixBound * V * Real.log (1 / δ)))
                / (P.L + 1 - P.firstAnchor : ℕ))
              result.ending} ∂μ
        ≤ ENNReal.ofReal (2 * δ))
    (r : ℕ) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {draws | ∃ hr : r < (loopRounds settings O draws K).length,
        ClusterSpec settings O P.D εcov slack
            (harvestsOf ((loopRounds settings O draws K).take r))
            ((loopRounds settings O draws K)[r]'hr).run
          ∧ ¬ Level4Spec M O P (loopTheta σ settings.indecisionLimit slack) δ εd k
            ((loopRounds settings O draws K)[r]'hr)} ≤ 2 * δ := by
  sorry

end Rounds

/-! ## The theorem -/

lemma flat_of_length (Ls : ℕ) :
    Flat {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls} := by
  intro p hp p' hp' v hv v' hv' e
  have hp : p.toList.length < Ls := hp
  have hp' : p'.toList.length < Ls := hp'
  have hv : v = 1 ∨ v.toList.length = Ls := hv
  have hv' : v' = 1 ∨ v'.toList.length = Ls := hv'
  have hl : p.toList ++ v.toList = p'.toList ++ v'.toList := by
    rw [← FreeMonoid.toList_mul, ← FreeMonoid.toList_mul, e]
  have hlen : p.toList.length + v.toList.length = p'.toList.length + v'.toList.length := by
    rw [← List.length_append, ← List.length_append, hl]
  have hpp : p.toList.length = p'.toList.length := by
    rcases hv with rfl | hv <;> rcases hv' with rfl | hv' <;>
      first | omega | (simp only [FreeMonoid.toList_one, List.length_nil] at hlen; omega)
  exact FreeMonoid.toList.injective (List.append_inj_left hl hpp)

lemma sum_indicator_singleton {zs : List (FreeMonoid α)} (hnd : zs.Nodup) (a : FreeMonoid α) :
    (zs.map (Set.indicator {a} (1 : FreeMonoid α → ℝ))).sum = if a ∈ zs then 1 else 0 := by
  induction zs with
  | nil => simp
  | cons z zs ih =>
    rw [List.nodup_cons] at hnd
    rw [List.map_cons, List.sum_cons, ih hnd.2]
    by_cases hz : z = a
    · subst hz
      simp [hnd.1]
    · have : z ∉ ({a} : Set (FreeMonoid α)) := hz
      rw [Set.indicator_of_notMem this]
      by_cases ha : a ∈ zs
      · simp [ha]
      · simp [ha, Ne.symm hz]

/-- A harvest with no repeats is spread as thinly as its length allows. -/
lemma collisionMass_harvestMeasure {zs : List (FreeMonoid α)} (hzs : zs ≠ []) (hnd : zs.Nodup) :
    collisionMass (harvestMeasure zs) = 1 / zs.length := by
  have hL : (0 : ℝ) < zs.length := by exact_mod_cast List.length_pos_of_ne_nil hzs
  have hone : ∀ a, (harvestMeasure zs).real {a} = if a ∈ zs then (1 : ℝ) / zs.length else 0 := by
    intro a
    have h := integral_harvestMeasure (Set.indicator {a} (1 : FreeMonoid α → ℝ)) hzs
    rw [integral_indicator_one (measurableSet_singleton a), sum_indicator_singleton hnd] at h
    rw [h]
    split_ifs <;> simp
  unfold collisionMass
  simp only [hone]
  rw [tsum_eq_sum (s := zs.toFinset) fun a ha => by
    rw [List.mem_toFinset] at ha; simp [ha]]
  rw [Finset.sum_congr rfl fun a ha => by rw [if_pos (List.mem_toFinset.1 ha)],
    Finset.sum_const, List.toFinset_card_of_nodup hnd, nsmul_eq_mul]
  field_simp

section Spread

variable {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

/-- Round `r` harvests a list more concentrated than `cap` only where two of its probes share
their length-`k` prefix. -/
theorem loop_harvest_spread_le (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf] (settings : LoopSettings K)
    (cap : ℝ) (hcap : 0 < cap)
    (hsize : ∀ (read : FreeMonoid α → ARU) (draws : Fin settings.level4.probeBudget → FreeMonoid α)
      zs, (indecisiveLStarLevel4 settings.level4 read draws).ending = .harvest zs →
        ⌈1 / cap⌉₊ ≤ zs.length)
    (r : ℕ) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {draws | ∃ hr : r < (loopRounds settings O draws K).length, ∃ zs,
        ((loopRounds settings O draws K)[r]'hr).result.ending = .harvest zs
          ∧ cap < collisionMass (harvestMeasure zs)}
      ≤ settings.level4.probeBudget ^ 2 * prefixMax P.D settings.level4.walkStart := by
  classical
  have hpm : 0 ≤ prefixMax P.D settings.level4.walkStart := by
    have hbdd : BddAbove (Set.range fun u =>
        P.D.real {x | prefixOf x settings.level4.walkStart = u}) :=
      ⟨1, by rintro _ ⟨u, rfl⟩; exact measureReal_le_one⟩
    exact measureReal_nonneg.trans (le_ciSup hbdd 1)
  by_cases hrK : r < K
  swap
  · have hempty : {draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget |
        ∃ hr : r < (loopRounds settings O draws K).length, ∃ zs,
          ((loopRounds settings O draws K)[r]'hr).result.ending = .harvest zs
            ∧ cap < collisionMass (harvestMeasure zs)} = ∅ := by
      refine Set.eq_empty_of_forall_notMem fun draws hd => ?_
      obtain ⟨hr, -⟩ := hd
      have := length_loopRounds_le settings O draws K
      omega
    rw [hempty, measureReal_empty]
    positivity
  have : IsProbabilityMeasure (clusterLaw μ P.D Dsf K) := by unfold clusterLaw; infer_instance
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K settings.level4.probeBudget) := by
    unfold loopMeasure; infer_instance
  have hf : MeasurePreserving
      (fun draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget =>
        (draws ⟨r, hrK⟩).2.2)
      (loopMeasure μ P.D Dsf K settings.level4.probeBudget)
      (Measure.pi fun _ : Fin settings.level4.probeBudget => P.D) :=
    (measurePreserving_snd.comp measurePreserving_snd).comp (measurePreserving_eval _ (⟨r, hrK⟩ : Fin K))
  set Bad := {xs : Fin settings.level4.probeBudget → FreeMonoid α |
    ¬ Function.Injective fun i => prefixOf (xs i) settings.level4.walkStart}
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget |
      ∃ hr : r < (loopRounds settings O draws K).length, ∃ zs,
        ((loopRounds settings O draws K)[r]'hr).result.ending = .harvest zs
          ∧ cap < collisionMass (harvestMeasure zs)}
      ⊆ (fun draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget =>
        (draws ⟨r, hrK⟩).2.2) ⁻¹' Bad := by
    rintro draws ⟨hr, zs, hz, hc⟩ hinj
    obtain ⟨hK', e⟩ := getElem_loopRounds settings O draws K r hr
    rw [e] at hz
    have hnd := harvest_nodup settings.level4 _ (draws ⟨r, hK'⟩).2.2 hinj hz
    have hlen := hsize _ _ zs hz
    have hH0 : 0 < ⌈1 / cap⌉₊ := Nat.ceil_pos.2 (by positivity)
    have hzs : zs ≠ [] := List.ne_nil_of_length_pos (by omega)
    rw [collisionMass_harvestMeasure hzs hnd] at hc
    have h1 : (1 : ℝ) / zs.length ≤ 1 / ⌈1 / cap⌉₊ :=
      one_div_le_one_div_of_le (by exact_mod_cast hH0) (by exact_mod_cast hlen)
    have h2 : (1 : ℝ) / ⌈1 / cap⌉₊ ≤ cap :=
      (one_div_le (by exact_mod_cast hH0) hcap).2 (Nat.le_ceil _)
    linarith
  calc _ ≤ (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
        ((fun draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget =>
          (draws ⟨r, hrK⟩).2.2) ⁻¹' Bad) := measureReal_mono hsub (measure_ne_top _ _)
    _ = (Measure.pi fun _ : Fin settings.level4.probeBudget => P.D).real Bad := by
        simp only [Measure.real]
        rw [hf.measure_preimage (Set.to_countable Bad).measurableSet.nullMeasurableSet]
    _ ≤ _ := prefix_collision_le P.D _ _

end Spread

theorem learner_loop_correct_holds : LearnerLoopCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K S Ls η₀ ι gate εcov δc slack pAP εd δ n A Kp X G
    hL hη hη₀ hι hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ
  classical
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  obtain ⟨ρ, hρ⟩ : ∃ x : ℝ, x = (S + 1) * εd / (256 * (Fintype.card σ + S + 1)
      * (Fintype.card σ * Fintype.card α + S + 1)) := ⟨_, rfl⟩
  have hρ0 : 0 < ρ := by rw [hρ]; positivity
  obtain ⟨cross, hcrossd⟩ : ∃ x : ℝ, x = ρ / (2 * (P.L + 1) * (Fintype.card σ + S + 1)) :=
    ⟨_, rfl⟩
  have hcr : 0 < cross := by rw [hcrossd]; positivity
  obtain ⟨cap, hcap, schedule, hband, hclust⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc slack pAP hη hη₀ (flat_of_length Ls) hpAP hι hι1 hg hg1 hcr hε hε1
    hδc hδc1 hsl
  obtain ⟨B, hB⟩ : ∃ x : ℝ, x = Kp * (G + ⌈1 / cap⌉₊) := ⟨_, rfl⟩
  have hΛ : 0 < unionLog (α := α) σ S δ := by
    unfold unionLog
    refine mul_pos (by positivity) (Real.log_pos ?_)
    rw [one_lt_div hδ]
    have : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by linarith [(Nat.cast_nonneg S : (0 : ℝ) ≤ S),
      (Nat.cast_nonneg (Fintype.card σ) : (0 : ℝ) ≤ Fintype.card σ)]
    nlinarith
  obtain ⟨q₀, hq₀⟩ : ∃ x : ℝ, x = min (δ / (B ^ 2 + 1)) (ρ / (6 * unionLog (α := α) σ S δ)) :=
    ⟨_, rfl⟩
  have hq₀0 : 0 < q₀ := by rw [hq₀]; exact lt_min (by positivity) (by positivity)
  refine ⟨cap, hcap, q₀, hq₀0, fun k hk => ?_⟩
  rw [hq₀] at hk
  obtain ⟨s4, hwalk, hdis, hbud, hmax, hsize, hl4⟩ :=
    level4_guarantee_uniform (μ := μ) M εd P δ S k ⌈1 / cap⌉₊ hδ hδ1 hεd hεd1 hL1
  have hbud' : (s4.probeBudget : ℝ) ≤ B := by rw [hB]; exact hbud
  set settings : LoopSettings K := ⟨schedule, ι, gate, s4⟩ with hset
  refine ⟨settings, rfl, rfl, hwalk, hdis, hbud, ?_⟩
  intro Dsf hDsf hDsfLen hpAPDsf hcolD hcolSf
  have hθ' : 384 * (P.L + 1) * (Fintype.card σ + S + 1) * loopTheta σ ι slack ≤ 1 := by
    have e : 384 * ((P.L : ℝ) + 1) * (Fintype.card σ + S + 1) * loopTheta σ ι slack
        = 512 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * ι + slack) := by
      simp only [loopTheta, n]; ring
    rw [e]; exact hθ
  have hθ0 : 0 ≤ loopTheta σ ι slack := by
    unfold loopTheta; positivity
  have hpm : 0 ≤ prefixMax P.D k := by
    have hbdd : BddAbove (Set.range fun u => P.D.real {x | prefixOf x k = u}) :=
      ⟨1, by rintro _ ⟨u, rfl⟩; exact measureReal_le_one⟩
    exact measureReal_nonneg.trans (le_ciSup hbdd 1)
  have hcross : (P.L + 1) * (Fintype.card σ + S + 1) * cross
      + 3 * unionLog (α := α) σ S δ * prefixMax P.D k
        ≤ (S + 1) * εd / (256 * (Fintype.card σ + S + 1)
          * (Fintype.card σ * Fintype.card α + S + 1)) := by
    have e1 : ((P.L : ℝ) + 1) * (Fintype.card σ + S + 1) * cross = ρ / 2 := by
      rw [hcrossd]; field_simp
    have e2 : 3 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ / 2 := by
      have h1 : prefixMax P.D k ≤ ρ / (6 * unionLog (α := α) σ S δ) :=
        hk.trans (min_le_right _ _)
      rw [le_div_iff₀ (by positivity)] at h1
      linarith
    rw [e1, ← hρ]
    linarith
  set E1 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O draws K).length,
      (∀ o ∈ (loopRounds settings O draws K).take r, ∀ zs, o.result.ending = .harvest zs →
        collisionMass (harvestMeasure zs) ≤ cap)
      ∧ ¬ ClusterSpec settings O P.D εcov slack
        (harvestsOf ((loopRounds settings O draws K).take r))
        ((loopRounds settings O draws K)[r]'hr).run} with hE1
  set E2 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O draws K).length,
      ClusterSpec settings O P.D εcov slack
          (harvestsOf ((loopRounds settings O draws K).take r))
          ((loopRounds settings O draws K)[r]'hr).run
        ∧ ¬ Level4Spec M O P (loopTheta σ ι slack) δ εd k
          ((loopRounds settings O draws K)[r]'hr)} with hE2
  set E3 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O draws K).length, ∃ zs,
      ((loopRounds settings O draws K)[r]'hr).result.ending = .harvest zs
        ∧ cap < collisionMass (harvestMeasure zs)} with hE3
  have h1 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E1 r) ≤ δc := fun r =>
    loop_cluster_round_le (σ := σ) O P Dsf settings S Ls εcov δc slack cap hδc.le hmax hLs
      hcolD (fun pops hpops D hD hDlen hDcol =>
        hclust pops hpops D Dsf hD hDsf (fun j hj => by
          refine measure_mono_null (fun p hp => ?_) (hDlen j hj)
          simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_lt] at hp ⊢
          exact hp)
          (by
            refine measure_mono_null (fun v hv => ?_) hDsfLen
            simp only [Set.mem_compl_iff, Set.mem_ofPred_eq] at hv ⊢
            exact hv) hpAPDsf hDcol hcolSf) r
  have h2 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E2 r) ≤ 2 * δ := fun r =>
    loop_level4_round_le M O hL P Dsf settings S k Ls εcov slack εd δ cross hLs hDsfLen hband
      hwalk hdis hδ hδ1 hεd hεd1 hL1 hcr hθ' hθ0 hcross
      (fun O' θ => hl4 O' θ) r
  have h3 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E3 r) ≤ δ := by
    intro r
    refine (loop_harvest_spread_le O P Dsf settings cap hcap hsize r).trans ?_
    rw [hwalk]
    have hT : (s4.probeBudget : ℝ) ^ 2 ≤ B ^ 2 :=
      pow_le_pow_left₀ (Nat.cast_nonneg _) hbud' 2
    have hq : prefixMax P.D k ≤ δ / (B ^ 2 + 1) := hk.trans (min_le_left _ _)
    have hB2 : B ^ 2 * (δ / (B ^ 2 + 1)) ≤ δ := by
      rw [← mul_div_assoc, div_le_iff₀ (by positivity)]
      nlinarith only [hδ]
    calc ((s4.probeBudget : ℕ) : ℝ) ^ 2 * prefixMax P.D k ≤ B ^ 2 * (δ / (B ^ 2 + 1)) :=
          mul_le_mul hT hq hpm (sq_nonneg B)
      _ ≤ δ := hB2
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K s4.probeBudget |
      let rounds := indecisiveLStarLoop settings O draws
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.Agrees O P.D k (εcov + 2 * ι + 2 * slack) (2 * ι + slack) εd)}
      ⊆ ⋃ r ∈ Finset.range (Fintype.card σ + 1), (E1 r ∪ (E2 r ∪ E3 r)) := by
    intro draws hd
    by_contra hno
    simp only [Set.mem_iUnion, Set.mem_union, Finset.mem_range, not_exists, not_or] at hno
    have hspec : ∀ i (hi : i < (loopRounds settings O draws K).length), i ≤ Fintype.card σ →
        ClusterSpec settings O P.D εcov slack
            (harvestsOf ((loopRounds settings O draws K).take i))
            (loopRounds settings O draws K)[i].run
          ∧ Level4Spec M O P (loopTheta σ settings.indecisionLimit slack) δ εd k
            (loopRounds settings O draws K)[i] := by
      intro i hi hiσ
      have hcol : ∀ o ∈ (loopRounds settings O draws K).take i, ∀ zs,
          o.result.ending = .harvest zs → collisionMass (harvestMeasure zs) ≤ cap := by
        intro o ho zs hzs
        obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
        rw [List.getElem_take] at hzs
        have hji : j < i := by
          simp only [List.length_take] at hj
          omega
        by_contra hc
        push Not at hc
        exact (hno j (by omega)).2.2 ⟨_, zs, hzs, hc⟩
      have hc : ClusterSpec settings O P.D εcov slack
          (harvestsOf ((loopRounds settings O draws K).take i))
          (loopRounds settings O draws K)[i].run := by
        by_contra hc
        exact (hno i (by omega)).1 ⟨hi, hcol, hc⟩
      refine ⟨hc, ?_⟩
      by_contra h4
      exact (hno i (by omega)).2.1 ⟨hi, hc, h4⟩
    exact hd (loop_agrees_of_specs M O hL P settings draws k εcov slack εd δ hK
      (fun pops B hB => (hband pops B hB).1) hspec)
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K s4.probeBudget) := by
    unfold loopMeasure clusterLaw
    infer_instance
  calc (loopMeasure μ P.D Dsf K s4.probeBudget).real _
      ≤ (loopMeasure μ P.D Dsf K s4.probeBudget).real
          (⋃ r ∈ Finset.range (Fintype.card σ + 1), (E1 r ∪ (E2 r ∪ E3 r))) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (Fintype.card σ + 1),
          (loopMeasure μ P.D Dsf K s4.probeBudget).real (E1 r ∪ (E2 r ∪ E3 r)) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (Fintype.card σ + 1), (δc + 3 * δ) := by
        refine Finset.sum_le_sum fun r _ => (measureReal_union_le _ _).trans ?_
        have := (measureReal_union_le (μ := loopMeasure μ P.D Dsf K s4.probeBudget)
          (E2 r) (E3 r)).trans (add_le_add (h2 r) (h3 r))
        linarith [h1 r]
    _ = (Fintype.card σ + 1) * (δc + 3 * δ) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

end OrthoDFA

end
