import OrthoDFA.Proofs.LoopTermination
import OrthoDFA.Proofs.Level4Correct
import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.BandWide
import OrthoDFA.Proofs.LoopIndep
import OrthoDFA.Proofs.LoopLaw
import OrthoDFA.Proofs.Level4Short
import OrthoDFA.Proofs.LoopMeasurable
import Mathlib.Data.List.GetD

/-!
# The loop's theorem

Each round meets the clustering's claim but for chance `δc` and, given that, level 4's but for
chance `δ`; on the rounds' draws these are the one-round claims, at the populations and the
family that round was handed (`loop_cluster_round_le`, `loop_level4_round_le`). A round that
meets level 4's claim hands on a population that puts at most `prefixMax / ρ` on any string, so
within the clustering's cap. Where every one of the first `|Q| + 1` rounds meets both claims, the
loop stops in agreement (`loop_agrees_of_specs`).
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
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hslack : 0 < slack) (cap : ℝ)
    (hcap : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      cap ≤ qualityCap η₀ pops indecisionLimit εcov gateLevel δc pAP
        (Real.exp (-2 / crossLimit)) slack) :
    ∃ schedule : Finset (Option (Fin K)) → Finset State,
      (∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ,
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
  rw [← hc] at hcap
  refine ⟨fun pops =>
    stoppable η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP c cap, ?_, ?_⟩
  · intro pops B hB
    obtain ⟨hlohi, hk2, hband⟩ := band_of_mem_schedule (le_trans (eta_nonneg O) hη) hη₀
      (Finset.mem_of_mem_filter _ hB)
    refine ⟨hlohi, hk2, fun N hN p r ε₂ κ hp0 hp1 hr0 hr1 _ =>
      bandPasses_of_wide hp0 hp1 hr0 hr1 hlohi ?_⟩
    have hlog : Real.log (1 / c) = 2 / crossLimit := by
      rw [hc, one_div, ← Real.exp_neg, Real.log_exp]; ring
    rw [hlog, mul_div_assoc', div_le_iff₀ hcross] at hband
    have hNk : (N : ℝ) + 1 ≤ B.k := by exact_mod_cast hN
    have hlh : (B.lo : ℝ) ≤ B.hi := by exact_mod_cast hlohi
    have h2 : ((B.hi : ℝ) - B.lo) ^ 2 ≤ ((B.hi : ℝ) + 1 - B.lo) ^ 2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h2 hcross.le]
  · intro pops hpops D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolD hcolSf
    obtain ⟨-, hq⟩ := clustering_quality_bad O pops none Pre Suf η₀ indecisionLimit εcov gateLevel
      δc pAP c slack hη hη₀ hpops hflat hpAP hι hι1 hg hg1 hε hε1 hδc hδc1 hc0 hslack
    obtain ⟨-, hbad⟩ := hq D Dsf hD hDsf hsupp hsuppSf hpAPDsf cap hcolD
      (hcap pops hpops) (hcolSf.trans (hcap pops hpops))
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

lemma lt_of_some_mem_loopPopulations {n : ℕ} {i : Fin K} (h : some i ∈ loopPopulations K n) :
    i.val < n := by
  simp only [loopPopulations, Finset.mem_insert, reduceCtorEq, false_or, Finset.mem_image,
    Finset.mem_filter, Finset.mem_univ, true_and, Option.some.injEq] at h
  obtain ⟨a, ha, rfl⟩ := h
  exact ha

omit [IsProbabilityMeasure μ] in
lemma popLaw_null {ν : Measure (FreeMonoid α)} {A : Set (FreeMonoid α)} (hA : ν A = 0)
    (h1 : (1 : FreeMonoid α) ∉ A) : popLaw ν A = 0 := by
  unfold popLaw
  split_ifs
  · rw [Measure.dirac_apply' _ (measurableSet_of_countable A), Set.indicator_of_notMem h1]
  · rw [Measure.smul_apply, hA, smul_zero]

omit [IsProbabilityMeasure μ] in
/-- A round's population is supported on strings shorter than `Ls`. -/
lemma population_short (P : PrefixDistribution α) (C : Level4Settings) (hC : 2 ≤ C.maxLeaves)
    {Ls : ℕ} (hLs : P.L + C.maxLeaves ≤ Ls) (k : ℕ) (o : LoopRound Ω α K) {T : ℕ}
    (xs : Fin T → FreeMonoid α) (ho : o.result = indecisiveLStarLevel4 C o.read xs)
    {ν : Measure (FreeMonoid α)} (hν : o.population P.D k = some ν) :
    ν {p | Ls ≤ p.toList.length} = 0 := by
  unfold LoopRound.population at hν
  rcases hh : o.result.ending.harvested with _ | h
  · rw [hh] at hν; cases hν
  rw [hh, Option.map_some, Option.some.injEq] at hν
  subst hν
  have hend : (indecisiveLStarLevel4 C o.read xs).ending = .harvest h := by
    rw [← ho]
    revert hh
    cases o.result.ending <;> simp [RoundEnd.harvested]
  refine popLaw_null (measure_mono_null (fun p hp => ?_)
    (harvestLaw_short C (fun z => (o.read z).cut) P.length_le k
      (harvest_short C hC o.read xs hend))) ?_
  · simp only [Set.mem_setOf_eq] at hp ⊢
    omega
  · simp only [Set.mem_setOf_eq, FreeMonoid.toList_one, List.length_nil, not_le]
    omega

/-- Round `r` meets the clustering's claim but for chance `δc`, unless an earlier round handed on
a population too concentrated for it. Given the earlier rounds, the round's run is drawn as
`runMeasure` draws it at their populations, which the earlier rounds' finite data fix and which
put nothing on strings as long as `Ls`. -/
theorem loop_cluster_round_le (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (settings : LoopSettings K) (S Ls N : ℕ) (εcov δc slack cap : ℝ) (hδc : 0 ≤ δc)
    (hmax : settings.level4.maxLeaves = Fintype.card σ + S + 3)
    (hLs : P.L + Fintype.card σ + S + 3 ≤ Ls) (hN : P.L + settings.level4.maxLeaves ≤ N)
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
      {draws | ∃ hr : r < (loopRounds settings O P.D draws K).length,
        (∀ ν ∈ harvestsOf P.D settings.level4.walkStart
            ((loopRounds settings O P.D draws K).take r), collisionMass ν ≤ cap)
        ∧ ¬ ClusterSpec settings O P.D εcov slack
          (harvestsOf P.D settings.level4.walkStart ((loopRounds settings O P.D draws K).take r))
          ((loopRounds settings O P.D draws K)[r]'hr).run} ≤ δc := by
  classical
  have : IsProbabilityMeasure (clusterLaw μ P.D Dsf K) := by unfold clusterLaw; infer_instance
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K settings.level4.probeBudget) := by
    unfold loopMeasure; infer_instance
  set T := settings.level4.probeBudget with hT
  set k := settings.level4.walkStart with hk
  have hC : 2 ≤ settings.level4.maxLeaves := by rw [hmax]; omega
  by_cases hrK : r < K
  swap
  · have hempty : {draws : Fin K → LoopRoundDraws Ω α K T |
        ∃ hr : r < (loopRounds settings O P.D draws K).length,
          (∀ ν ∈ harvestsOf P.D k ((loopRounds settings O P.D draws K).take r),
            collisionMass ν ≤ cap)
          ∧ ¬ ClusterSpec settings O P.D εcov slack
            (harvestsOf P.D k ((loopRounds settings O P.D draws K).take r))
            ((loopRounds settings O P.D draws K)[r]'hr).run} = ∅ := by
      refine Set.eq_empty_of_forall_notMem fun draws hd => ?_
      obtain ⟨hr, -⟩ := hd
      have := length_loopRounds_le settings O P.D draws K
      omega
    rw [hempty, measureReal_empty]
    exact hδc
  set rK : Fin K := ⟨r, hrK⟩
  set ν : Measure (LoopRoundDraws Ω α K T) :=
    (clusterLaw μ P.D Dsf K).prod (μ.prod (Measure.pi fun _ : Fin T => P.D))
  have hLs' : P.L < Ls := by omega
  set dec : HarvestCode α → Measure (FreeMonoid α) := decodePop P.D k with hdec
  have hdecP : ∀ c, IsProbabilityMeasure (dec c) := fun c => by
    simp only [hdec, decodePop]; infer_instance
  -- the round's bad draws at given earlier data
  set Bad : List (HarvestCode α) → Set (LoopRoundDraws Ω α K T) := fun c =>
    {d | c.length = r ∧ (∀ ν' ∈ c.map dec,
        collisionMass ν' ≤ cap ∧ ν' {p | Ls ≤ p.toList.length} = 0)
      ∧ ¬ ClusterSpec settings O P.D εcov slack (c.map dec) (clusterRun (c.map dec) d.1)}
    with hBadd
  have hBad : ∀ c, ν (Bad c) ≤ ENNReal.ofReal δc := by
    intro c
    by_cases hH : ∀ ν' ∈ c.map dec, collisionMass ν' ≤ cap ∧ ν' {p | Ls ≤ p.toList.length} = 0
    swap
    · have : Bad c = ∅ := Set.eq_empty_of_forall_notMem fun d hd => hH hd.2.1
      rw [this, measure_empty]
      exact zero_le
    set H := c.map dec
    have hpops : ∀ ν' ∈ H, IsProbabilityMeasure ν' := fun ν' h => by
      obtain ⟨c', -, rfl⟩ := List.mem_map.1 h; exact hdecP c'
    set Tset := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      ¬ ClusterSpec settings O P.D εcov slack H x}
    have hcr := measurePreserving_clusterRun (μ := μ) (K := K) P.D Dsf H hpops
    have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) P.D H j) :=
      isProbabilityMeasure_loopDistributions P.D H hpops
    have hgetD : ∀ i : Fin K, some i ∈ loopPopulations K H.length →
        H.getD i (Measure.dirac 1) ∈ H := by
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
      | some i => exact (hH _ (hgetD i hj)).2
    have hcolp : ∀ j ∈ loopPopulations K H.length,
        collisionMass (loopDistributions P.D H j) ≤ cap := by
      intro j hj
      cases j with
      | none => exact hcolD
      | some i => exact (hH _ (hgetD i hj)).1
    have hc := hclust (loopPopulations K H.length) (Finset.mem_insert_self _ _)
      (loopDistributions P.D H) (isProbabilityMeasure_loopDistributions P.D H hpops) hsupp hcolp
    calc ν (Bad c) ≤ ν (Prod.fst ⁻¹' (clusterRun H ⁻¹' Tset)) :=
          measure_mono fun d hd => hd.2.2
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
    (fun draws => codesOf N (loopRounds settings O P.D draws r))
    (fun x x' hxx' _ => by
      rw [loopRounds_congr settings O P.D x x' r fun i hi => hxx' i hi])
    (measurableSet_codesOf settings O P.D P.length_le hC hN r) Bad (ENNReal.ofReal δc) hBad
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K T |
        ∃ hr : r < (loopRounds settings O P.D draws K).length,
          (∀ ν ∈ harvestsOf P.D k ((loopRounds settings O P.D draws K).take r),
            collisionMass ν ≤ cap)
          ∧ ¬ ClusterSpec settings O P.D εcov slack
            (harvestsOf P.D k ((loopRounds settings O P.D draws K).take r))
            ((loopRounds settings O P.D draws K)[r]'hr).run}
      ⊆ {draws | ((() : Unit), draws rK).2 ∈ Bad (codesOf N (loopRounds settings O P.D draws r))}
      := by
    rintro draws ⟨hr, hcolE, hspec⟩
    have htake := take_loopRounds settings O P.D draws (le_of_lt hrK)
    simp only [Set.mem_ofPred_eq]
    set rounds := loopRounds settings O P.D draws r
    have hres := loopRounds_result settings O P.D draws r
    have hcd : (codesOf N rounds).map dec = harvestsOf P.D k rounds :=
      codesOf_decode settings.level4 P.length_le hC hN k rounds hres
    rw [htake] at hcolE hspec
    have hall : AllHarvested rounds := by
      intro o ho
      rw [← htake] at ho
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
      simp only [List.length_take] at hi
      rw [List.getElem_take]
      exact harvested_of_lt settings O P.D draws K i (by omega)
    refine ⟨?_, fun ν' hν' => ?_, ?_⟩
    · rw [length_codesOf N P.D k rounds, length_harvestsOf hall, ← htake, List.length_take]
      omega
    · rw [hcd] at hν'
      refine ⟨hcolE ν' hν', ?_⟩
      obtain ⟨o, ho, hoν⟩ := List.mem_filterMap.1 hν'
      obtain ⟨T', xs, hxs⟩ := hres o ho
      exact population_short P settings.level4 hC (by rw [hmax]; omega) k o xs hxs hoν
    · obtain ⟨hK', e⟩ := getElem_loopRounds settings O P.D draws K r hr
      rw [e] at hspec
      rw [hcd]
      rw [htake] at hspec
      exact hspec
  calc (loopMeasure μ P.D Dsf K T).real _
      ≤ (loopMeasure μ P.D Dsf K T).real
          {draws | ((() : Unit), draws rK).2 ∈ Bad (codesOf N (loopRounds settings O P.D draws r))}
        := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ δc := ENNReal.toReal_le_of_le_ofReal hδc hmain

instance : Countable State :=
  Function.Injective.countable (f := fun B : State => (B.nsuff, B.npref, B.k, B.lo, B.hi, B.sc,
    B.scd)) fun a b h => by
    cases a; cases b
    simp only [Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
    subst h1 h2 h3 h4 h5 h6 h7
    rfl

/-- `O` with a rate no string has set to zero. -/
def Oracle.clamped (O : Oracle μ (FreeMonoid α)) : Oracle μ (FreeMonoid α) :=
  { O with
    ηIn := max O.ηIn 0
    ηOut := max O.ηOut 0
    noise_mean_in := fun w hw => by
      have h := O.rate_nonneg w
      unfold Oracle.rate at h
      rw [if_pos hw] at h
      rw [O.noise_mean_in w hw, max_eq_left h]
    noise_mean_out := fun w hw => by
      have h := O.rate_nonneg w
      unfold Oracle.rate at h
      rw [if_neg hw] at h
      rw [O.noise_mean_out w hw, max_eq_left h] }

/-- Round `r`, meeting the clustering's claim, meets level 4's but for chance `δ`. Given the
earlier rounds and the round's clustering, level 4 reads a suffix family drawn afresh. -/
theorem loop_level4_round_le [Nonempty α] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept)
    (P : PrefixDistribution α) (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (settings : LoopSettings K) (S k Ls N : ℕ) (εcov slack εd δ crossLimit y : ℝ)
    (hmax : settings.level4.maxLeaves = Fintype.card σ + S + 3)
    (hN : P.L + settings.level4.maxLeaves ≤ N)
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hη1 : O.η ≤ 1)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hwalk : settings.level4.walkStart = k) (hdis : settings.level4.disagreementRate = εd)
    (hδ : 0 < δ) (hεd : 0 < εd) (hcross : 0 < crossLimit)
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
      let bad := fun q => ¬ O'.Good θ q
      0 ≤ θ → 384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
      0 < O'.κ → 4 * O'.κ * (8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1) ≤ ρ →
      0 ≤ O'.ε → (P.L + 1) * (n + 1) * O'.ε + 3 * unionLog (α := α) σ S δ * prefixMax P.D k
        ≤ ρ →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.level4.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel4 settings.level4 (O'.read · ω) draws
            ¬ EndsWell M bad P.D settings.level4.disagreementRate
              (searchingProbes (O'.read · ω) settings.level4.walkStart result.hypothesis)
              settings.level4.walkStart
              (harvestLaw P.D (fun z => (O'.read z ω).cut) settings.level4.walkStart) y
              result.ending} ∂μ
        ≤ ENNReal.ofReal δ)
    (r : ℕ) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {draws | ∃ hr : r < (loopRounds settings O P.D draws K).length,
        ClusterSpec settings O P.D εcov slack
            (harvestsOf P.D k ((loopRounds settings O P.D draws K).take r))
            ((loopRounds settings O P.D draws K)[r]'hr).run
          ∧ ¬ Level4Spec M O P (loopTheta σ settings.indecisionLimit slack) εd y k
            ((loopRounds settings O P.D draws K)[r]'hr)} ≤ δ := by
  classical
  have : IsProbabilityMeasure (clusterLaw μ P.D Dsf K) := by unfold clusterLaw; infer_instance
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K settings.level4.probeBudget) := by
    unfold loopMeasure; infer_instance
  set T := settings.level4.probeBudget with hT
  set θ := loopTheta σ settings.indecisionLimit slack
  have hC : 2 ≤ settings.level4.maxLeaves := by rw [hmax]; omega
  by_cases hrK : r < K
  swap
  · have hempty : {draws : Fin K → LoopRoundDraws Ω α K T |
        ∃ hr : r < (loopRounds settings O P.D draws K).length,
          ClusterSpec settings O P.D εcov slack
              (harvestsOf P.D k ((loopRounds settings O P.D draws K).take r))
              ((loopRounds settings O P.D draws K)[r]'hr).run
            ∧ ¬ Level4Spec M O P θ εd y k ((loopRounds settings O P.D draws K)[r]'hr)} = ∅ := by
      refine Set.eq_empty_of_forall_notMem fun draws hd => ?_
      obtain ⟨hr, -⟩ := hd
      have := length_loopRounds_le settings O P.D draws K
      omega
    rw [hempty, measureReal_empty]
    positivity
  set rK : Fin K := ⟨r, hrK⟩
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set x₀ : Run Ω (FreeMonoid α) (Option (Fin K)) := (ω₀, ((fun _ => 1, fun _ _ => 1), fun _ _ => 1))
  set ν₂ : Measure (Ω × (Fin T → FreeMonoid α)) := μ.prod (Measure.pi fun _ : Fin T => P.D)
  set dec : HarvestCode α → Measure (FreeMonoid α) := decodePop P.D settings.level4.walkStart
    with hdec
  have hdecP : ∀ c, IsProbabilityMeasure (dec c) := fun c => by
    simp only [hdec, decodePop]; infer_instance
  have hdecL : ∀ c : List (HarvestCode α), ∀ ν' ∈ c.map dec, IsProbabilityMeasure ν' := by
    intro c ν' h
    obtain ⟨c', -, rfl⟩ := List.mem_map.1 h; exact hdecP c'
  set Y := List (HarvestCode α) × State × Finset (FreeMonoid α)
  set Bad : Y → Set (Ω × (Fin T → FreeMonoid α)) := fun yv =>
    {d | SuffixFree yv.2.2 ∧ yv.2.2.card + 1 ≤ yv.2.1.k
      ∧ yv.2.1 ∈ settings.schedule (loopPopulations K (yv.1.map dec).length)
      ∧ ¬ Level4Spec M O P θ εd y k (mkRound settings O x₀ yv.2.1 yv.2.2 d)} with hBadd
  -- the family's reads, as an oracle
  set n : ℝ := (Fintype.card σ : ℝ) + S
  set A : ℝ := (Fintype.card α : ℝ)
  set ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
  have hρ0 : 0 < ρ := by positivity
  set W : ℝ := 8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1
  have hW : 0 < W := by positivity
  set κ₀ : ℝ := ρ / (4 * W)
  have hκ₀ : 0 < κ₀ := by positivity
  have hκ₀W : 4 * κ₀ * W ≤ ρ := by
    simp only [κ₀]
    rw [show 4 * (ρ / (4 * W)) * W = ρ by field_simp]
  set Oc := O.clamped
  have hηIn : 0 ≤ 1 - Oc.ηIn ∧ 1 - Oc.ηIn ≤ 1 := by
    have h1 : O.ηIn ≤ O.η := le_max_left _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; have := le_max_left O.ηIn 0
              exact sub_nonneg.2 (max_le (by linarith) (by norm_num)),
      by simp only [Oc, Oracle.clamped]; linarith [le_max_right O.ηIn 0]⟩
  have hηOut : 0 ≤ Oc.ηOut ∧ Oc.ηOut ≤ 1 := by
    have h1 : O.ηOut ≤ O.η := le_max_right _ _
    exact ⟨by simp only [Oc, Oracle.clamped]; exact le_max_right _ _,
      by simp only [Oc, Oracle.clamped]; exact max_le (by linarith) (by norm_num)⟩
  have hBad : ∀ yv, ν₂ (Bad yv) ≤ ENNReal.ofReal δ := by
    rintro ⟨H, B, F⟩
    by_cases hc : SuffixFree F ∧ F.card + 1 ≤ B.k
      ∧ B ∈ settings.schedule (loopPopulations K (H.map dec).length)
    swap
    · have : Bad (H, B, F) = ∅ :=
        Set.eq_empty_of_forall_notMem fun d hd => hc ⟨hd.1, hd.2.1, hd.2.2.1⟩
      rw [this, measure_empty]
      exact zero_le
    obtain ⟨hsf, hcard, hB⟩ := hc
    have hb := (hband _ B hB).2.2 F.card hcard (1 - Oc.ηIn) Oc.ηOut 0 κ₀ hηIn.1 hηIn.2
      hηOut.1 hηOut.2 hκ₀.le
    let fam : SuffixFamily μ α σ := ⟨Oc, M, F, B.lo, B.hi + 1, crossLimit, 0, κ₀, hL, hsf, hb⟩
    have h4 := hlevel4 fam.oracle θ hθ0 hθ hκ₀ hκ₀W hcross.le hε
    rw [hdis, hwalk] at h4
    have hmeas := measurableSet_level4Fail M O P θ εd y k settings hC x₀ B F
    have hBe : Bad (H, B, F) = {d : Ω × (Fin T → FreeMonoid α) |
        ¬ Level4Spec M O P θ εd y k (mkRound settings O x₀ B F d)} := by
      ext d
      simp only [hBadd, Set.mem_ofPred_eq]
      exact ⟨fun h => h.2.2.2, fun h => ⟨hsf, hcard, hB, h⟩⟩
    rw [hBe, Measure.prod_apply hmeas]
    refine le_trans (lintegral_mono fun ω => measure_mono fun xs hxs => ?_) h4
    exact hxs
  have hσ : MeasurePreserving (id : LoopRoundDraws Ω α K T → LoopRoundDraws Ω α K T)
      ((clusterLaw μ P.D Dsf K).prod ν₂) ((clusterLaw μ P.D Dsf K).prod ν₂) :=
    MeasurePreserving.id _
  set codes : (Fin K → LoopRoundDraws Ω α K T) → List (HarvestCode α) := fun x =>
    codesOf N (loopRounds settings O P.D x r) with hcodes
  set stage : (Fin K → LoopRoundDraws Ω α K T) → Y := fun x =>
    (codes x,
      roundState settings O ((codes x).map dec) (x rK).1,
      clusterAt O.mq (loopPopulations K ((codes x).map dec).length)
        (clusterRun ((codes x).map dec) (x rK).1)
        (roundState settings O ((codes x).map dec) (x rK).1)) with hstage
  have hdep : ∀ x x' : Fin K → LoopRoundDraws Ω α K T, (∀ i, i < rK → x i = x' i) →
      (id (x rK)).1 = (id (x' rK)).1 → stage x = stage x' := by
    intro x x' hxx' h1
    have hH := loopRounds_congr settings O P.D x x' r fun i hi => hxx' i hi
    simp only [id] at h1
    simp only [hstage, hcodes, hH, h1]
  have hmeas : ∀ yv, MeasurableSet {x | stage x = yv} := by
    rintro ⟨H, B, F⟩
    have hset : {x | stage x = (H, B, F)} = {x : Fin K → LoopRoundDraws Ω α K T |
        codesOf N (loopRounds settings O P.D x r) = H}
        ∩ (fun x : Fin K → LoopRoundDraws Ω α K T => (x rK).1) ⁻¹'
          {c | roundState settings O (H.map dec) c = B
            ∧ clusterAt O.mq (loopPopulations K (H.map dec).length) (clusterRun (H.map dec) c)
              (roundState settings O (H.map dec) c) = F} := by
      ext x
      simp only [Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_preimage]
      constructor
      · intro h
        simp only [hstage] at h
        obtain ⟨h1, h23⟩ := Prod.mk.inj h
        obtain ⟨h2, h3⟩ := Prod.mk.inj h23
        simp only [hcodes] at h1 h2 h3
        subst h1
        exact ⟨rfl, h2, h3⟩
      · rintro ⟨h1, h2, h3⟩
        simp only [hstage, hcodes]
        rw [h1]
        exact Prod.ext rfl (Prod.ext h2 h3)
    rw [hset]
    exact (measurableSet_codesOf settings O P.D P.length_le hC hN r H).inter
      (measurable_fst.comp (measurable_pi_apply rK)
        (measurableSet_roundPick settings O (H.map dec) (hdecL H) B F))
  have hmain := round_stage_le ((clusterLaw μ P.D Dsf K).prod ν₂) (clusterLaw μ P.D Dsf K) ν₂ id
    hσ id measurable_id (fun w => rfl) rK stage hdep hmeas Bad (ENNReal.ofReal δ) hBad
  -- a suffix of the wrong length is drawn with chance zero
  set N' := {x : Fin K → LoopRoundDraws Ω α K T | ∃ m, ((x rK).1.2.1 m).toList.length ≠ Ls}
  have hN' : loopMeasure μ P.D Dsf K T N' = 0 := by
    have hset : N' = ⋃ m, (fun x : Fin K → LoopRoundDraws Ω α K T => (x rK).1.2.1 m) ⁻¹'
        {v | v.toList.length ≠ Ls} := by
      ext x; simp [N']
    rw [hset]
    refine measure_iUnion_null fun m => ?_
    have hev : MeasurePreserving (fun u : ℕ → FreeMonoid α => u m)
        (Measure.infinitePi fun _ : ℕ => Dsf) Dsf :=
      ⟨measurable_pi_apply m, Measure.infinitePi_map_eval _ m⟩
    have hf : MeasurePreserving (fun x : Fin K → LoopRoundDraws Ω α K T => (x rK).1.2.1 m)
        (Measure.pi fun _ : Fin K => (clusterLaw μ P.D Dsf K).prod ν₂) Dsf :=
      hev.comp (measurePreserving_fst.comp (measurePreserving_snd.comp
        (measurePreserving_fst.comp (measurePreserving_eval _ rK))))
    unfold loopMeasure
    rw [hf.measure_preimage (measurableSet_of_countable _).nullMeasurableSet]
    exact hDsf
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K T |
        ∃ hr : r < (loopRounds settings O P.D draws K).length,
          ClusterSpec settings O P.D εcov slack
              (harvestsOf P.D k ((loopRounds settings O P.D draws K).take r))
              ((loopRounds settings O P.D draws K)[r]'hr).run
            ∧ ¬ Level4Spec M O P θ εd y k ((loopRounds settings O P.D draws K)[r]'hr)}
      ⊆ {x | (id (x rK)).2 ∈ Bad (stage x)} ∪ N' := by
    rintro x ⟨hr, hspec, hnot⟩
    by_cases hlong : x ∈ N'
    · exact Or.inr hlong
    left
    simp only [N', Set.mem_ofPred_eq, not_exists, not_not] at hlong
    simp only [id, Set.mem_ofPred_eq, hstage, hBadd]
    obtain ⟨hK', e⟩ := getElem_loopRounds settings O P.D x K r hr
    rw [take_loopRounds settings O P.D x (le_of_lt hrK)] at e hspec
    have hcd : (codes x).map dec = harvestsOf P.D k (loopRounds settings O P.D x r) := by
      rw [← hwalk]
      exact codesOf_decode settings.level4 P.length_le hC hN _ (loopRounds settings O P.D x r)
        (loopRounds_result settings O P.D x r)
    rw [hwalk] at e
    set H := harvestsOf P.D k (loopRounds settings O P.D x r)
    have hxr : x ⟨r, hK'⟩ = x rK := rfl
    rw [e, hxr] at hspec hnot
    rw [hcd]
    have hret := loopRound_returns settings O H (x rK) hspec.1
    set B := roundState settings O H (x rK).1
    set c := (x rK).1
    have hBs : B ∈ settings.schedule (loopPopulations K H.length) := hret.1
    obtain ⟨-, hk2, -⟩ := hband _ B hBs
    refine ⟨?_, ?_, hBs, hnot⟩
    · -- the family's members are suffix draws, all of length `Ls`
      intro v hv v' hv' hsuf
      have hlen : ∀ w ∈ clusterAt O.mq (loopPopulations K H.length) (clusterRun H c) B,
          w.toList.length = Ls := by
        intro w hw
        unfold clusterAt at hw
        obtain ⟨hw1, hw⟩ := Finset.mem_erase.1 hw
        have := clusterAround_subset _ 1 2 _ _ _ _
          (one_mem_screenedAt O (loopPopulations K H.length) B (clusterRun H c)) hw
        have := screenedAt_subset O (loopPopulations K H.length) B (clusterRun H c) this
        simp only [poolAt, Finset.mem_insert, Finset.mem_image, Finset.mem_range] at this
        rcases this with h | ⟨i, -, rfl⟩
        · exact absurd h hw1
        · exact hlong i
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlen v hv).trans (hlen v' hv').symm)
    · have h1 := clusterAt_card_add_one (rule := lloydClusterer) O (loopPopulations K H.length) B
        (clusterRun H c)
      have h2 := lloydClusterer.card_le (fun w => O.mq w (oracleNoise (clusterRun H c)) = 1)
        (prefixesAt (loopPopulations K H.length) B.npref (clusterRun H c))
        (screenedAt O.mq (loopPopulations K H.length) B (clusterRun H c)) B.k (by omega)
      rw [clusterAt_eq]
      unfold clusterFull at h1
      omega
  calc (loopMeasure μ P.D Dsf K T).real _
      ≤ (loopMeasure μ P.D Dsf K T).real ({x | (id (x rK)).2 ∈ Bad (stage x)} ∪ N') :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (loopMeasure μ P.D Dsf K T).real {x | (id (x rK)).2 ∈ Bad (stage x)}
        + (loopMeasure μ P.D Dsf K T).real N' := measureReal_union_le _ _
    _ ≤ δ + 0 := by
        refine add_le_add (ENNReal.toReal_le_of_le_ofReal hδ.le hmain) ?_
        rw [Measure.real, hN', ENNReal.toReal_zero]
    _ = δ := add_zero _

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


lemma le_prefixMax_of_spread {D : Measure (FreeMonoid α)} [IsProbabilityMeasure D] {k : ℕ}
    {ν : Measure (FreeMonoid α)} (h : Spread D k ν) (t : FreeMonoid α) :
    ν.real {t} ≤ prefixMax D k := by
  refine (h t).trans (le_of_eq_of_le ?_ (le_prefixMax D k (prefixOf t k)))
  congr 1

/-- A population puts at most `q / y` on any string where its sampler finds a string at most `q`
of the time and yields at least `y`. -/
lemma collisionMass_popLaw_le {ν : Measure (FreeMonoid α)} {q y : ℝ} (hy : 0 < y)
    (hyν : y ≤ ν.real Set.univ) (htop : ν Set.univ ≠ ∞) (hq : ∀ t, ν.real {t} ≤ q) :
    collisionMass (popLaw ν) ≤ q / y := by
  have hpos : 0 < ν.real Set.univ := lt_of_lt_of_le hy hyν
  have h0 : ν Set.univ ≠ 0 := fun h => by
    rw [measureReal_def, h, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
  have hq0 : 0 ≤ q := measureReal_nonneg.trans (hq 1)
  have hat : ∀ t, (popLaw ν).real {t} ≤ q / y := by
    intro t
    rw [popLaw_real h0 htop]
    calc ν.real {t} / ν.real Set.univ ≤ q / ν.real Set.univ :=
          div_le_div_of_nonneg_right (hq t) hpos.le
      _ ≤ q / y := div_le_div_of_nonneg_left hq0 hy hyν
  unfold collisionMass
  refine tsum_le_of_sum_le' (div_nonneg hq0 hy.le) fun s => ?_
  calc ∑ t ∈ s, (popLaw ν).real {t} ^ 2 ≤ ∑ t ∈ s, q / y * (popLaw ν).real {t} :=
        Finset.sum_le_sum fun t _ => by
          rw [sq]; exact mul_le_mul_of_nonneg_right (hat t) measureReal_nonneg
    _ = q / y * (popLaw ν).real s := by rw [← Finset.mul_sum, sum_measureReal_singleton]
    _ ≤ q / y * 1 := mul_le_mul_of_nonneg_left measureReal_le_one (div_nonneg hq0 hy.le)
    _ = q / y := mul_one _

/-- The loop's claim at a collision cap `cap` and walk start `k`: `cap` within what the
clustering's proof admits of every set of populations, and the probes' mass on one length-`k`
prefix small against `cap` and level 4's `ρ`. -/
def LoopCore : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (K S Ls : ℕ) (η₀ indecisionLimit gateLevel εcov δc slack pAP εd δ cap : ℝ) (k : ℕ),
  let n : ℝ := Fintype.card σ + S
  let A : ℝ := Fintype.card α
  let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
  let Kp : ℝ := (n + 1) * (2 * (n + 3) * A + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
    + 8 * (n + 3) * A * (P.L + 1)
  let X : ℝ := 4 * (2 ^ (Fintype.card σ + S + 5) * A + 7 * Kp ^ 2) / δ
  let G : ℝ := Real.log X + 2 * Real.log (2 * Real.log X)
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < indecisionLimit → indecisionLimit ≤ 1 / 2 →
  0 < gateLevel → gateLevel < 1 / 2 →
  0 < εcov → εcov ≤ 1 →
  0 < δc → δc ≤ 1 →
  0 < slack → 0 < pAP →
  0 < εd → εd ≤ 1 → 0 < δ → δ ≤ 1 → 1 ≤ P.L →
  Fintype.card σ + 1 ≤ K →
  P.L + Fintype.card σ + S + 3 ≤ Ls →
  512 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * indecisionLimit + slack) ≤ 1 →
  0 < cap →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    cap ≤ qualityCap η₀ pops indecisionLimit εcov gateLevel δc pAP
      (Real.exp (-2 / (ρ / (2 * (P.L + 1) * (n + 1))))) slack) →
  prefixMax P.D k ≤ cap * ρ →
  6 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ →
  ∃ settings : LoopSettings K,
    settings.indecisionLimit = indecisionLimit ∧ settings.gateLevel = gateLevel ∧
    settings.level4.walkStart = k ∧ settings.level4.disagreementRate = εd ∧
    (settings.level4.probeBudget : ℝ) ≤ Kp * G ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ Ls} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      collisionMass P.D ≤ cap → collisionMass Dsf ≤ cap →
      (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
          {draws | let rounds := indecisiveLStarLoop settings O P.D draws
            ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                o.Agrees O P.D k (εcov + 2 * indecisionLimit + 2 * slack)
                  (2 * indecisionLimit + slack) εd)}
        ≤ (Fintype.card σ + 1) * (δc + δ)

theorem loop_core_holds : LoopCore := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K S Ls η₀ ι gate εcov δc slack pAP εd δ cap k n A ρ Kp X G
    hL hη hη₀ hι hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ hcap hcapq
    hk1 hk2
  classical
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hρ0 : 0 < ρ := by positivity
  obtain ⟨cross, hcrossd⟩ : ∃ x : ℝ, x = ρ / (2 * (P.L + 1) * (n + 1)) := ⟨_, rfl⟩
  have hcr : 0 < cross := by rw [hcrossd]; positivity
  obtain ⟨schedule, hband, hclust⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc slack pAP hη hη₀ (flat_of_length Ls) hpAP hι hι1 hg hg1 hcr hε hε1
    hδc hδc1 hsl cap (fun pops hp => by rw [hcrossd]; exact hcapq pops hp)
  have hΛ : 0 < unionLog (α := α) σ S δ := by
    unfold unionLog
    refine mul_pos (by positivity) (Real.log_pos ?_)
    rw [one_lt_div hδ]
    have : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by linarith [(Nat.cast_nonneg S : (0 : ℝ) ≤ S),
      (Nat.cast_nonneg (Fintype.card σ) : (0 : ℝ) ≤ Fintype.card σ)]
    nlinarith only [this, hA1, hδ1, hδ]
  obtain ⟨s4, hwalk, hdis, hbud, hmax, hl4⟩ :=
    level4_guarantee_uniform (μ := μ) M εd P δ S k hδ hδ1 hεd hεd1 hL1
  set settings : LoopSettings K := ⟨schedule, ι, gate, s4⟩ with hset
  refine ⟨settings, rfl, rfl, hwalk, hdis, hbud, ?_⟩
  intro Dsf hDsf hDsfLen hpAPDsf hcolD hcolSf
  set N := P.L + s4.maxLeaves
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
      rw [hcrossd]; simp only [n]; field_simp
    have e2 : 3 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ / 2 := by linarith
    have e3 : (S + 1) * εd / (256 * (Fintype.card σ + S + 1)
        * (Fintype.card σ * Fintype.card α + S + 1)) = ρ := by
      simp only [ρ, n, A]
    rw [e1, e3]
    linarith
  set E1 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O P.D draws K).length,
      (∀ ν ∈ harvestsOf P.D settings.level4.walkStart
          ((loopRounds settings O P.D draws K).take r), collisionMass ν ≤ cap)
      ∧ ¬ ClusterSpec settings O P.D εcov slack
        (harvestsOf P.D settings.level4.walkStart ((loopRounds settings O P.D draws K).take r))
        ((loopRounds settings O P.D draws K)[r]'hr).run} with hE1
  set E2 : ℕ → Set (Fin K → LoopRoundDraws Ω α K s4.probeBudget) := fun r =>
    {draws | ∃ hr : r < (loopRounds settings O P.D draws K).length,
      ClusterSpec settings O P.D εcov slack
          (harvestsOf P.D k ((loopRounds settings O P.D draws K).take r))
          ((loopRounds settings O P.D draws K)[r]'hr).run
        ∧ ¬ Level4Spec M O P (loopTheta σ ι slack) εd ρ k
          ((loopRounds settings O P.D draws K)[r]'hr)} with hE2
  have h1 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E1 r) ≤ δc := fun r =>
    loop_cluster_round_le (σ := σ) O P Dsf settings S Ls N εcov δc slack cap hδc.le hmax hLs
      le_rfl hcolD (fun pops hpops D hD hDlen hDcol =>
        hclust pops hpops D Dsf hD hDsf (fun j hj => by
          refine measure_mono_null (fun p hp => ?_) (hDlen j hj)
          simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_lt] at hp ⊢
          exact hp)
          (by
            refine measure_mono_null (fun v hv => ?_) hDsfLen
            simp only [Set.mem_compl_iff, Set.mem_ofPred_eq] at hv ⊢
            exact hv) hpAPDsf hDcol hcolSf) r
  have h2 : ∀ r, (loopMeasure μ P.D Dsf K s4.probeBudget).real (E2 r) ≤ δ := fun r =>
    loop_level4_round_le M O hL P Dsf settings S k Ls N εcov slack εd δ cross ρ hmax le_rfl
      hDsfLen (by unfold Oracle.η at hη ⊢; linarith) hband hwalk hdis hδ hεd hcr hθ' hθ0 hcross
      (fun O' θ => hl4 O' θ) r
  have hsub : {draws : Fin K → LoopRoundDraws Ω α K s4.probeBudget |
      let rounds := indecisiveLStarLoop settings O P.D draws
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.Agrees O P.D k (εcov + 2 * ι + 2 * slack) (2 * ι + slack) εd)}
      ⊆ ⋃ r ∈ Finset.range (Fintype.card σ + 1), (E1 r ∪ E2 r) := by
    intro draws hd
    by_contra hno
    simp only [Set.mem_iUnion, Set.mem_union, Finset.mem_range, not_exists, not_or] at hno
    set L := loopRounds settings O P.D draws K with hLdef
    have hspec : ∀ i, i ≤ Fintype.card σ → ∀ hi : i < L.length,
        ClusterSpec settings O P.D εcov slack (harvestsOf P.D k (L.take i)) L[i].run
          ∧ Level4Spec M O P (loopTheta σ ι slack) εd ρ k L[i] := by
      intro i
      induction i using Nat.strong_induction_on with
      | _ i ih =>
        intro hiσ hi
        have hcol : ∀ ν ∈ harvestsOf P.D k (L.take i), collisionMass ν ≤ cap := by
          intro ν hν
          obtain ⟨o, ho, hoν⟩ := List.mem_filterMap.1 hν
          obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
          simp only [List.length_take] at hj
          rw [List.getElem_take] at hoν
          have hji : j < i := by omega
          have h4 := (ih j hji (by omega) (by omega)).2
          unfold LoopRound.population at hoν
          rcases hh : (L[j]'(by omega)).result.ending.harvested with _ | h
          · rw [hh] at hoν; cases hoν
          rw [hh, Option.map_some, Option.some.injEq] at hoν
          subst hoν
          have he : (L[j]'(by omega)).result.ending = .harvest h := by
            revert hh
            cases (L[j]'(by omega)).result.ending <;> simp [RoundEnd.harvested]
          simp only [Level4Spec] at h4
          rw [he] at h4
          obtain ⟨-, hsp, hyl⟩ := h4
          refine (collisionMass_popLaw_le hρ0 hyl
            (harvestLaw_univ_ne_top P.D _ k P.length_le h)
            (le_prefixMax_of_spread hsp)).trans ?_
          rw [div_le_iff₀ hρ0]
          exact hk1
        have hc : ClusterSpec settings O P.D εcov slack (harvestsOf P.D k (L.take i)) L[i].run := by
          by_contra hc
          refine (hno i (by omega)).1 ⟨hi, ?_, ?_⟩
          · rw [hwalk]; exact hcol
          · rw [hwalk]; exact hc
        refine ⟨hc, ?_⟩
        by_contra h4
        exact (hno i (by omega)).2 ⟨hi, hc, h4⟩
    exact hd (loop_agrees_of_specs M O hL P settings draws k hwalk εcov slack εd ρ hK
      (fun pops B hB => (hband pops B hB).1) fun i hi hiσ => hspec i hiσ hi)
  have : IsProbabilityMeasure (loopMeasure μ P.D Dsf K s4.probeBudget) := by
    unfold loopMeasure clusterLaw
    infer_instance
  calc (loopMeasure μ P.D Dsf K s4.probeBudget).real _
      ≤ (loopMeasure μ P.D Dsf K s4.probeBudget).real
          (⋃ r ∈ Finset.range (Fintype.card σ + 1), (E1 r ∪ E2 r)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (Fintype.card σ + 1),
          (loopMeasure μ P.D Dsf K s4.probeBudget).real (E1 r ∪ E2 r) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (Fintype.card σ + 1), (δc + δ) :=
        Finset.sum_le_sum fun r _ => (measureReal_union_le _ _).trans (add_le_add (h1 r) (h2 r))
    _ = (Fintype.card σ + 1) * (δc + δ) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

theorem learner_loop_correct_holds : LearnerLoopCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K S Ls η₀ ι gate εcov δc slack pAP εd δ n A ρ Kp X G
    hL hη hη₀ hι hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ
  classical
  have hρ0 : 0 < ρ := by positivity
  obtain ⟨c, hc⟩ : ∃ x : ℝ, x = Real.exp (-2 / (ρ / (2 * (P.L + 1) * (n + 1)))) := ⟨_, rfl⟩
  have hc0 : 0 < c := hc ▸ Real.exp_pos _
  -- one cap within every set of populations' own
  set capAt : Finset (Option (Fin K)) → ℝ := fun pops =>
    if none ∈ pops then qualityCap η₀ pops ι εcov gate δc pAP c slack else 1 with hcapAt
  have hcapAt0 : ∀ pops, 0 < capAt pops := by
    intro pops
    simp only [hcapAt]
    split_ifs with h
    · exact (clustering_quality_bad O pops none {p : FreeMonoid α | p.toList.length < Ls}
        {v : FreeMonoid α | v.toList.length = Ls} η₀ ι εcov gate δc pAP c slack hη hη₀ h
        (flat_of_length Ls) hpAP hι hι1 hg hg1 hε hε1 hδc hδc1 hc0 hsl).1
    · norm_num
  obtain ⟨cap, hcapd⟩ : ∃ cap : ℝ,
      cap = (Finset.univ : Finset (Finset (Option (Fin K)))).inf' ⟨∅, Finset.mem_univ _⟩ capAt :=
    ⟨_, rfl⟩
  have hcap0 : 0 < cap := by
    rw [hcapd]; exact (Finset.lt_inf'_iff _).2 fun pops _ => hcapAt0 pops
  have hcapq : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      cap ≤ qualityCap η₀ pops ι εcov gate δc pAP c slack := by
    intro pops h
    have := Finset.inf'_le capAt (Finset.mem_univ pops)
    simp only [hcapAt, if_pos h] at this
    rw [hcapd]; exact this
  refine ⟨cap, hcap0, fun k hk1 hk2 => ?_⟩
  exact loop_core_holds M O P K S Ls η₀ ι gate εcov δc slack pAP εd δ cap k hL hη hη₀ hι hι1
    hg hg1 hε hε1 hδc hδc1 hsl hpAP hεd hεd1 hδ hδ1 hL1 hK hLs hθ hcap0
    (fun pops h => by rw [← hc]; exact hcapq pops h) hk1 hk2

end OrthoDFA

end
