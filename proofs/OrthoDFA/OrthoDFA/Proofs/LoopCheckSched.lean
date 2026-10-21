import OrthoDFA.Proofs.LearnerLoop

/-!
# The clustering's schedule, holding the uniform pool alone tame

`clustering_schedule_exists` with only the uniform pool tame: the fresh-noise bound on what the
family leaves undecided is on the uniform pool alone, and asks the collision caps of it alone.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `clustering_schedule_exists`, holding only the uniform pool tame. -/
theorem clustering_schedule_uni {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (K : ℕ) (Pre Suf : Set (FreeMonoid α))
    (η₀ indecisionLimit gateLevel crossLimit εcov δc slack pAP : ℝ)
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hflat : Flat Pre Suf) (hpAP : 0 < pAP)
    (hι : 0 < indecisionLimit) (hι1 : indecisionLimit ≤ 1 / 2) (hg : 0 < gateLevel)
    (hg1 : gateLevel < 1 / 2) (hcross : 0 < crossLimit) (hε : 0 < εcov) (hε1 : εcov ≤ 1 / 2)
    (hδc : 0 < δc) (hδc1 : δc ≤ 1) (hslack : 0 < slack) :
    ∃ schedule : Finset (Option (Fin K)) → Finset State,
      (∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k → ∀ p r ε₂ κ : ℝ,
        0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
      ∧ (∀ pops, ∀ B ∈ schedule pops,
        B.nsuff = poolCount η₀ pops indecisionLimit εcov (δc / 2) pAP (crossLimit ^ 4)
        ∧ B.npref ≤ prefCount η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP
          (crossLimit ^ 4) 0)
      ∧ (∀ pops, ∑ B ∈ schedule pops, B.npref
        ≤ 2 * prefCount η₀ pops indecisionLimit εcov (δc / 2) gateLevel pAP
          (crossLimit ^ 4) 0)
      ∧ (∀ pops : Finset (Option (Fin K)), none ∈ pops →
        ∀ (D : Option (Fin K) → Measure (FreeMonoid α)) (Dsf : Measure (FreeMonoid α)),
          (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
          (∀ j ∈ pops, D j Preᶜ = 0) → Dsf Sufᶜ = 0 →
          pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
          (∀ B ∈ schedule pops, ∀ t,
            2 ^ 21 * ((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (D none).real {t}
              ≤ δc * rateScale η₀ indecisionLimit εcov ^ 2) →
          (∀ B ∈ schedule pops, ∀ t,
            2 ^ 8 * ((pops.card : ℝ) + 1) * (B.npref : ℝ) * (D none).real {t}
              ≤ δc * slack ^ 2) →
          (∀ B ∈ schedule pops,
            2 ^ 13 * ((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Dsf ≤ δc * pAP) →
          (runMeasure μ D Dsf).real
            {x | ¬ ((∃ B ∈ schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B)
              ∧ ∀ B ∈ schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B →
                ∫ p, miscutProb O B.lo (B.hi + 1)
                    (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p ∂(D none)
                    ≤ 513 / 512 * εcov + slack
                  ∧ ∫ p, undecidedProb O B.lo (B.hi + 1)
                    (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                    ∂(D none) ≤ 2 * indecisionLimit + slack)} ≤ δc + gateLevel)
      ∧ ∀ pops : Finset (Option (Fin K)), none ∈ pops →
        ∀ (D : Option (Fin K) → Measure (FreeMonoid α)) (Dsf : Measure (FreeMonoid α)),
          (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
          (∀ j ∈ pops, D j Preᶜ = 0) → Dsf Sufᶜ = 0 →
          pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
          (∀ B ∈ schedule pops, ∀ t,
            2 ^ 21 * ((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (D none).real {t}
              ≤ δc * rateScale η₀ indecisionLimit εcov ^ 2) →
          (∀ B ∈ schedule pops,
            2 ^ 13 * ((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Dsf ≤ δc * pAP) →
          (runMeasure μ D Dsf).real
            {x | ¬ UniSpec O pops (schedule pops) η₀ indecisionLimit εcov gateLevel (D none)
              (streams (fun _ => ∅) none x)} ≤ δc + gateLevel := by
  classical
  obtain ⟨c, hc⟩ : ∃ c : ℝ, c = crossLimit ^ 4 := ⟨_, rfl⟩
  have hc0 : 0 < c := hc ▸ pow_pos hcross 4
  rw [← hc]
  have hH : (allHarvest (fun _ : Option (Fin K) => (∅ : Finset (FreeMonoid α)))).card = 0 :=
    Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun s hs => by
      simp [allHarvest] at hs)
  have hq : ∀ pops : Finset (Option (Fin K)), none ∈ pops → _ := fun pops hpops =>
    clustering_quality_sized O pops {none} none (fun _ => ∅) Pre Suf η₀ indecisionLimit εcov
      gateLevel δc pAP c slack hη hη₀ (Finset.mem_singleton_self _)
      (Finset.singleton_subset_iff.2 hpops) hflat
      (fun _ p hp => absurd hp (Finset.notMem_empty p)) hpAP hι hι1 hg hg1 hε hε1 hδc hδc1 hc0
      hslack
  refine ⟨fun pops => if h : none ∈ pops then Classical.choose (hq pops h) else ∅, ?_, ?_, ?_, ?_,
    ?_⟩
  · intro pops B hB
    dsimp only at hB
    split_ifs at hB with hp
    swap
    · simp at hB
    have hs := (Classical.choose_spec (hq pops hp)).2.1 hB
    have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
    obtain ⟨hBk, hBlo, hBhi⟩ := band_of_mem_schedule hs
    obtain ⟨-, hhalf, hb1⟩ := famCount_cross (indecisionLimit := indecisionLimit) (crossLimit := c)
      η₀ pops εcov (δc / 2) hη0 hη₀ hc0
    have hev := famCount_even (indecisionLimit := indecisionLimit) (crossLimit := c)
      η₀ pops εcov (δc / 2)
    have h4 := famCount_ge_four (indecisionLimit := indecisionLimit) (crossLimit := c)
      η₀ pops εcov (δc / 2) hη₀
    rw [bandHalf] at hb1
    set κc : ℕ := famCount η₀ pops indecisionLimit εcov (δc / 2) c with hκc
    obtain ⟨hlo, hhi⟩ := loAt_hiAt η₀ hev h4 hη0 hη₀
    have hb1R : (1 : ℝ) ≤ bandAt η₀ κc := by exact_mod_cast hb1
    have hsum : loAt η₀ κc + hiAt η₀ κc + 1 = κc := by
      have : ((loAt η₀ κc + hiAt η₀ κc + 1 : ℕ) : ℝ) = κc := by push_cast; rw [hlo, hhi]; ring
      exact_mod_cast this
    have hlo3 : loAt η₀ κc + 3 ≤ hiAt η₀ κc := by
      have : ((loAt η₀ κc + 3 : ℕ) : ℝ) ≤ hiAt η₀ κc := by push_cast; rw [hlo, hhi]; linarith
      exact_mod_cast this
    have hhalf' : halfTail η₀ κc ≤ crossLimit :=
      (pow_le_pow_iff_left₀ (halfTail_nonneg η₀ κc) hcross.le (by norm_num)).1 (hhalf.trans_eq hc)
    refine ⟨lohi_of_mem_schedule hs, by rw [hBk]; omega, fun N hN p r ε₂ κ hp0 hp1 hr0 hr1 _ => ?_⟩
    rw [hBlo, hBhi]
    exact bandPasses_of_halfTail hp0 hp1 hr0 hr1 (by omega) hsum hlo3 hhalf'
  · intro pops B hB
    dsimp only at hB
    split_ifs at hB with hp
    swap
    · simp at hB
    have hs := (Classical.choose_spec (hq pops hp)).2.1 hB
    rw [hH] at hs
    refine ⟨nsuff_of_mem_schedule hs, ?_⟩
    obtain ⟨i, -, hi⟩ := Finset.mem_image.1 hs
    rw [← hi]
    exact Nat.div_le_self _ _
  · intro pops
    dsimp only
    split_ifs with hp
    · have hsub : Classical.choose (hq pops hp) ⊆ schedule η₀ pops indecisionLimit εcov (δc / 2)
          gateLevel pAP c 0 := fun B hB => by
        have h := (Classical.choose_spec (hq pops hp)).2.1 hB
        rwa [hH] at h
      have h := le_trans (Finset.sum_le_sum_of_subset_of_nonneg hsub
        (fun _ _ _ => Nat.cast_nonneg _))
        (schedule_npref_sum_le η₀ pops εcov (δc / 2) gateLevel pAP)
      exact_mod_cast h
    · simp
  · intro pops hpops D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolD hslD hcolSf
    simp only [dif_pos hpops] at hcolD hslD hcolSf ⊢
    set states := Classical.choose (hq pops hpops)
    obtain ⟨-, -, -, hgood⟩ := Classical.choose_spec (hq pops hpops)
    have hg := hgood D Dsf hD hDsf hsupp hsuppSf hpAPDsf
      (fun B hB j hj t => by rw [Finset.mem_singleton.1 hj]; exact hcolD B hB t)
      (fun B hB j hj t => by rw [Finset.mem_singleton.1 hj]; exact hslD B hB t) hcolSf
    set SpecR := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      (∃ B ∈ states, x ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B)
        ∧ ∀ B ∈ states, x ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B →
          ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
              ≤ 513 / 512 * εcov + slack
            ∧ ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
              ∂(D none) ≤ 2 * indecisionLimit + slack}
    have hSm : MeasurableSet (streams (Ω := Ω) (fun _ => ∅) none ⁻¹' SpecR) :=
      measurable_streams _ none (measurableSet_specAt O pops states η₀ indecisionLimit εcov
        gateLevel (fun B F => ∫ p, miscutProb O B.lo (B.hi + 1) F p ∂(D none)
          ≤ 513 / 512 * εcov + slack
          ∧ ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(D none)
            ≤ 2 * indecisionLimit + slack))
    have hsub : {x : Draws Ω (FreeMonoid α) (Option (Fin K)) |
        (∃ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
          ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B.val)
          ∧ ∀ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
            ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B.val →
            ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                ∂(D none) ≤ 513 / 512 * εcov + slack
            ∧ ∀ j ∈ ({none} : Finset (Option (Fin K))),
              ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                  ∂(D j) ≤ 1 / 2 + εcov / 512 + slack
              ∧ ∫ p, undecidedProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                  ∂(D j) ≤ 2 * indecisionLimit + slack}
        ⊆ streams (fun _ => ∅) none ⁻¹' SpecR := by
      rintro x ⟨⟨B, hret⟩, hall⟩
      exact ⟨⟨B.val, B.property, hret⟩, fun B' hB' hret' =>
        ⟨(hall ⟨B', hB'⟩ hret').1, ((hall ⟨B', hB'⟩ hret').2 none (Finset.mem_singleton_self _)).2⟩⟩
    have hmono := measureReal_mono (μ := runMeasure μ D Dsf) hsub (measure_ne_top _ _)
    have hcompl : (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹' SpecR)ᶜ
        = 1 - (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹' SpecR) :=
      probReal_compl_eq_one_sub hSm
    show (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹' SpecR)ᶜ ≤ δc + gateLevel
    linarith
  · intro pops hpops D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolU hcolSf
    simp only [dif_pos hpops] at hcolU hcolSf ⊢
    set states := Classical.choose (hq pops hpops)
    obtain ⟨-, -, hgood, -⟩ := Classical.choose_spec (hq pops hpops)
    have hg := hgood D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolU hcolSf
    have : IsProbabilityMeasure (D none) := hD none
    have hSm : MeasurableSet (streams (Ω := Ω) (fun _ => ∅) none ⁻¹'
        {x | UniSpec O pops states η₀ indecisionLimit εcov gateLevel (D none) x}) :=
      measurable_streams _ none (measurableSet_uniSpec O pops states η₀ indecisionLimit εcov
        gateLevel (D none))
    have hsub : {x : Draws Ω (FreeMonoid α) (Option (Fin K)) |
        (∃ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
          ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B.val)
          ∧ ∀ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
            ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B.val →
            (D none).real {p | ¬ cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p (oracleNoise x.1)}
              ≤ 513 / 512 * εcov
            ∧ (D none).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p (oracleNoise x.1)}
              ≤ 2 * indecisionLimit}
        ⊆ streams (fun _ => ∅) none ⁻¹'
          {x | UniSpec O pops states η₀ indecisionLimit εcov gateLevel (D none) x} := by
      rintro x ⟨⟨B, hret⟩, hall⟩
      exact ⟨⟨B.val, B.property, hret⟩, fun B' hB' hret' => hall ⟨B', hB'⟩ hret'⟩
    have hmono := measureReal_mono (μ := runMeasure μ D Dsf) hsub (measure_ne_top _ _)
    have hcompl : (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹'
          {x | UniSpec O pops states η₀ indecisionLimit εcov gateLevel (D none) x})ᶜ
        = 1 - (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹'
          {x | UniSpec O pops states η₀ indecisionLimit εcov gateLevel (D none) x}) :=
      probReal_compl_eq_one_sub hSm
    show (runMeasure μ D Dsf).real (streams (fun _ => ∅) none ⁻¹'
      {x | UniSpec O pops states η₀ indecisionLimit εcov gateLevel (D none) x})ᶜ ≤ δc + gateLevel
    linarith

end OrthoDFA

end
