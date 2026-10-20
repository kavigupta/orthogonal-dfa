import OrthoDFA.Proofs.LoopColl
import OrthoDFA.Proofs.LoopUniSpec
import OrthoDFA.Proofs.ClusteringQuality
import Mathlib.Data.List.GetD

/-!
# The loop's theorem

The whole run reads one noise. Each round draws off every prefix and suffix the earlier rounds
read, so no round reads a string an earlier round read but with chance zero (`loop_coll_null`).
Round `r` fails the clustering's claim, given the earlier rounds, but for chance `δc`
(`loop_cluster_round_le_persist`): off what the earlier rounds read the clustering meets its
claim at `slack - m`, and a population's average moves by at most the `m` the earlier rounds'
prefixes hold of it. It fails level 5's but for chance `δ` (`loop_round_le_persist`). A
round meeting level 5's claim hands on a population putting at most `prefixMax / (ρ/4)` on any
string, within the cap. Where every one of the first `|Q| + 1` rounds meets both claims, the loop
stops in agreement (`loop_agrees_of_specs`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## The clustering's schedule -/

/-- One schedule serves every set of the loop's populations: every state of it reads a family of
at most `k - 1` members at a band whose far side `crossLimit` bounds, whatever the read's two
rates, draws the one pool and at most the top prefix count, and the prefix counts sum to at most
twice the top's. Nothing is harvested into a table, and every population is tame. -/
theorem clustering_schedule_exists {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
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
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            2 ^ 21 * ((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (D j).real {t}
              ≤ δc * rateScale η₀ indecisionLimit εcov ^ 2) →
          (∀ B ∈ schedule pops, ∀ j ∈ pops, ∀ t,
            2 ^ 8 * ((pops.card : ℝ) + 1) * (B.npref : ℝ) * (D j).real {t}
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
                  ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1)
                    (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                    ∂(D j) ≤ 2 * indecisionLimit + slack)} ≤ δc + gateLevel)
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
    clustering_quality_sized O pops pops none (fun _ => ∅) Pre Suf η₀ indecisionLimit εcov
      gateLevel δc pAP c slack hη hη₀ hpops subset_rfl hflat
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
    obtain ⟨hlohi, hk2, hband⟩ := band_of_mem_schedule (le_trans (eta_nonneg O) hη) hη₀ hs
    refine ⟨hlohi, hk2, fun N hN p r ε₂ κ hp0 hp1 hr0 hr1 _ =>
      bandPasses_of_wide hp0 hp1 hr0 hr1 hlohi hcross ?_⟩
    have hlog : Real.log (1 / c) = 4 * Real.log (1 / crossLimit) := by
      rw [hc, one_div, Real.log_inv, Real.log_pow, one_div, Real.log_inv]; push_cast; ring
    rw [hlog] at hband
    have hNk : (N : ℝ) + 1 ≤ B.k := by exact_mod_cast hN
    have hlh : (B.lo : ℝ) ≤ B.hi := by exact_mod_cast hlohi
    have h2 : ((B.hi : ℝ) - B.lo) ^ 2 ≤ ((B.hi : ℝ) + 1 - B.lo) ^ 2 := by nlinarith
    have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
    rcases le_or_gt 0 (Real.log (1 / crossLimit)) with hl | hl
    · nlinarith [mul_le_mul_of_nonneg_right hNk hl]
    · nlinarith [mul_nonneg hN0 (le_of_lt (neg_pos.2 hl))]
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
    have hg := hgood D Dsf hD hDsf hsupp hsuppSf hpAPDsf hcolD hslD hcolSf
    set SpecR := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      (∃ B ∈ states, x ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B)
        ∧ ∀ B ∈ states, x ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B →
          ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
              ≤ 513 / 512 * εcov + slack
            ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
              ∂(D j) ≤ 2 * indecisionLimit + slack}
    have hSm : MeasurableSet (streams (Ω := Ω) (fun _ => ∅) none ⁻¹' SpecR) :=
      measurable_streams _ none (measurableSet_specAt O pops states η₀ indecisionLimit εcov
        gateLevel (fun B F => ∫ p, miscutProb O B.lo (B.hi + 1) F p ∂(D none)
          ≤ 513 / 512 * εcov + slack
          ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(D j)
            ≤ 2 * indecisionLimit + slack))
    have hsub : {x : Draws Ω (FreeMonoid α) (Option (Fin K)) |
        (∃ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
          ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B.val)
          ∧ ∀ B : {B : State // B ∈ states}, streams (fun _ => ∅) none x
            ∈ ret O.mq pops none η₀ indecisionLimit εcov gateLevel B.val →
            ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                ∂(D none) ≤ 513 / 512 * εcov + slack
            ∧ ∀ j ∈ pops,
              ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                  ∂(D j) ≤ 1 / 2 + εcov / 512 + slack
              ∧ ∫ p, undecidedProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B.val) p
                  ∂(D j) ≤ 2 * indecisionLimit + slack}
        ⊆ streams (fun _ => ∅) none ⁻¹' SpecR := by
      rintro x ⟨⟨B, hret⟩, hall⟩
      exact ⟨⟨B.val, B.property, hret⟩, fun B' hB' hret' =>
        ⟨(hall ⟨B', hB'⟩ hret').1, fun j hj => ((hall ⟨B', hB'⟩ hret').2 j hj).2⟩⟩
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

/-! ## The theorem -/

section Pops

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}

omit [MeasurableSpace Ω] in
/-- A round whose tree is in `classSet |Q|` hands on a population supported on strings shorter
than `Ls`. -/
lemma population_short {σ : Type*} [Fintype σ] (P : PrefixDistribution α) {Ls : ℕ}
    (hLs : P.L + Fintype.card σ < Ls) (k : ℕ) (o : LoopRound Ω α K)
    (hcls : o.result.hypothesis.tree ∈ Random.classSet (Fintype.card σ))
    {ν : Measure (FreeMonoid α)} (hν : o.population P.D k = some ν) :
    ν {p | Ls ≤ p.toList.length} = 0 := by
  unfold LoopRound.population at hν
  rcases hh : o.result.ending.harvested with _ | site
  · rw [hh] at hν; cases hν
  rw [hh, Option.map_some, Option.some.injEq] at hν
  subst hν
  have h0 : Random.harvestLaw o.read P.D k o.result.hypothesis site {p | Ls ≤ p.toList.length}
      = 0 :=
    measure_mono_null (fun p hp => by simp only [Set.mem_ofPred_eq] at hp ⊢; omega)
      (Random.harvestLaw_short o.read P.length_le k (Random.classSet_props _ _ hcls).1 site)
  refine popLaw_null (le_antisymm ((Measure.restrict_apply_le _ _).trans h0.le) bot_le) ?_
  · simp only [Set.mem_ofPred_eq, FreeMonoid.toList_one, List.length_nil, not_le]
    omega

omit [MeasurableSpace Ω] in
/-- A round whose tree is in `classSet |Q|` and whose suffixes are longer than `L + |Q|` finds no
string its population drops. -/
lemma harvestLaw_shorter {σ : Type*} [Fintype σ] (P : PrefixDistribution α) (k : ℕ)
    (o : LoopRound Ω α K) (hcls : o.result.hypothesis.tree ∈ Random.classSet (Fintype.card σ))
    (hs : ∀ v ∈ o.suffs, v ≠ 1 → P.L + Fintype.card σ < v.toList.length) (site : Random.Level3Site) :
    Random.harvestLaw o.read P.D k o.result.hypothesis site (shorterThan o.suffs)ᶜ = 0 := by
  refine measure_mono_null (fun z hz => ?_)
    (Random.harvestLaw_short o.read P.length_le k (Random.classSet_props _ _ hcls).1 site)
  simp only [shorterThan, Set.mem_compl_iff, Set.mem_ofPred_eq, not_forall, not_lt] at hz
  obtain ⟨v, hv, hv1, hle⟩ := hz
  have := hs v hv hv1
  simp only [Set.mem_ofPred_eq]
  omega

end Pops

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


lemma real_singleton_le_prefixMax (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (t : FreeMonoid α) : D.real {t} ≤ prefixMax D k :=
  (measureReal_mono (fun x hx => by rw [Set.mem_singleton_iff.1 hx]; rfl)
    (measure_ne_top _ _)).trans (le_prefixMax D k (prefixOf t k))

lemma popLaw_atom_le {ν : Measure (FreeMonoid α)} {q y : ℝ} (hy : 0 < y)
    (hyν : y ≤ ν.real Set.univ) (htop : ν Set.univ ≠ ∞) (hq : ∀ t, ν.real {t} ≤ q) (t : FreeMonoid α) :
    (popLaw ν).real {t} ≤ q / y := by
  have hpos : 0 < ν.real Set.univ := lt_of_lt_of_le hy hyν
  have h0 : ν Set.univ ≠ 0 := fun h => by
    rw [measureReal_def, h, ENNReal.toReal_zero] at hpos; exact lt_irrefl _ hpos
  have hq0 : 0 ≤ q := measureReal_nonneg.trans (hq 1)
  rw [popLaw_real h0 htop]
  calc ν.real {t} / ν.real Set.univ ≤ q / ν.real Set.univ :=
        div_le_div_of_nonneg_right (hq t) hpos.le
    _ ≤ q / y := div_le_div_of_nonneg_left hq0 hy hyν

section Bridge

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}
  (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  {L N : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
  (hN : L + settings.level3.probeBudget + 2 ≤ N)

omit [IsProbabilityMeasure μ] in
include hlen hN in
/-- Round `i` of the persistent run is the round after the history the rounds before it hand
on. -/
lemma persist_bridge (p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget)) (i : ℕ)
    (hi : i < (loopRounds settings O D Dsf (persistDraws p) K).length) :
    ∃ hK : i < K, (histCodes settings O D Dsf N p.1 p.2 i).length = i
      ∧ decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N p.1 p.2 i)
        = histOf D settings.level3.walkStart
          ((loopRounds settings O D Dsf (persistDraws p) K).take i)
      ∧ (loopRounds settings O D Dsf (persistDraws p) K)[i] = loopRound settings O D Dsf
        (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N p.1 p.2 i))
        (p.1, p.2 ⟨i, hK⟩) := by
  set draws := persistDraws p
  obtain ⟨hK, e⟩ := getElem_loopRounds settings O D Dsf draws K i hi
  have hiK : i ≤ K := (hi.trans_le (length_loopRounds_le settings O D Dsf draws K)).le
  have htake := take_loopRounds settings O D Dsf draws hiK
  have hall : AllHarvested (loopRounds settings O D Dsf draws i) := by
    intro o ho
    rw [← htake] at ho
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
    rw [List.getElem_take]
    simp only [List.length_take] at hj
    exact harvested_of_lt settings O D Dsf draws K j (by omega)
  have hdec := decodeHist_codesOf settings.level3 hlen (Nat.le_trans (Nat.le_add_right _ 2) hN)
    settings.level3.walkStart
    (loopRounds settings O D Dsf draws i) (loopRounds_result settings O D Dsf draws i) hall
  have hc : histCodes settings O D Dsf N p.1 p.2 i
      = codesOf N (loopRounds settings O D Dsf draws i) := rfl
  refine ⟨hK, ?_, ?_, ?_⟩
  · rw [hc, length_codesOf N D settings.level3.walkStart, length_harvestsOf hall,
      length_eq_of_allHarvested settings O D Dsf draws hiK hall]
  · rw [hc, hdec, htake]
  · rw [e, hc, hdec, htake]
    rfl

include hlen hN in
/-- Every earlier clustering returned, where each met its claim. -/
lemma histOK_of_specs (εcov slack : ℝ)
    (p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget)) :
    ∀ i, i ≤ (loopRounds settings O D Dsf (persistDraws p) K).length →
      (∀ j (hj : j < (loopRounds settings O D Dsf (persistDraws p) K).length), j < i →
        ClusterSpec settings O D εcov slack (harvestsOf D settings.level3.walkStart
          ((loopRounds settings O D Dsf (persistDraws p) K).take j))
          (loopRounds settings O D Dsf (persistDraws p) K)[j].run) →
      HistOK settings O D Dsf N p.1 p.2 i
  | 0, _, _ => trivial
  | i + 1, hi, h => by
    refine ⟨histOK_of_specs εcov slack p i (by omega) fun j hj hji => h j hj (by omega),
      fun hK hlen' => ?_⟩
    obtain ⟨hK', -, hdec, e⟩ := persist_bridge settings O D Dsf hlen hN p i (by omega)
    have hs := (h i (by omega) (by omega)).1
    rw [← histOf_pops, ← hdec, e] at hs
    exact hs

end Bridge

lemma card_foldr_union_le {β γ : Type*} [DecidableEq γ] (f : β → Finset γ) (l : List β) (b : ℕ)
    (h : ∀ x ∈ l, (f x).card ≤ b) : (l.foldr (fun c T => f c ∪ T) ∅).card ≤ l.length * b := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.foldr_cons, List.length_cons]
    refine (Finset.card_union_le _ _).trans ?_
    have h1 := h a (by simp)
    have h2 := ih fun x hx => h x (by simp [hx])
    rw [Nat.succ_mul]
    omega

omit [DecidableEq α] in
lemma measureReal_le_card_mul {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] {q : ℝ}
    (T : Finset (FreeMonoid α)) (hq : ∀ t, ν.real {t} ≤ q) : ν.real ↑T ≤ T.card * q := by
  rw [← sum_measureReal_singleton]
  calc ∑ t ∈ T, ν.real {t} ≤ ∑ _t ∈ T, q := Finset.sum_le_sum fun t _ => hq t
    _ = T.card * q := by rw [Finset.sum_const, nsmul_eq_mul]

/-- Every round the loop runs follows rounds whose clusterings returned, returns itself, and
reads no prefix or suffix an earlier round read. -/
def RoundsFresh {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}
    (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
    (N Ls : ℕ) (p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget)) : Prop :=
  ∀ i < (indecisiveLStarLoop settings O D Dsf (persistDraws p)).length, ∃ hK : i < K,
    HistOK settings O D Dsf N p.1 p.2 i
      ∧ Returns settings O D Dsf (decodeHist D settings.level3.walkStart
          (histCodes settings O D Dsf N p.1 p.2 i)) (p.1, (p.2 ⟨i, hK⟩).1)
      ∧ NoColl settings D Dsf Ls (histCodes settings O D Dsf N p.1 p.2 i) p.2 i hK

/-- The loop's claim at a cap `cap` and level 3's settings `C`, `NS` and `NP` bounding the
schedule's suffix pool and prefix count, the clustering run at `slack - m` and `pAP - ms`. The
clustering's band passes at `cross`, small enough that level 5's wrong reads fit level 3's
premises and the probes reading wrong past a midfix weigh more than `u` with chance at most
`C.δ / 8`. The last round has at most `C.disagreementRate + u + mr` of the probes in
`disagreeSet`, `mr` bounding the mass the round's probes avoid. -/
def LoopCore : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (K Ls NS NP : ℕ) (η₀ indecisionLimit gateLevel εcov δc slack pAP cap m ms mr u cross : ℝ)
    (C : Random.Level3Settings),
  let c : ℝ := cross ^ 4
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < indecisionLimit → indecisionLimit ≤ 1 / 2 →
  0 < gateLevel → gateLevel < 1 / 2 →
  0 < εcov → εcov ≤ 1 / 2 →
  0 < δc → δc ≤ 1 →
  0 < slack → 0 < pAP → 1 ≤ P.L →
  Fintype.card σ + 1 ≤ K →
  P.L + Fintype.card σ < Ls →
  0 < cap → m < slack → m ≤ 1 / 2 → ms < pAP → ms ≤ 1 / 2 →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    poolCount η₀ pops indecisionLimit εcov (δc / 2) (pAP - ms) c ≤ NS) →
  (∀ pops : Finset (Option (Fin K)), none ∈ pops →
    prefCount η₀ pops indecisionLimit εcov (δc / 2) gateLevel (pAP - ms) c 0 ≤ NP) →
  2 ^ 22 * ((K : ℝ) + 2) ^ 2 * NP * cap
    ≤ δc * rateScale η₀ indecisionLimit εcov ^ 2 →
  2 ^ 9 * ((K : ℝ) + 2) * NP * cap ≤ δc * (slack - m) ^ 2 →
  2 ^ 14 * ((K : ℝ) + 2) * NS * cap ≤ δc * (pAP - ms) →
  6 * K * (K + 1) * NP * cap ≤ m →
  K * (NS + 1) * cap ≤ ms →
  1 ≤ C.walkStart → prefixMax P.D C.walkStart ≤ cap * C.harvestRate / 2 →
  0 < u → 0 < C.harvestRate →
  6 * (K + 1) * NP * prefixMax P.D C.walkStart ≤ mr → mr ≤ 1 / 2 →
  mr ≤ (1 - mr) * C.harvestRate → 0 < cross →
  RoundPremises σ α P.L C (loopTheta σ indecisionLimit slack)
    (2 * prefixMax P.D C.walkStart) cross →
  ∃ settings : LoopSettings K,
    settings.η₀ = η₀ ∧ settings.indecisionLimit = indecisionLimit ∧ settings.εcov = εcov ∧
    settings.gateLevel = gateLevel ∧ settings.level3 = C ∧ settings.avoidDepth = C.walkStart ∧
    (∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP) ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ Ls} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ v, Dsf.real {v} ≤ cap) →
      (loopMeasure μ α K settings.level3.probeBudget).real
          {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
            ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                o.Checks M O P.D C.walkStart (513 / 512 * εcov + 2 * indecisionLimit + 2 * slack)
                  (2 * indecisionLimit + slack) (C.disagreementRate + u + mr))
              ∧ RoundsFresh settings O P.D Dsf (P.L + settings.level3.probeBudget + 2) Ls p)}
        ≤ (Fintype.card σ + 1) * (δc + gateLevel
          + (C.δ + C.δ / 8 + (shortStrings (α := α) (Fintype.card σ)).card * cross / u))

set_option maxHeartbeats 2000000 in -- the three round bounds and the induction
theorem loop_core_holds : LoopCore := by
  intro α _ _ _ σ _ Ω _ μ _ M O P K Ls NS NP η₀ ι gate εcov δc slack pAP cap m ms mr u cross C
    c hL hη hη₀ hι hι1 hg hg1 hε hε1 hδc hδc1 hsl hpAP hL1 hK hLs hcap hmsl hm1 hmsp
    hms1 hNSb hNPb hq1 hq2 hq3 hqm hqms hk0 hk1 hu hy0 hmr hmr1 hmrc hcr hprem
  classical
  have hδ : 0 < C.δ := hprem.2.1
  have hc : cross ^ 4 = c := rfl
  obtain ⟨schedule₀, hband₀, hsize₀, hsum₀, hclust₀, -⟩ := clustering_schedule_exists O K
    {p : FreeMonoid α | p.toList.length < Ls} {v : FreeMonoid α | v.toList.length = Ls}
    η₀ ι gate cross εcov δc (slack - m) (pAP - ms) hη hη₀ (flat_of_length Ls) (by linarith)
    hι hι1 hg hg1 hcr hε hε1 hδc hδc1 (by linarith)
  rw [hc] at hsize₀ hsum₀
  -- the loop's populations always hold `D`'s
  set schedule : Finset (Option (Fin K)) → Finset State := fun pops =>
    if none ∈ pops then schedule₀ pops else ∅ with hschedd
  have hband : ∀ pops, ∀ B ∈ schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r cross ε₂ κ := fun pops B hB => by
    simp only [hschedd] at hB
    split_ifs at hB
    · exact hband₀ pops B hB
    · simp at hB
  set settings : LoopSettings K := ⟨schedule, η₀, ι, εcov, gate, C, C.walkStart⟩ with hset
  refine ⟨settings, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_⟩
  · intro pops
    simp only [hset, hschedd]
    split_ifs with hp
    · exact (hsum₀ pops).trans (Nat.mul_le_mul_left 2 (hNPb pops hp))
    · simp
  intro Dsf hDsf hDsfLen hpAPDsf hatom
  set k := C.walkStart
  set T := C.probeBudget
  set N := P.L + T + 2
  have hN : P.L + settings.level3.probeBudget + 2 ≤ N := le_rfl
  set dec := decodeHist P.D k with hdecd
  set θ := loopTheta σ ι slack
  set μL := loopMeasure μ α K T
  have hpm0 : 0 ≤ prefixMax P.D k := measureReal_nonneg.trans (le_prefixMax P.D k 1)
  have hy1 : C.harvestRate ≤ 1 := by
    have h1 := hprem.2.2.2.2.2.1
    have h2 := hprem.2.2.2.2.2.2.1
    have : C.harvestRate ≤ 1 / 48 := by linarith
    linarith
  have hcap4 : cap * C.harvestRate / 2 ≤ cap := by
    have : cap * C.harvestRate ≤ cap * 1 := mul_le_mul_of_nonneg_left hy1 hcap.le
    linarith
  -- the schedule's sizes
  have hsizes : ∀ pops, ∀ B ∈ settings.schedule pops, B.nsuff = poolCount η₀ pops ι εcov (δc / 2)
      (pAP - ms) c ∧ B.npref ≤ NP ∧ none ∈ pops := fun pops B hB => by
    simp only [hset, hschedd] at hB
    split_ifs at hB with hp
    · exact ⟨(hsize₀ pops B hB).1, (hsize₀ pops B hB).2.trans (hNPb pops hp), hp⟩
    · simp at hB
  have hNPsum : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP := fun pops => by
    simp only [hset, hschedd]
    split_ifs with hp
    · exact (hsum₀ pops).trans (Nat.mul_le_mul_left 2 (hNPb pops hp))
    · simp
  have hDat : ∀ t, P.D.real {t} ≤ cap := fun t =>
    (real_singleton_le_prefixMax P.D k t).trans (hk1.trans hcap4)
  have hLs' : P.L < Ls := by omega
  have hK1 : (K : ℝ) + 1 ≤ K + 2 := by linarith
  have hpopc : ∀ pops : Finset (Option (Fin K)), ((pops.card : ℕ) : ℝ) + 1 ≤ K + 2 := by
    intro pops
    have : pops.card ≤ K + 1 := by
      have := Finset.card_le_univ pops
      simpa using this
    have : ((pops.card : ℕ) : ℝ) ≤ K + 1 := by exact_mod_cast this
    linarith
  have hη1 : O.η ≤ 1 := by linarith
  have hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, ∀ t, (L j).real {t} ≤ 2 * cap) →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - ms ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * cap) →
        (runMeasure μ L Lsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                  settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
              ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p ∂(L none)
                  ≤ 513 / 512 * εcov + (slack - m)
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                  ∂(L j) ≤ 2 * settings.indecisionLimit + (slack - m))} ≤ δc + gate := by
    intro pops hpops L Lsf hLp hLsfp hLs hLat hLsfLen hpAPL hLsfat
    have hs : settings.schedule pops = schedule₀ pops := if_pos hpops
    rw [hs]
    have hLat0 : ∀ j t, 0 ≤ (L j).real {t} := fun _ _ => measureReal_nonneg
    have hcolS : collisionMass Lsf ≤ 2 * cap := collisionMass_le_of_atoms Lsf hLsfat
    have hcolS0 : 0 ≤ collisionMass Lsf := tsum_nonneg fun t => sq_nonneg _
    have hNP0 : (0 : ℝ) ≤ NP := Nat.cast_nonneg _
    have hNS0 : (0 : ℝ) ≤ NS := Nat.cast_nonneg _
    refine hclust₀ pops hpops L Lsf hLp hLsfp (fun j hj => ?_) ?_ hpAPL
      (fun B hB j hj t => ?_) (fun B hB j hj t => ?_) (fun B hB => ?_)
    · refine measure_mono_null (fun p hp => ?_) (hLs j hj)
      simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_lt] at hp ⊢
      exact hp
    · refine measure_mono_null (fun v hv => ?_) hLsfLen
      simpa using hv
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hc1 := hpopc pops
      have ha : ((pops.card : ℝ) + 1) ^ 2 ≤ ((K : ℝ) + 2) ^ 2 :=
        pow_le_pow_left₀ (add_nonneg (Nat.cast_nonneg _) zero_le_one) hc1 2
      have hb := mul_le_mul ha hnp (Nat.cast_nonneg _) (by positivity)
      have hcc := mul_le_mul hb (hLat j hj t) (hLat0 j t) (by positivity)
      calc 2 ^ 21 * ((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (L j).real {t}
          = 2 ^ 21 * (((pops.card : ℝ) + 1) ^ 2 * (B.npref : ℝ) * (L j).real {t}) := by ring
        _ ≤ 2 ^ 21 * (((K : ℝ) + 2) ^ 2 * NP * (2 * cap)) :=
            mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 22 * ((K : ℝ) + 2) ^ 2 * NP * cap := by ring
        _ ≤ _ := hq1
    · have hnp : (B.npref : ℝ) ≤ NP := by
        exact_mod_cast (hsizes pops B (by rw [hs]; exact hB)).2.1
      have hc1 := hpopc pops
      have hb := mul_le_mul hc1 hnp (Nat.cast_nonneg _) (by positivity)
      have hcc := mul_le_mul hb (hLat j hj t) (hLat0 j t) (by positivity)
      calc 2 ^ 8 * ((pops.card : ℝ) + 1) * (B.npref : ℝ) * (L j).real {t}
          = 2 ^ 8 * (((pops.card : ℝ) + 1) * (B.npref : ℝ) * (L j).real {t}) := by ring
        _ ≤ 2 ^ 8 * (((K : ℝ) + 2) * NP * (2 * cap)) :=
            mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 9 * ((K : ℝ) + 2) * NP * cap := by ring
        _ ≤ _ := hq2
    · have hns : (B.nsuff : ℝ) ≤ NS := by
        rw [(hsizes pops B (by rw [hs]; exact hB)).1]
        exact_mod_cast hNSb pops hpops
      have hc1 := hpopc pops
      have hb := mul_le_mul hc1 hns (Nat.cast_nonneg _) (by positivity)
      have hcc := mul_le_mul hb hcolS hcolS0 (by positivity)
      calc 2 ^ 13 * ((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Lsf
          = 2 ^ 13 * (((pops.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Lsf) := by ring
        _ ≤ 2 ^ 13 * (((K : ℝ) + 2) * NS * (2 * cap)) :=
            mul_le_mul_of_nonneg_left hcc (by norm_num)
        _ = 2 ^ 14 * ((K : ℝ) + 2) * NS * cap := by ring
        _ ≤ _ := hq3
  -- the history's reads number at most `6 (K + 1) NP` prefixes and `NS + 1` suffixes a round
  have hcodes : ∀ (ω : Ω) (g : Fin K → RoundDraws α K T) (r : ℕ),
      ∀ code ∈ histCodes settings O P.D Dsf N ω g r,
        code.2.2.2.1.card ≤ 6 * (K + 1) * NP ∧ code.2.2.2.2.card ≤ NS + 1 := by
    intro ω g r code hcode
    obtain ⟨i, hi, hget⟩ := List.getElem_of_mem hcode
    obtain ⟨hiK, code', hget', h1, h2⟩ :=
      histCodes_getElem settings O P.D Dsf N P.length_le hN ω g r i hi
    rw [List.getElem?_eq_getElem hi, hget, Option.some.injEq] at hget'
    subst hget'
    set h := histAt settings P.D (histCodes settings O P.D Dsf N ω g r) i
    set pops := loopPopulations K h.pops.length
    have hpops : pops.card ≤ K + 1 := by
      have := Finset.card_le_univ pops
      simpa using this
    refine ⟨?_, ?_⟩
    · rw [h1]
      refine (card_roundPrefs_le _ _ _).trans ?_
      calc 3 * pops.card * ∑ B ∈ settings.schedule pops, B.npref
          ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) (hNPsum pops)
        _ = 6 * (K + 1) * NP := by ring
    · rw [h2]
      have hn : ∀ B ∈ settings.schedule pops,
          B.nsuff = poolCount η₀ pops ι εcov (δc / 2) (pAP - ms) c := fun B hB =>
        (hsizes pops B hB).1
      refine (card_roundSuffs_le _ hn _).trans (Nat.succ_le_succ ?_)
      exact hNSb pops (Finset.mem_insert_self _ _)
  -- the rounds' bad events
  set Hist : (r : ℕ) → Ω × (Fin K → RoundDraws α K T) → List (HarvestCode α) := fun r p =>
    histCodes settings O P.D Dsf N p.1 p.2 r
  set Masses : ℝ → ℝ → List (HarvestCode α) → Prop := fun a b c' =>
    (∀ j ∈ loopPopulations K c'.length,
      (loopDistributions P.D (dec c').pops j).real ↑(dec c').prefs ≤ a)
      ∧ Dsf.real ↑(dec c').suffs ≤ b
  set E1 : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | HistOK settings O P.D Dsf N p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ (dec (histCodes settings O P.D Dsf N p.1 p.2 r)).pops,
          (∀ t, ν'.real {t} ≤ cap) ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops j).real
              ↑(dec (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs ≤ m)
        ∧ Dsf.real ↑(dec (histCodes settings O P.D Dsf N p.1 p.2 r)).suffs ≤ ms
        ∧ ¬ ClusterSpec settings O P.D (513 / 512 * εcov) slack (dec
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops
          (clusterRun P.D Dsf (dec (histCodes settings O P.D Dsf N p.1 p.2 r))
            (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr}
  set E2 : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | HistOK settings O P.D Dsf N p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (dec (histCodes settings O P.D Dsf N p.1 p.2 r))
            (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ RoundSpec M O P θ (C.disagreementRate + u + mr) (C.harvestRate / 2) k
            (loopRound settings O P.D Dsf (dec (histCodes settings O P.D Dsf N p.1 p.2 r))
              (p.1, p.2 ⟨r, hr⟩))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr}
  set E3 : (r : ℕ) → r < K → Set (Ω × (Fin K → RoundDraws α K T)) := fun r hr =>
    {p | (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ (∀ ν ∈ (dec (histCodes settings O P.D Dsf N p.1 p.2 r)).pops,
          ν {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops j).real
              ↑(dec (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs ≤ 1 / 2)
        ∧ Dsf.real ↑(dec (histCodes settings O P.D Dsf N p.1 p.2 r)).suffs ≤ 1 / 2
        ∧ ¬ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr}
  have h1 : ∀ r (hr : r < K), μL.real (E1 r hr) ≤ δc + gate := fun r hr =>
    loop_cluster_round_le_persist settings O P Dsf Ls N (513 / 512 * εcov) (δc + gate) slack cap cap
      m ms pAP (by linarith) hN hm1 hms1 hclust hLs' hDat hDsfLen hpAPDsf hatom r hr
  set δ3 : ℝ := C.δ + C.δ / 8 + (shortStrings (α := α) (Fintype.card σ)).card * cross / u
    with hδ3d
  have h2 : ∀ r (hr : r < K), μL.real (E2 r hr) ≤ δ3 := fun r hr =>
    loop_round_le_persist M O hL P Dsf settings Ls NP N θ mr u cross hN hNPsum hband hk0 rfl hη1
      hcr hprem hu hy0 hmr hmr1 hmrc r hr
  have h3 : ∀ r (hr : r < K), μL.real (E3 r hr) = 0 := fun r hr => by
    rw [measureReal_def, loop_coll_null settings O P Dsf Ls N hN hLs' hDsfLen r hr,
      ENNReal.toReal_zero]
  set Ev : ℕ → Set (Ω × (Fin K → RoundDraws α K T)) := fun r =>
    if hr : r < K then (E1 r hr ∪ E2 r hr) ∪ E3 r hr else ∅ with hEvd
  have hEv : ∀ r, μL.real (Ev r) ≤ δc + gate + δ3 := by
    intro r
    simp only [hEvd]
    split_ifs with hr
    · calc μL.real ((E1 r hr ∪ E2 r hr) ∪ E3 r hr)
          ≤ μL.real (E1 r hr) + μL.real (E2 r hr) + μL.real (E3 r hr) :=
            (measureReal_union_le _ _).trans (add_le_add (measureReal_union_le _ _) le_rfl)
        _ ≤ δc + gate + δ3 + 0 :=
            add_le_add (add_le_add (h1 r hr) (h2 r hr)) (le_of_eq (h3 r hr))
        _ = δc + gate + δ3 := by ring
    · rw [measureReal_empty]
      have : 0 ≤ (shortStrings (α := α) (Fintype.card σ)).card * cross / u := by positivity
      have : 0 ≤ δc + gate + δ3 := by simp only [hδ3d]; linarith
      exact this
  -- a failing run fails one of the first `|Q| + 1` rounds' claims
  have hsub : {p : Ω × (Fin K → RoundDraws α K T) |
      let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (513 / 512 * εcov + 2 * ι + 2 * slack) (2 * ι + slack)
            (C.disagreementRate + u + mr))
        ∧ RoundsFresh settings O P.D Dsf N Ls p)}
      ⊆ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Ev r := by
    intro p hp
    by_contra hno
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists] at hno
    set Lr := loopRounds settings O P.D Dsf (persistDraws p) K with hLr
    have hspec : ∀ i, i ≤ Fintype.card σ → ∀ hi : i < Lr.length,
        (ClusterSpec settings O P.D (513 / 512 * εcov) slack (harvestsOf P.D k (Lr.take i))
            Lr[i].run
          ∧ RoundSpec M O P θ (C.disagreementRate + u + mr) (C.harvestRate / 2) k Lr[i]
          ∧ ∀ site, Random.harvestLaw Lr[i].read P.D k Lr[i].result.hypothesis site
            (shorterThan Lr[i].suffs)ᶜ = 0)
        ∧ ∃ hK : i < K, HistOK settings O P.D Dsf N p.1 p.2 i
          ∧ Returns settings O P.D Dsf (dec (histCodes settings O P.D Dsf N p.1 p.2 i))
            (p.1, (p.2 ⟨i, hK⟩).1)
          ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 i) p.2 i hK := by
      intro i
      induction i using Nat.strong_induction_on with
      | _ i ih =>
        intro hiσ hi
        obtain ⟨hK', hlen, hdec, e⟩ :=
          persist_bridge settings O P.D Dsf P.length_le hN p i hi
        have hnot : p ∉ Ev i := hno i (by omega)
        simp only [hEvd, dif_pos hK', Set.mem_union, not_or] at hnot
        obtain ⟨⟨hn1, hn2⟩, hn3⟩ := hnot
        set cH := histCodes settings O P.D Dsf N p.1 p.2 i
        have hok : HistOK settings O P.D Dsf N p.1 p.2 i :=
          histOK_of_specs settings O P.D Dsf P.length_le hN (513 / 512 * εcov) slack p i hi.le
            fun j hj hji => (ih j hji (by omega) hj).1.1
        have hpops : ∀ ν ∈ (dec cH).pops,
            (∀ t, ν.real {t} ≤ cap) ∧ ν {q | Ls ≤ q.toList.length} = 0 := by
          show ∀ ν ∈ (decodeHist P.D settings.level3.walkStart cH).pops, _
          rw [hdec, histOf_pops]
          intro ν hν
          obtain ⟨o, ho, hoν⟩ := List.mem_filterMap.1 hν
          obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ho
          simp only [List.length_take] at hj
          rw [List.getElem_take] at hoν
          have hji : j < i := by omega
          obtain ⟨⟨h4, hcls⟩, hshj⟩ := (ih j hji (by omega) (by omega)).1.2
          have hsupp := population_short P (Ls := Ls) (by omega) k _ hcls hoν
          unfold LoopRound.population at hoν
          rcases hh : (Lr[j]'(by omega)).result.ending.harvested with _ | h
          · rw [hh] at hoν; cases hoν
          rw [hh, Option.map_some, Option.some.injEq,
            Measure.restrict_eq_self_of_ae_mem (ae_iff.2 (hshj h))] at hoν
          subst hoν
          have he : (Lr[j]'(by omega)).result.ending = .harvest h := by
            revert hh
            cases (Lr[j]'(by omega)).result.ending <;> simp [RoundEnd.harvested]
          rw [he] at h4
          obtain ⟨-, hsp, hyl⟩ := h4
          have hatc : ∀ t, (popLaw (Random.harvestLaw (Lr[j]'(by omega)).read P.D k
              (Lr[j]'(by omega)).result.hypothesis h)).real {t} ≤ cap := fun t =>
            (popLaw_atom_le (by positivity : (0 : ℝ) < C.harvestRate / 2) hyl
              (Random.harvestLaw_univ_ne_top _ P.D k _ h) (le_prefixMax_of_spread hsp) t).trans
              (by rw [div_le_iff₀ (by positivity)]; linarith)
          exact ⟨hatc, hsupp⟩
        have hpl : (dec cH).pops.length = i := by simp [hdecd, decodeHist, cH, hlen]
        have hatoms : ∀ j ∈ loopPopulations K i, ∀ t,
            (loopDistributions P.D (dec cH).pops j).real {t} ≤ cap := by
          intro j hj t
          cases j with
          | none => exact hDat t
          | some i' =>
            have hi' : i'.val < (dec cH).pops.length := by
              rw [hpl]; exact lt_of_some_mem_loopPops hj
            show ((dec cH).pops.getD i' (Measure.dirac 1)).real {t} ≤ cap
            rw [List.getD_eq_getElem (hn := hi')]
            exact (hpops _ (List.getElem_mem _)).1 t
        have hiK : (i : ℝ) ≤ K := by exact_mod_cast hK'.le
        have hcardP : ((dec cH).prefs.card : ℝ) ≤ i * (6 * (K + 1) * NP) := by
          have := card_foldr_union_le (fun c' : HarvestCode α => c'.2.2.2.1) cH (6 * (K + 1) * NP)
            fun x hx => (hcodes p.1 p.2 i x hx).1
          rw [hlen] at this
          exact_mod_cast this
        have hcardS : ((dec cH).suffs.card : ℝ) ≤ i * (NS + 1) := by
          have := card_foldr_union_le (fun c' : HarvestCode α => c'.2.2.2.2) cH (NS + 1)
            fun x hx => (hcodes p.1 p.2 i x hx).2
          rw [hlen] at this
          exact_mod_cast this
        have hmP : ∀ j ∈ loopPopulations K i,
            (loopDistributions P.D (dec cH).pops j).real ↑(dec cH).prefs ≤ m := by
          intro j hj
          have : IsProbabilityMeasure (loopDistributions (K := K) P.D (dec cH).pops j) :=
            isProbabilityMeasure_loopDistributions P.D _ (decodePop_prob settings P.D cH) j
          refine (measureReal_le_card_mul _ (hatoms j hj)).trans ?_
          calc ((dec cH).prefs.card : ℝ) * cap ≤ i * (6 * (K + 1) * NP) * cap :=
                mul_le_mul_of_nonneg_right hcardP hcap.le
            _ ≤ K * (6 * (K + 1) * NP) * cap := by gcongr
            _ = 6 * K * (K + 1) * NP * cap := by ring
            _ ≤ m := hqm
        have hmS : Dsf.real ↑(dec cH).suffs ≤ ms := by
          refine (measureReal_le_card_mul _ hatom).trans ?_
          calc ((dec cH).suffs.card : ℝ) * cap ≤ i * (NS + 1) * cap :=
                mul_le_mul_of_nonneg_right hcardS hcap.le
            _ ≤ K * (NS + 1) * cap := by gcongr
            _ ≤ ms := hqms
        have hnc : NoColl settings P.D Dsf Ls cH p.2 i hK' := by
          by_contra h
          exact hn3 ⟨hlen, fun ν hν => (hpops ν hν).2, fun j hj => (hmP j hj).trans hm1,
            hmS.trans hms1, h⟩
        have hcs : ClusterSpec settings O P.D (513 / 512 * εcov) slack (dec cH).pops
            (clusterRun P.D Dsf (dec cH) (p.1, (p.2 ⟨i, hK'⟩).1)) := by
          by_contra h
          exact hn1 ⟨hok, hlen, hpops, hmP, hmS, h, hnc⟩
        have h4 : RoundSpec M O P θ (C.disagreementRate + u + mr) (C.harvestRate / 2) k
            (loopRound settings O P.D Dsf (dec cH) (p.1, p.2 ⟨i, hK'⟩)) := by
          by_contra h
          exact hn2 ⟨hok, hlen, hcs.1, h, hnc⟩
        have hret := hcs.1
        have hhist : histAt settings P.D cH i = dec cH := by
          simp only [histAt, List.take_of_length_le (le_of_eq hlen), hdecd]
          rfl
        have hsh : ∀ site, Random.harvestLaw
            (loopRound settings O P.D Dsf (dec cH) (p.1, p.2 ⟨i, hK'⟩)).read P.D k
            (loopRound settings O P.D Dsf (dec cH) (p.1, p.2 ⟨i, hK'⟩)).result.hypothesis site
            (shorterThan (loopRound settings O P.D Dsf (dec cH) (p.1, p.2 ⟨i, hK'⟩)).suffs)ᶜ
              = 0 := fun site =>
          harvestLaw_shorter P k _ h4.2 (fun v hv hv1 => by
            have hvs : v ∈ suffsOf settings P.D Dsf (histAt settings P.D cH i)
                (p.2 ⟨i, hK'⟩).1 := by
              rw [hhist]
              exact Finset.mem_erase.2 ⟨hv1, hv⟩
            have := hnc.1 ⟨i, hK'⟩ le_rfl v hvs
            omega) site
        rw [← histOf_pops, ← hdec, e]
        exact ⟨⟨hcs, h4, hsh⟩, hK', hok, hret, hnc⟩
    have hag := loop_agrees_of_specs M O hL P Dsf settings (persistDraws p) k rfl (513 / 512 * εcov)
      slack (C.disagreementRate + u + mr) (C.harvestRate / 2) hK
      (fun pops B hB => (hband pops B hB).1)
      (fun i hi hiσ => ⟨(hspec i hiσ hi).1.1, (hspec i hiσ hi).1.2.1⟩)
      fun i hi hiσ => (hspec i hiσ hi).1.2.2
    refine hp ⟨hag, fun i hi => ?_⟩
    have hle := hag.1
    unfold indecisiveLStarLoop at hle hi
    exact (hspec i (by omega) hi).2
  calc μL.real _ ≤ μL.real (⋃ r ∈ Finset.range (Fintype.card σ + 1), Ev r) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (Fintype.card σ + 1), μL.real (Ev r) := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (Fintype.card σ + 1), (δc + gate + δ3) :=
        Finset.sum_le_sum fun r _ => hEv r
    _ = (Fintype.card σ + 1) * (δc + gate + δ3) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

end OrthoDFA
